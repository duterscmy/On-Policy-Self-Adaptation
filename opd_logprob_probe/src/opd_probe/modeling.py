from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name.lower() not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name.lower()]


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def assert_same_tokenizer(student_tok, teacher_tok):
    if len(student_tok) != len(teacher_tok):
        raise ValueError(
            f"Tokenizer sizes differ: student={len(student_tok)}, teacher={len(teacher_tok)}. "
            "This experiment requires identical token IDs."
        )
    if student_tok.get_vocab() != teacher_tok.get_vocab():
        raise ValueError(
            "Student and teacher vocabularies/token IDs are not identical. "
            "Use tokenizer-compatible models for this exact-token analysis."
        )


def load_model(model_name: str, dtype_name: str, device: str, attn_implementation: str):
    kwargs = {
        "torch_dtype": resolve_dtype(dtype_name),
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model.to(device)


def build_prompt_ids(tokenizer, problem: str, prompt_cfg: dict, device: str):
    messages = []
    if prompt_cfg.get("system"):
        messages.append({"role": "system", "content": prompt_cfg["system"]})
    messages.append({
        "role": "user",
        "content": prompt_cfg.get("user_template", "{problem}").format(problem=problem),
    })

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if "enable_thinking" in prompt_cfg:
        kwargs["enable_thinking"] = bool(prompt_cfg["enable_thinking"])
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device), messages


def generate_rollout_batch(model, tokenizer, prompt_ids, batch_size: int, generation_cfg: dict, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_ids = prompt_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        seqs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(generation_cfg["max_new_tokens"]),
            do_sample=bool(generation_cfg.get("do_sample", True)),
            temperature=float(generation_cfg.get("temperature", 1.0)),
            top_p=float(generation_cfg.get("top_p", 1.0)),
            top_k=int(generation_cfg.get("top_k", 0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_len = prompt_ids.shape[1]
    generated = [row[prompt_len:].detach().cpu() for row in seqs]
    del seqs, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated


@dataclass
class SequenceScores:
    logprob: np.ndarray
    prob: np.ndarray
    entropy: np.ndarray
    rank: np.ndarray
    top1_prob: np.ndarray


def score_generated_tokens(
    model,
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    device: str,
    logit_chunk_size: int = 128,
    compute_entropy: bool = True,
    compute_exact_rank: bool = True,
) -> SequenceScores:
    # Teacher-force the complete prompt+response. Logits at position i-1 predict token i.
    full_ids_cpu = full_ids_cpu.to(torch.long).flatten().cpu()
    n_total = int(full_ids_cpu.numel())
    n_gen = n_total - prompt_len
    if n_gen < 0:
        raise ValueError("prompt_len exceeds sequence length")
    if n_gen == 0:
        ef = np.empty((0,), dtype=np.float32)
        ei = np.empty((0,), dtype=np.int32)
        return SequenceScores(ef, ef, ef, ei, ef)

    input_ids = full_ids_cpu.unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0]

    pred_logits = logits[prompt_len - 1 : n_total - 1]
    targets = input_ids[0, prompt_len:n_total]

    lp_parts, p_parts, h_parts, r_parts, t1_parts = [], [], [], [], []
    for start in range(0, n_gen, logit_chunk_size):
        end = min(start + logit_chunk_size, n_gen)
        z = pred_logits[start:end].float()
        y = targets[start:end]
        lse = torch.logsumexp(z, dim=-1)
        selected_logits = z.gather(1, y.unsqueeze(1)).squeeze(1)
        selected_logp = selected_logits - lse
        selected_p = torch.exp(selected_logp)

        if compute_entropy:
            p = torch.softmax(z, dim=-1)
            entropy = lse - (p * z).sum(dim=-1)
        else:
            entropy = torch.full_like(selected_logp, float("nan"))

        if compute_exact_rank:
            rank = (z > selected_logits.unsqueeze(1)).sum(dim=-1) + 1
        else:
            rank = torch.full_like(y, -1)

        top1_prob = torch.exp(z.max(dim=-1).values - lse)
        lp_parts.append(selected_logp.cpu())
        p_parts.append(selected_p.cpu())
        h_parts.append(entropy.cpu())
        r_parts.append(rank.cpu())
        t1_parts.append(top1_prob.cpu())

    result = SequenceScores(
        logprob=torch.cat(lp_parts).numpy().astype(np.float32),
        prob=torch.cat(p_parts).numpy().astype(np.float32),
        entropy=torch.cat(h_parts).numpy().astype(np.float32),
        rank=torch.cat(r_parts).numpy().astype(np.int32),
        top1_prob=torch.cat(t1_parts).numpy().astype(np.float32),
    )
    del outputs, logits, pred_logits, targets, input_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
