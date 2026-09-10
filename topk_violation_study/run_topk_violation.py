#!/usr/bin/env python3
"""
Top-K violation study for Qwen3.

Question:
    Does sampling a token outside the model's own Top-K candidate set
    predict eventual rollout failure?

Outputs:
    raw_rollouts.jsonl.gz
    trajectory_metrics.csv
    summary.csv
    prefix_summary.csv
    count_summary.csv
    first_position_summary.csv
    figures/*.png
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

AIME24_REVISION = "1c625e328db94ec7ef7ff169016b097c468d60b9"
OPSA_PRESET = dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=32768)
TAIL_PRESET = dict(temperature=0.7, top_p=1.0, top_k=-1, max_tokens=32768)
DEFAULT_KS = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_PREFIXES = [32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local HF checkpoint")
    p.add_argument("--dataset", required=True, choices=["aime24", "math500", "gpqa_diamond"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-prompt", type=int, default=None,
                   help="Default: AIME24=32, MATH500=8, GPQA_D=8")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-preset", choices=["tail_analysis", "opsa"], default="tail_analysis")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None, dest="sampling_top_k")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--k-values", type=str, default=",".join(map(str, DEFAULT_KS)))
    p.add_argument("--primary-k", type=int, default=10)
    p.add_argument("--prefix-tokens", type=str, default=",".join(map(str, DEFAULT_PREFIXES)))
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--prompt-batch-size", type=int, default=16)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--store-logprobs", action="store_true")
    return p.parse_args()


def csv_ints(s: str) -> list[int]:
    vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not vals or min(vals) < 1:
        raise ValueError("Values must be positive integers")
    return vals


def preprocess_gpqa(text: Any) -> str:
    if text is None:
        return " "
    return str(text).strip().replace(" [title]", ". ").replace("  ", " ")


def load_aime24() -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id="zhuzilin/aime-2024",
        filename="aime-2024.jsonl",
        repo_type="dataset",
        revision=AIME24_REVISION,
    )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            rows.append({
                "problem_id": f"aime24_{i:03d}",
                "messages": d["prompt"],
                "gold": str(d["label"]),
                "question": d["prompt"][-1]["content"],
                "task": "math",
            })
    return rows


def load_math500() -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, d in enumerate(ds):
        gold = d.get("answer")
        if gold is None:
            gold = d.get("solution", "")
        rows.append({
            "problem_id": str(d.get("unique_id", f"math500_{i:04d}")),
            "messages": [{"role": "user", "content": str(d["problem"])}],
            "gold": str(gold),
            "question": str(d["problem"]),
            "task": "math",
        })
    return rows


def load_gpqa_diamond(seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    random.seed(seed)
    rows = []
    for i, d in enumerate(ds):
        correct = preprocess_gpqa(d["Correct Answer"])
        choices = [
            preprocess_gpqa(d["Incorrect Answer 1"]),
            preprocess_gpqa(d["Incorrect Answer 2"]),
            preprocess_gpqa(d["Incorrect Answer 3"]),
            correct,
        ]
        random.shuffle(choices)
        gold_idx = choices.index(correct)
        gold = f"({chr(65 + gold_idx)})"
        q = preprocess_gpqa(d["Question"])
        prompt = (
            f"What is the correct answer to this question:{q}\n"
            "Choices:\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}\n"
            "Let's think step by step: "
        )
        rows.append({
            "problem_id": f"gpqa_diamond_{i:04d}",
            "messages": [{"role": "user", "content": prompt}],
            "gold": gold,
            "question": q,
            "task": "gpqa",
        })
    return rows


def load_benchmark(name: str, seed: int) -> list[dict[str, Any]]:
    if name == "aime24":
        return load_aime24()
    if name == "math500":
        return load_math500()
    if name == "gpqa_diamond":
        return load_gpqa_diamond(seed)
    raise ValueError(name)


def score_math(pred: str, gold: str) -> tuple[bool, str | None]:
    try:
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        pred_cfg = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
        gold_cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        g = gold.strip()
        if "$" not in g and ("\\" in g or "{" in g or "}" in g):
            g = f"${g}$"
        gp = parse(g, extraction_config=gold_cfg)
        pp = parse(pred, extraction_config=pred_cfg)
        return bool(verify(gp, pp)), None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


_GPQA_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?([A-D])\)?", re.I),
    re.compile(r"\(([A-D])\)", re.I),
    re.compile(r"\b([A-D])\b", re.I),
]


def extract_gpqa_letter(text: str) -> str | None:
    for pat in _GPQA_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    return None


def score_gpqa(pred: str, gold: str) -> tuple[bool, str | None]:
    letter = extract_gpqa_letter(pred)
    if letter is None:
        return False, "no_answer_letter"
    gold_letter = re.sub(r"[^A-D]", "", gold.upper())
    return letter == gold_letter, None


def score_response(task: str, pred: str, gold: str) -> tuple[bool, str | None]:
    return score_math(pred, gold) if task == "math" else score_gpqa(pred, gold)


def render_non_thinking(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as e:
        raise RuntimeError(
            "Tokenizer does not accept enable_thinking=False. Upgrade transformers; "
            "this study intentionally refuses to silently fall back to thinking mode."
        ) from e


def normalize_rank(rank: Any) -> int:
    r = int(rank)
    return 1 if r == 0 else r


def extract_selected_ranks_and_logps(completion) -> tuple[list[int], list[float]]:
    if completion.logprobs is None:
        raise RuntimeError("vLLM returned no logprobs")
    token_ids = list(completion.token_ids)
    if len(token_ids) != len(completion.logprobs):
        raise RuntimeError("token_ids/logprobs length mismatch")
    ranks, logps = [], []
    for tok_id, lp_dict in zip(token_ids, completion.logprobs):
        chosen = lp_dict.get(int(tok_id))
        if chosen is None:
            raise RuntimeError(f"Chosen token {tok_id} missing from vLLM logprobs")
        ranks.append(normalize_rank(chosen.rank))
        logps.append(float(chosen.logprob))
    return ranks, logps


def default_n(dataset: str) -> int:
    return {"aime24": 32, "math500": 8, "gpqa_diamond": 8}[dataset]


def resolve_sampling(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(TAIL_PRESET if args.sampling_preset == "tail_analysis" else OPSA_PRESET)
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_p is not None:
        cfg["top_p"] = args.top_p
    if args.sampling_top_k is not None:
        cfg["top_k"] = args.sampling_top_k
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    return cfg


def iter_chunks(xs: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def load_completed_problem_ids(raw_path: Path, n_samples: int) -> set[str]:
    if not raw_path.exists():
        return set()
    counts: dict[str, set[int]] = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            counts.setdefault(d["problem_id"], set()).add(int(d["sample_id"]))
    return {pid for pid, ids in counts.items() if len(ids) >= n_samples}


def run_rollouts(args: argparse.Namespace, rows: list[dict[str, Any]], out_dir: Path) -> Path:
    sampling = resolve_sampling(args)
    n_samples = args.samples_per_prompt or default_n(args.dataset)
    ks = csv_ints(args.k_values)
    if sampling["top_k"] > 0 and max(ks) >= sampling["top_k"]:
        warnings.warn(
            f"Sampling top_k={sampling['top_k']} censors the tail; K >= {sampling['top_k']} "
            "violations will be mechanically absent. Use tail_analysis for the core study."
        )
    if sampling["top_p"] < 1.0:
        warnings.warn(
            f"Sampling top_p={sampling['top_p']} also censors the tail. "
            "Use tail_analysis for the core study."
        )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    # vLLM always returns the chosen token and its exact vocabulary rank when
    # logprobs is requested. logprobs=1 therefore avoids large top-N payloads.
    sp = SamplingParams(
        n=n_samples,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=0.0,
        max_tokens=sampling["max_tokens"],
        logprobs=1,
        seed=args.seed,
    )

    raw_path = out_dir / "raw_rollouts.jsonl.gz"
    completed = load_completed_problem_ids(raw_path, n_samples) if args.resume else set()
    mode = "at" if args.resume and raw_path.exists() else "wt"
    pending = [r for r in rows if r["problem_id"] not in completed]
    print(f"[rollout] prompts={len(rows)} complete={len(completed)} pending={len(pending)}")
    print(f"[rollout] model={args.model}; sampling={sampling}; n={n_samples}; non-thinking=True")

    with gzip.open(raw_path, mode, encoding="utf-8") as wf:
        for chunk in iter_chunks(pending, args.prompt_batch_size):
            prompts = [render_non_thinking(tok, r["messages"]) for r in chunk]
            req_outputs = llm.generate(prompts, sp, use_tqdm=True)
            for row, req in zip(chunk, req_outputs):
                for sample_id, comp in enumerate(req.outputs):
                    ranks, logps = extract_selected_ranks_and_logps(comp)
                    correct, score_error = score_response(row["task"], comp.text, row["gold"])
                    rec = {
                        "model": args.model,
                        "dataset": args.dataset,
                        "problem_id": row["problem_id"],
                        "sample_id": sample_id,
                        "correct": int(correct),
                        "score_error": score_error,
                        "gold": row["gold"],
                        "num_tokens": len(ranks),
                        "finish_reason": getattr(comp, "finish_reason", None),
                        "token_ranks": ranks,
                        "response": comp.text,
                        "sampling": sampling,
                        "non_thinking": True,
                    }
                    if args.store_logprobs:
                        rec["sampled_token_logprobs"] = logps
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()
    return raw_path


def read_raw(raw_path: Path) -> list[dict[str, Any]]:
    out = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or reps <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sims = np.empty(reps, dtype=float)
    n = len(values)
    for b in range(reps):
        sims[b] = np.mean(values[rng.integers(0, n, size=n)])
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return float(lo), float(hi)


def within_problem_effect(df: pd.DataFrame, group_col: str) -> tuple[float, int, np.ndarray]:
    diffs = []
    for _, g in df.groupby("problem_id"):
        a = g[g[group_col] == 1]["correct"]
        b = g[g[group_col] == 0]["correct"]
        if len(a) and len(b):
            diffs.append(float(a.mean() - b.mean()))
    arr = np.asarray(diffs, dtype=float)
    return (float(arr.mean()) if len(arr) else float("nan"), len(arr), arr)


def group_effect_summary(df: pd.DataFrame, group_col: str, reps: int, seed: int) -> dict[str, Any]:
    pos = df[df[group_col] == 1]
    neg = df[df[group_col] == 0]
    acc_pos = float(pos["correct"].mean()) if len(pos) else float("nan")
    acc_neg = float(neg["correct"].mean()) if len(neg) else float("nan")
    raw_delta = acc_pos - acc_neg if len(pos) and len(neg) else float("nan")
    fe, n_paired, per_problem_diffs = within_problem_effect(df, group_col)
    lo, hi = bootstrap_mean_ci(per_problem_diffs, reps, seed)
    fail_pos = 1.0 - acc_pos if len(pos) else float("nan")
    fail_neg = 1.0 - acc_neg if len(neg) else float("nan")
    fail_rr = fail_pos / fail_neg if np.isfinite(fail_pos) and np.isfinite(fail_neg) and fail_neg > 0 else float("nan")
    return {
        "n_group1": len(pos),
        "n_group0": len(neg),
        "acc_group1": acc_pos,
        "acc_group0": acc_neg,
        "raw_acc_delta_group1_minus_group0": raw_delta,
        "failure_risk_ratio_group1_over_group0": fail_rr,
        "within_problem_acc_delta": fe,
        "within_problem_ci_low": lo,
        "within_problem_ci_high": hi,
        "n_paired_problems": n_paired,
    }


def build_trajectory_metrics(raw: list[dict[str, Any]], ks: list[int], prefixes: list[int]) -> pd.DataFrame:
    rows = []
    for rec in raw:
        ranks = np.asarray(rec["token_ranks"], dtype=np.int64)
        T = len(ranks)
        for k in ks:
            mask = ranks > k
            idx = np.flatnonzero(mask)
            first = int(idx[0] + 1) if len(idx) else np.nan
            r = {
                "model": rec["model"], "dataset": rec["dataset"],
                "problem_id": rec["problem_id"], "sample_id": int(rec["sample_id"]),
                "correct": int(rec["correct"]), "num_tokens": T, "K": k,
                "violation_count": int(mask.sum()),
                "violation_rate": float(mask.mean()) if T else 0.0,
                "any_violation": int(bool(mask.any())),
                "first_violation_pos": first,
                "first_violation_frac": float(first / T) if T and len(idx) else np.nan,
            }
            for n in prefixes:
                r[f"eligible_prefix_{n}"] = int(T >= n)
                r[f"violation_prefix_{n}"] = int(bool((ranks[:n] > k).any())) if T >= n else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_all(tm: pd.DataFrame, prefixes: list[int], primary_k: int, reps: int, seed: int):
    summary_rows = []
    for (model, dataset, k), g in tm.groupby(["model", "dataset", "K"]):
        d = {
            "model": model, "dataset": dataset, "K": int(k), "n_rollouts": len(g),
            "overall_accuracy": float(g["correct"].mean()),
            "mean_tokens": float(g["num_tokens"].mean()),
            "mean_violation_count": float(g["violation_count"].mean()),
            "mean_violation_rate": float(g["violation_rate"].mean()),
            "fraction_with_any_violation": float(g["any_violation"].mean()),
            "median_first_violation_pos": float(g["first_violation_pos"].median()) if g["first_violation_pos"].notna().any() else np.nan,
        }
        d.update(group_effect_summary(g, "any_violation", reps, seed + int(k)))
        summary_rows.append(d)
    summary = pd.DataFrame(summary_rows)

    prefix_rows = []
    for (model, dataset, k), g0 in tm.groupby(["model", "dataset", "K"]):
        for n in prefixes:
            col = f"violation_prefix_{n}"
            g = g0[g0[f"eligible_prefix_{n}"] == 1].copy()
            if not len(g):
                continue
            g[col] = g[col].astype(int)
            d = {
                "model": model, "dataset": dataset, "K": int(k), "prefix_tokens": int(n),
                "n_eligible_rollouts": len(g), "overall_accuracy": float(g["correct"].mean()),
                "fraction_prefix_violation": float(g[col].mean()),
            }
            d.update(group_effect_summary(g, col, reps, seed + int(k) * 1000 + n))
            prefix_rows.append(d)
    prefix_summary = pd.DataFrame(prefix_rows)

    gp = tm[tm["K"] == primary_k].copy()
    if len(gp):
        bins = [-0.5, 0.5, 1.5, 3.5, 7.5, np.inf]
        labels = ["0", "1", "2-3", "4-7", "8+"]
        gp["count_bin"] = pd.cut(gp["violation_count"], bins=bins, labels=labels)
        count_summary = (gp.groupby(["model", "dataset", "count_bin"], observed=False)
                         .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_tokens=("num_tokens", "mean"))
                         .reset_index())

        def pos_bin(row):
            if row["any_violation"] == 0:
                return "No violation"
            x = row["first_violation_frac"]
            if x <= 0.25: return "0-25%"
            if x <= 0.50: return "25-50%"
            if x <= 0.75: return "50-75%"
            return "75-100%"

        gp["first_pos_bin"] = gp.apply(pos_bin, axis=1)
        order = ["No violation", "0-25%", "25-50%", "50-75%", "75-100%"]
        gp["first_pos_bin"] = pd.Categorical(gp["first_pos_bin"], categories=order, ordered=True)
        first_summary = (gp.groupby(["model", "dataset", "first_pos_bin"], observed=False)
                         .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_tokens=("num_tokens", "mean"))
                         .reset_index())
    else:
        count_summary, first_summary = pd.DataFrame(), pd.DataFrame()
    return summary, prefix_summary, count_summary, first_summary


def make_plots(summary, prefix_summary, count_summary, first_summary, primary_k: int, fig_dir: Path):
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

    for (model, dataset), g in summary.groupby(["model", "dataset"]):
        g = g.sort_values("K")
        tag = f"{safe(model)}__{dataset}"

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(g["K"], g["acc_group0"], marker="o", label="No Top-K violation")
        ax.plot(g["K"], g["acc_group1"], marker="o", label="≥1 Top-K violation")
        ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("Final-answer accuracy")
        ax.set_title(f"Does a Top-K violation predict failure?\n{model} · {dataset}")
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / f"{tag}__accuracy_by_violation.png", dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        y = 100.0 * g["within_problem_acc_delta"]
        lo = 100.0 * g["within_problem_ci_low"]; hi = 100.0 * g["within_problem_ci_high"]
        ax.plot(g["K"], y, marker="o"); ax.fill_between(g["K"], lo, hi, alpha=0.18)
        ax.axhline(0.0, linewidth=1.0); ax.set_xscale("log"); ax.set_xlabel("K")
        ax.set_ylabel("Within-question Δ accuracy (pp)\nviolation − no violation")
        ax.set_title(f"Question-controlled predictive effect\n{model} · {dataset}")
        ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(fig_dir / f"{tag}__within_problem_effect.png", dpi=220); plt.close(fig)

        pg = prefix_summary[(prefix_summary["model"] == model) & (prefix_summary["dataset"] == dataset) & (prefix_summary["K"] == primary_k)].sort_values("prefix_tokens")
        if len(pg):
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            y = 100.0 * pg["within_problem_acc_delta"]
            lo = 100.0 * pg["within_problem_ci_low"]; hi = 100.0 * pg["within_problem_ci_high"]
            ax.plot(pg["prefix_tokens"], y, marker="o"); ax.fill_between(pg["prefix_tokens"], lo, hi, alpha=0.18)
            ax.axhline(0.0, linewidth=1.0); ax.set_xlabel("Observed prefix length (tokens)")
            ax.set_ylabel("Within-question Δ accuracy (pp)\nprefix violation − no prefix violation")
            ax.set_title(f"Early Top-{primary_k} violation predicts final outcome\n{model} · {dataset}")
            ax.grid(alpha=0.25); fig.tight_layout()
            fig.savefig(fig_dir / f"{tag}__prefix_predictiveness_K{primary_k}.png", dpi=220); plt.close(fig)

        cg = count_summary[(count_summary["model"] == model) & (count_summary["dataset"] == dataset)]
        if len(cg):
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.bar(cg["count_bin"].astype(str), cg["accuracy"])
            ax.set_xlabel(f"Number of Top-{primary_k} violations"); ax.set_ylabel("Final-answer accuracy")
            ax.set_title(f"Failure vs. violation count\n{model} · {dataset}")
            ax.grid(axis="y", alpha=0.25); fig.tight_layout()
            fig.savefig(fig_dir / f"{tag}__count_K{primary_k}.png", dpi=220); plt.close(fig)

        fg = first_summary[(first_summary["model"] == model) & (first_summary["dataset"] == dataset)]
        if len(fg):
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.bar(fg["first_pos_bin"].astype(str), fg["accuracy"])
            ax.set_xlabel(f"Position of first Top-{primary_k} violation"); ax.set_ylabel("Final-answer accuracy")
            ax.set_title(f"Failure vs. first violation position\n{model} · {dataset}")
            ax.grid(axis="y", alpha=0.25); fig.tight_layout()
            fig.savefig(fig_dir / f"{tag}__first_position_K{primary_k}.png", dpi=220); plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ks = csv_ints(args.k_values); prefixes = csv_ints(args.prefix_tokens)
    if args.primary_k not in ks:
        ks = sorted(set(ks + [args.primary_k]))

    benchmark = load_benchmark(args.dataset, args.seed)
    if args.limit is not None:
        benchmark = benchmark[:args.limit]

    metadata = {
        "model": args.model, "dataset": args.dataset, "n_prompts": len(benchmark),
        "samples_per_prompt": args.samples_per_prompt or default_n(args.dataset),
        "seed": args.seed, "sampling_preset": args.sampling_preset,
        "sampling": resolve_sampling(args), "k_values": ks, "primary_k": args.primary_k,
        "prefix_tokens": prefixes, "non_thinking": True,
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    raw_path = run_rollouts(args, benchmark, out_dir)
    raw = read_raw(raw_path)
    print(f"[analysis] loaded {len(raw)} rollouts")
    tm = build_trajectory_metrics(raw, ks, prefixes)
    tm.to_csv(out_dir / "trajectory_metrics.csv", index=False)
    summary, prefix_summary, count_summary, first_summary = summarize_all(
        tm, prefixes, args.primary_k, args.bootstrap_reps, args.seed
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    prefix_summary.to_csv(out_dir / "prefix_summary.csv", index=False)
    count_summary.to_csv(out_dir / "count_summary.csv", index=False)
    first_summary.to_csv(out_dir / "first_position_summary.csv", index=False)

    print("\n=== Main summary ===")
    cols = ["K", "n_rollouts", "overall_accuracy", "fraction_with_any_violation",
            "acc_group0", "acc_group1", "within_problem_acc_delta",
            "within_problem_ci_low", "within_problem_ci_high", "n_paired_problems"]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary[cols].to_string(index=False))

    if not args.no_plots:
        make_plots(summary, prefix_summary, count_summary, first_summary, args.primary_k, out_dir / "figures")
        print(f"[done] figures: {out_dir / 'figures'}")
    print(f"[done] results: {out_dir}")


if __name__ == "__main__":
    main()
