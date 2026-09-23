#!/usr/bin/env python
"""
Minimal Stage-1 smoke test for OPD log-probability probing.

Run from the repository root:

    export PYTHONPATH=$PWD/src
    python stage1_smoke_test.py

Optional:
    python stage1_smoke_test.py \
        --student Qwen/Qwen3-1.7B \
        --teacher Qwen/Qwen3-4B \
        --max-new-tokens 128 \
        --output-dir runs/stage1_smoke

What this tests:
  1. CUDA / BF16 availability
  2. student + teacher model loading
  3. tokenizer compatibility
  4. one student rollout
  5. student teacher-forced token scoring
  6. teacher teacher-forced token scoring
  7. p_s, p_t, log-ratio and gradient-proxy computation
  8. Parquet output
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from opd_probe.modeling import (
    assert_same_tokenizer,
    build_prompt_ids,
    generate_rollout_batch,
    load_model,
    load_tokenizer,
    score_generated_tokens,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="Qwen/Qwen3-1.7B")
    p.add_argument("--teacher", default="Qwen/Qwen3-4B")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output-dir", default="runs/stage1_smoke")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-implementation", default="sdpa")
    return p.parse_args()


def section(name: str):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Environment
    # ------------------------------------------------------------------
    section("0. Environment")

    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() == False")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU memory free/total: {free_b/2**30:.1f}/{total_b/2**30:.1f} GiB")

    # ------------------------------------------------------------------
    # 1. Tokenizers
    # ------------------------------------------------------------------
    section("1. Load tokenizers")

    print("Student:", args.student)
    student_tok = load_tokenizer(args.student)

    print("Teacher:", args.teacher)
    teacher_tok = load_tokenizer(args.teacher)

    assert_same_tokenizer(student_tok, teacher_tok)
    print(f"Tokenizer compatibility: PASS (vocab size = {len(student_tok)})")

    # ------------------------------------------------------------------
    # 2. Models
    # ------------------------------------------------------------------
    section("2. Load models")

    print("Loading student...")
    student = load_model(
        args.student,
        "bfloat16",
        args.device,
        args.attn_implementation,
    )
    print(
        "Student parameters:",
        f"{sum(p.numel() for p in student.parameters()) / 1e9:.2f}B",
    )

    print("Loading teacher...")
    teacher = load_model(
        args.teacher,
        "bfloat16",
        args.device,
        args.attn_implementation,
    )
    print(
        "Teacher parameters:",
        f"{sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B",
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        print(f"CUDA memory allocated/reserved: {allocated:.1f}/{reserved:.1f} GiB")

    # ------------------------------------------------------------------
    # 3. One simple prompt
    # ------------------------------------------------------------------
    section("3. Build prompt")

    problem = (
        "Let x be a real number satisfying x^2 - 5x + 6 = 0. "
        "Find the sum of all possible values of x."
    )

    prompt_cfg = {
        "enable_thinking": True,
        "system": (
            "You are a careful mathematical reasoner. "
            "Solve the problem step by step and give the final answer clearly."
        ),
        "user_template": "{problem}",
    }

    prompt_ids, messages = build_prompt_ids(
        student_tok,
        problem,
        prompt_cfg,
        args.device,
    )
    prompt_len = int(prompt_ids.shape[1])

    print("Prompt token length:", prompt_len)
    print("Problem:", problem)

    # ------------------------------------------------------------------
    # 4. One student rollout
    # ------------------------------------------------------------------
    section("4. Student rollout")

    generation_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,       # no top-k truncation
    }

    generated_batch = generate_rollout_batch(
        student,
        student_tok,
        prompt_ids,
        batch_size=1,
        generation_cfg=generation_cfg,
        seed=args.seed,
    )

    generated_ids = generated_batch[0]
    response_text = student_tok.decode(
        generated_ids.tolist(),
        skip_special_tokens=False,
    )

    print("Generated token count:", len(generated_ids))
    print("\n--- Response ---")
    print(response_text)
    print("--- End response ---")

    if len(generated_ids) == 0:
        raise RuntimeError("Student generated zero tokens.")

    # ------------------------------------------------------------------
    # 5. Score exact same sampled tokens under student and teacher
    # ------------------------------------------------------------------
    section("5. Teacher-forced scoring")

    full_ids_cpu = torch.cat(
        [prompt_ids[0].detach().cpu(), generated_ids.to(torch.long).cpu()],
        dim=0,
    )

    print("Scoring with student...")
    s = score_generated_tokens(
        student,
        full_ids_cpu,
        prompt_len,
        device=args.device,
        logit_chunk_size=64,
        compute_entropy=True,
        compute_exact_rank=True,
    )

    print("Scoring with teacher...")
    t = score_generated_tokens(
        teacher,
        full_ids_cpu,
        prompt_len,
        device=args.device,
        logit_chunk_size=64,
        compute_entropy=True,
        compute_exact_rank=True,
    )

    n = len(generated_ids)
    assert len(s.prob) == n
    assert len(t.prob) == n

    # ------------------------------------------------------------------
    # 6. Build token table
    # ------------------------------------------------------------------
    section("6. Build token-level statistics")

    rows = []
    token_ids = generated_ids.tolist()

    for pos, token_id in enumerate(token_ids):
        ps = float(s.prob[pos])
        pt = float(t.prob[pos])
        lps = float(s.logprob[pos])
        lpt = float(t.logprob[pos])

        prob_diff = pt - ps
        advantage = lpt - lps
        grad_proxy = advantage * (1.0 - ps)

        rows.append(
            {
                "token_position": pos,
                "token_id": int(token_id),
                "token_text": student_tok.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                ),

                "student_prob": ps,
                "student_logprob": lps,
                "student_entropy": float(s.entropy[pos]),
                "student_rank": int(s.rank[pos]),
                "student_top1_prob": float(s.top1_prob[pos]),

                "teacher_prob": pt,
                "teacher_logprob": lpt,
                "teacher_entropy": float(t.entropy[pos]),
                "teacher_rank": int(t.rank[pos]),
                "teacher_top1_prob": float(t.top1_prob[pos]),

                "prob_diff_signed": prob_diff,
                "prob_diff_abs": abs(prob_diff),

                # OPD token-level advantage:
                # A_t = log p_T(y_t) - log p_S(y_t)
                "logratio_signed": advantage,
                "logratio_abs": abs(advantage),

                # Sampled-token selected-logit gradient proxy:
                # A_t * (1 - p_S(y_t))
                "grad_proxy_signed": grad_proxy,
                "grad_proxy_abs": abs(grad_proxy),
            }
        )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    section("7. Sanity checks")

    numeric_cols = [
        "student_prob",
        "student_logprob",
        "teacher_prob",
        "teacher_logprob",
        "prob_diff_signed",
        "logratio_signed",
        "grad_proxy_signed",
    ]

    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise RuntimeError("Found NaN/Inf in core probability statistics.")

    if not ((df["student_prob"] >= 0) & (df["student_prob"] <= 1)).all():
        raise RuntimeError("student_prob is outside [0, 1].")

    if not ((df["teacher_prob"] >= 0) & (df["teacher_prob"] <= 1)).all():
        raise RuntimeError("teacher_prob is outside [0, 1].")

    # log(prob) consistency check
    valid_s = df["student_prob"] > 0
    err_s = np.max(
        np.abs(
            np.log(df.loc[valid_s, "student_prob"].to_numpy())
            - df.loc[valid_s, "student_logprob"].to_numpy()
        )
    )

    valid_t = df["teacher_prob"] > 0
    err_t = np.max(
        np.abs(
            np.log(df.loc[valid_t, "teacher_prob"].to_numpy())
            - df.loc[valid_t, "teacher_logprob"].to_numpy()
        )
    )

    print(f"max |log(student_prob) - student_logprob| = {err_s:.3e}")
    print(f"max |log(teacher_prob) - teacher_logprob| = {err_t:.3e}")

    print("\nStudent probability summary:")
    print(df["student_prob"].describe().to_string())

    print("\n|p_teacher - p_student| summary:")
    print(df["prob_diff_abs"].describe().to_string())

    print("\n|log p_teacher - log p_student| summary:")
    print(df["logratio_abs"].describe().to_string())

    print("\n|A| * (1 - p_student) summary:")
    print(df["grad_proxy_abs"].describe().to_string())

    # Show the most extreme low-logp tokens.
    show_cols = [
        "token_position",
        "token_text",
        "student_prob",
        "teacher_prob",
        "prob_diff_signed",
        "logratio_signed",
        "grad_proxy_abs",
        "student_entropy",
        "student_rank",
    ]

    print("\n10 lowest-student-probability sampled tokens:")
    print(
        df.nsmallest(10, "student_prob")[show_cols]
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    section("8. Save outputs")

    parquet_path = out_dir / "tokens.parquet"
    csv_path = out_dir / "tokens_preview.csv"
    metadata_path = out_dir / "metadata.json"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    metadata = {
        "student": args.student,
        "teacher": args.teacher,
        "seed": args.seed,
        "problem": problem,
        "prompt_messages": messages,
        "prompt_len_tokens": prompt_len,
        "response_len_tokens": len(generated_ids),
        "response_text": response_text,
        "generation": generation_cfg,
        "num_token_rows": len(df),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved:", parquet_path)
    print("Saved:", csv_path)
    print("Saved:", metadata_path)

    # Confirm Parquet can actually be read back.
    df2 = pd.read_parquet(parquet_path)
    assert len(df2) == len(df)

    section("SMOKE TEST PASSED")
    print(
        f"Successfully generated and scored {len(df)} tokens with\n"
        f"  student = {args.student}\n"
        f"  teacher = {args.teacher}\n"
        f"Output directory: {out_dir}"
    )


if __name__ == "__main__":
    main()
