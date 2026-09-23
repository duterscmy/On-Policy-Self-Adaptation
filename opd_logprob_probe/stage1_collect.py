#!/usr/bin/env python
from __future__ import annotations

import argparse, json, platform, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from opd_probe.data import load_problem_subset
from opd_probe.modeling import (
    assert_same_tokenizer, build_prompt_ids, generate_rollout_batch,
    load_model, load_tokenizer, score_generated_tokens,
)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def token_rows(problem, rollout_idx, generated_ids, tokenizer, s, t):
    rows = []
    for j, tid in enumerate(generated_ids):
        sp, tp = float(s.prob[j]), float(t.prob[j])
        slp, tlp = float(s.logprob[j]), float(t.logprob[j])
        logratio = tlp - slp
        pdiff = tp - sp
        grad = logratio * (1.0 - sp)
        rows.append({
            "problem_local_index": int(problem.local_index),
            "problem_source_index": int(problem.source_index),
            "problem_id": str(problem.problem_id),
            "rollout_id": int(rollout_idx),
            "token_position": int(j),
            "token_id": int(tid),
            "token_text": tokenizer.decode([int(tid)], skip_special_tokens=False),
            "student_prob": sp,
            "student_logprob": slp,
            "student_entropy": float(s.entropy[j]),
            "student_rank": int(s.rank[j]),
            "student_top1_prob": float(s.top1_prob[j]),
            "teacher_prob": tp,
            "teacher_logprob": tlp,
            "teacher_entropy": float(t.entropy[j]),
            "teacher_rank": int(t.rank[j]),
            "teacher_top1_prob": float(t.top1_prob[j]),
            "prob_diff_signed": pdiff,
            "prob_diff_abs": abs(pdiff),
            "logratio_signed": logratio,
            "logratio_abs": abs(logratio),
            "grad_proxy_signed": grad,
            "grad_proxy_abs": abs(grad),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 1234))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    out = Path(cfg["output_dir"])
    token_dir, rollout_dir = out / "tokens", out / "rollouts"
    token_dir.mkdir(parents=True, exist_ok=True); rollout_dir.mkdir(parents=True, exist_ok=True)

    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), "bf16:", torch.cuda.is_bf16_supported())

    device = cfg["models"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    problems = load_problem_subset(cfg["dataset"], seed)
    write_json(out / "selected_problems.json", [
        {"local_index": p.local_index, "source_index": p.source_index, "problem_id": p.problem_id,
         "problem": p.problem, "answer": p.answer, "solution": p.solution}
        for p in problems
    ])

    sname, tname = cfg["models"]["student"], cfg["models"]["teacher"]
    stok, ttok = load_tokenizer(sname), load_tokenizer(tname)
    assert_same_tokenizer(stok, ttok)
    print("Tokenizer compatibility: OK")

    print("Loading student", sname)
    student = load_model(sname, cfg["models"].get("dtype", "bfloat16"), device, cfg["models"].get("attn_implementation", "sdpa"))
    print("Loading teacher", tname)
    teacher = load_model(tname, cfg["models"].get("dtype", "bfloat16"), device, cfg["models"].get("attn_implementation", "sdpa"))

    write_json(out / "manifest.json", {
        "experiment_name": cfg.get("experiment_name"), "config": cfg,
        "student_model": sname, "teacher_model": tname,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "platform": platform.platform(), "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "created_unix_time": time.time(),
    })

    gen_cfg, score_cfg = cfg["generation"], cfg["scoring"]
    n_rollouts = int(gen_cfg["num_rollouts_per_problem"])
    batch_size = int(gen_cfg.get("rollout_batch_size", 1))

    pbar = tqdm(problems, desc="Problems")
    for problem in pbar:
        prompt_ids, messages = build_prompt_ids(stok, problem.problem, cfg["prompt"], device)
        prompt_len = int(prompt_ids.shape[1])
        missing = [r for r in range(n_rollouts)
                   if not (token_dir / f"p{problem.local_index:04d}__r{r:03d}.parquet").exists()]
        for cursor in range(0, len(missing), batch_size):
            rids = missing[cursor:cursor + batch_size]
            if not rids: continue
            batch_seed = seed + problem.local_index * 100_000 + rids[0]
            gens = generate_rollout_batch(student, stok, prompt_ids, len(rids), gen_cfg, batch_seed)
            for rid, gen_ids in zip(rids, gens):
                stem = f"p{problem.local_index:04d}__r{rid:03d}"
                gen_ids = gen_ids.to(torch.long).flatten().cpu()
                full_ids = torch.cat([prompt_ids[0].detach().cpu(), gen_ids])
                common = dict(
                    full_ids_cpu=full_ids, prompt_len=prompt_len, device=device,
                    logit_chunk_size=int(score_cfg.get("logit_chunk_size", 128)),
                    compute_entropy=bool(score_cfg.get("compute_entropy", True)),
                    compute_exact_rank=bool(score_cfg.get("compute_exact_rank", True)),
                )
                ss = score_generated_tokens(student, **common)
                ts = score_generated_tokens(teacher, **common)
                df = pd.DataFrame(token_rows(problem, rid, gen_ids.tolist(), stok, ss, ts))
                path = token_dir / f"{stem}.parquet"
                tmp = token_dir / f"{stem}.tmp.parquet"
                df.to_parquet(tmp, index=False); tmp.replace(path)
                write_json(rollout_dir / f"{stem}.json", {
                    "problem_local_index": problem.local_index, "problem_source_index": problem.source_index,
                    "problem_id": problem.problem_id, "problem": problem.problem,
                    "answer": problem.answer, "solution": problem.solution,
                    "rollout_id": rid, "prompt_messages": messages,
                    "prompt_len_tokens": prompt_len, "response_len_tokens": len(gen_ids),
                    "response_text": stok.decode(gen_ids.tolist(), skip_special_tokens=False),
                    "generation": gen_cfg, "student_model": sname, "teacher_model": tname,
                })
                pbar.set_postfix(problem=problem.local_index, rollout=rid, tokens=len(gen_ids))

    print("Done:", out)
    print("Token shards:", len(list(token_dir.glob("*.parquet"))))


if __name__ == "__main__":
    main()
