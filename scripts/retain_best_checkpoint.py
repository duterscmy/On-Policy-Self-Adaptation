#!/usr/bin/env python3
"""Retain only the best-eval and final Megatron checkpoints for one run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


CHECKPOINT_RE = re.compile(r"iter_(\d{7})$")
EVAL_RE = re.compile(r"eval (\d+): \{(.*)\}")


def checkpoint_steps(save_dir: Path) -> dict[int, Path]:
    checkpoints = {}
    for path in save_dir.glob("iter_*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints[int(match.group(1))] = path
    return checkpoints


def metric_by_step(log_file: Path, metric: str) -> dict[int, float]:
    metric_re = re.compile(rf"['\"]{re.escape(metric)}['\"]:\s*([-+0-9.eE]+)")
    scores = {}
    with log_file.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            eval_match = EVAL_RE.search(line)
            if not eval_match:
                continue
            metric_match = metric_re.search(eval_match.group(2))
            if metric_match:
                scores[int(eval_match.group(1))] = float(metric_match.group(1))
    return scores


def retention_plan(save_dir: Path, log_file: Path, metric: str) -> dict:
    checkpoints = checkpoint_steps(save_dir)
    if not checkpoints:
        raise RuntimeError(f"no iter_XXXXXXX checkpoints found in {save_dir}")

    scores = metric_by_step(log_file, metric)
    scored_checkpoints = {step: scores[step] for step in checkpoints.keys() & scores.keys()}
    if not scored_checkpoints:
        raise RuntimeError(f"no checkpoint-aligned {metric!r} metrics found in {log_file}")

    # Prefer the later checkpoint when several evaluations share the best score.
    best_step = max(scored_checkpoints, key=lambda step: (scored_checkpoints[step], step))
    last_step = max(checkpoints)
    # All-zero evaluation histories commonly indicate a scorer-routing error.
    # Keep every candidate rather than irreversibly pruning on invalid evidence.
    if set(scored_checkpoints.values()) == {0.0}:
        retained_steps = sorted(checkpoints)
    else:
        retained_steps = sorted({best_step, last_step})
    removed_steps = sorted(set(checkpoints) - set(retained_steps))
    return {
        "metric": metric,
        "best_step": best_step,
        "best_score": scored_checkpoints[best_step],
        "last_step": last_step,
        "retained_steps": retained_steps,
        "removed_steps": removed_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--metric", default="eval/aime")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete non-retained checkpoint directories; otherwise print the plan only",
    )
    args = parser.parse_args()

    plan = retention_plan(args.save_dir, args.log_file, args.metric)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.apply:
        return

    checkpoints = checkpoint_steps(args.save_dir)
    for step in plan["removed_steps"]:
        shutil.rmtree(checkpoints[step])

    metadata_path = args.save_dir / "checkpoint_retention.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)


if __name__ == "__main__":
    main()
