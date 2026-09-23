from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class Problem:
    local_index: int
    source_index: int
    problem_id: str
    problem: str
    answer: Any = None
    solution: Any = None
    raw: dict | None = None


def _safe_get(row: dict, key: str | None, default=None):
    if not key:
        return default
    return row.get(key, default)


def load_problem_subset(cfg: dict, seed: int) -> list[Problem]:
    ds = load_dataset(cfg["name"], split=cfg["split"])
    n_total = len(ds)
    n = cfg.get("num_questions")
    if n is None or int(n) <= 0 or int(n) > n_total:
        n = n_total
    else:
        n = int(n)

    indices = list(range(n_total))
    if cfg.get("random_subset", True):
        rng = random.Random(seed)
        rng.shuffle(indices)
    indices = indices[:n]

    problems = []
    for local_idx, src_idx in enumerate(indices):
        row = dict(ds[src_idx])
        raw_id = _safe_get(row, cfg.get("id_field"), src_idx)
        problems.append(
            Problem(
                local_index=local_idx,
                source_index=src_idx,
                problem_id=str(raw_id),
                problem=str(row[cfg["problem_field"]]),
                answer=_safe_get(row, cfg.get("answer_field")),
                solution=_safe_get(row, cfg.get("solution_field")),
                raw=row,
            )
        )
    return problems
