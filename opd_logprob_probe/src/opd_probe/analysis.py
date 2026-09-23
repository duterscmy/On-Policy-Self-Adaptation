from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_token_shards(input_dir):
    paths = sorted((Path(input_dir) / "tokens").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No token shards found under {Path(input_dir) / 'tokens'}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def add_bins(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["student_logprob"]).copy()
    mode = cfg.get("mode", "fixed")
    if mode == "quantile":
        df["logp_bin"] = pd.qcut(df["student_logprob"], q=int(cfg.get("num_quantile_bins", 20)), duplicates="drop")
    elif mode == "fixed":
        lo, hi = float(cfg.get("logp_min", -12)), float(cfg.get("logp_max", 0))
        width = float(cfg.get("bin_width", 0.5))
        clipped = df["student_logprob"].clip(lower=lo, upper=hi - 1e-9)
        edges = np.arange(lo, hi + width * 0.5, width)
        if edges[-1] < hi:
            edges = np.append(edges, hi)
        df["logp_bin"] = pd.cut(clipped, bins=edges, include_lowest=True)
    else:
        raise ValueError(f"Unknown bin mode: {mode}")
    return df.dropna(subset=["logp_bin"])


def aggregate_bins(df: pd.DataFrame) -> pd.DataFrame:
    q25 = lambda x: x.quantile(0.25)
    q75 = lambda x: x.quantile(0.75)
    g = df.groupby("logp_bin", observed=True)
    out = g.agg(
        count=("student_logprob", "size"),
        student_logprob_mean=("student_logprob", "mean"),
        student_logprob_median=("student_logprob", "median"),
        prob_diff_signed_mean=("prob_diff_signed", "mean"),
        prob_diff_signed_median=("prob_diff_signed", "median"),
        prob_diff_abs_mean=("prob_diff_abs", "mean"),
        prob_diff_abs_median=("prob_diff_abs", "median"),
        prob_diff_abs_q25=("prob_diff_abs", q25),
        prob_diff_abs_q75=("prob_diff_abs", q75),
        logratio_signed_mean=("logratio_signed", "mean"),
        logratio_signed_median=("logratio_signed", "median"),
        logratio_abs_mean=("logratio_abs", "mean"),
        logratio_abs_median=("logratio_abs", "median"),
        logratio_abs_q25=("logratio_abs", q25),
        logratio_abs_q75=("logratio_abs", q75),
        grad_proxy_abs_mean=("grad_proxy_abs", "mean"),
        grad_proxy_abs_median=("grad_proxy_abs", "median"),
        grad_proxy_abs_q25=("grad_proxy_abs", q25),
        grad_proxy_abs_q75=("grad_proxy_abs", q75),
        grad_proxy_abs_sum=("grad_proxy_abs", "sum"),
        student_entropy_mean=("student_entropy", "mean"),
        student_entropy_median=("student_entropy", "median"),
        teacher_entropy_mean=("teacher_entropy", "mean"),
        teacher_entropy_median=("teacher_entropy", "median"),
    ).reset_index()
    out["token_share"] = out["count"] / out["count"].sum()
    total_g = out["grad_proxy_abs_sum"].sum()
    out["grad_mass_share"] = out["grad_proxy_abs_sum"] / total_g if total_g > 0 else 0.0
    out["bin_label"] = out["logp_bin"].astype(str)
    return out
