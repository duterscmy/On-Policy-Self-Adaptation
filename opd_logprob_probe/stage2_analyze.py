#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

from opd_probe.analysis import load_token_shards, add_bins, aggregate_bins


def savefig(fig, path, dpi):
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def ticks(ax, labels, max_ticks=12):
    n = len(labels); idx = np.arange(n) if n <= max_ticks else np.unique(np.linspace(0, n-1, max_ticks).round().astype(int))
    ax.set_xticks(idx); ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")


def plot_distribution(s, path, dpi):
    x = np.arange(len(s)); fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x, 100*s["token_share"]); ax.set_xlabel("Student sampled-token log-probability bin")
    ax.set_ylabel("Share of sampled tokens (%)"); ax.set_title("Student sampled-token log-probability distribution")
    ticks(ax, s["bin_label"].tolist()); savefig(fig, path, dpi)


def plot_overlay(s, metric, ylabel, title, path, dpi, iqr=None):
    x = np.arange(len(s)); fig, ax = plt.subplots(figsize=(10,5)); ax2 = ax.twinx()
    ax.bar(x, s[metric].to_numpy(), alpha=.75)
    ax2.plot(x, 100*s["token_share"].to_numpy(), marker="o", linewidth=1.5)
    if iqr:
        ax.fill_between(x, s[iqr[0]].to_numpy(), s[iqr[1]].to_numpy(), alpha=.15)
    ax.set_xlabel("Student sampled-token log-probability bin"); ax.set_ylabel(ylabel)
    ax2.set_ylabel("Share of sampled tokens (%)"); ax.set_title(title)
    ticks(ax, s["bin_label"].tolist()); savefig(fig, path, dpi)


def plot_gradient(s, central, path, dpi):
    x = np.arange(len(s)); fig, ax = plt.subplots(figsize=(10,5)); ax2 = ax.twinx()
    ax.bar(x, s[f"grad_proxy_abs_{central}"].to_numpy(), alpha=.75)
    ax2.plot(x, 100*s["grad_mass_share"].to_numpy(), marker="o", linewidth=1.5)
    ax.set_xlabel("Student sampled-token log-probability bin")
    ax.set_ylabel(f"{central.title()} per-token |A|(1-p_s)")
    ax2.set_ylabel("Share of total gradient-proxy mass (%)")
    ax.set_title("Per-token gradient intensity vs total training-signal mass")
    ticks(ax, s["bin_label"].tolist()); savefig(fig, path, dpi)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    df = load_token_shards(cfg["input_dir"])
    if cfg.get("output", {}).get("write_consolidated_parquet", True):
        df.to_parquet(out / "token_level_all.parquet", index=False)
    s = aggregate_bins(add_bins(df, cfg["binning"])); s.to_csv(out / "binned_statistics.csv", index=False)
    central = cfg["statistics"].get("central", "median")
    show_iqr = bool(cfg["statistics"].get("show_iqr", True))
    fmt, dpi = cfg["plot"].get("format", "pdf"), int(cfg["plot"].get("dpi", 200))
    plot_distribution(s, out/f"01_student_logprob_distribution.{fmt}", dpi)
    plot_overlay(s, f"prob_diff_abs_{central}", f"{central.title()} |p_teacher - p_student|",
                 "Absolute teacher-student probability disagreement", out/f"02_absolute_probability_gap.{fmt}", dpi,
                 ("prob_diff_abs_q25","prob_diff_abs_q75") if show_iqr else None)
    plot_overlay(s, f"logratio_abs_{central}", f"{central.title()} |log p_teacher - log p_student|",
                 "Reverse-KL log-ratio disagreement", out/f"03_logratio_gap.{fmt}", dpi,
                 ("logratio_abs_q25","logratio_abs_q75") if show_iqr else None)
    plot_gradient(s, central, out/f"04_gradient_signal.{fmt}", dpi)
    print("Token rows:", len(df), "Problems:", df.problem_local_index.nunique())
    print("Saved to", out)


if __name__ == "__main__":
    main()
