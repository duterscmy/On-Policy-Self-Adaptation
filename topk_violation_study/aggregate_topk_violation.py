#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--primary-k', type=int, default=10)
    return p.parse_args()


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)

    summaries, prefixes = [], []
    for p in root.rglob('summary.csv'):
        if out in p.parents:
            continue
        summaries.append(pd.read_csv(p))
        pp = p.with_name('prefix_summary.csv')
        if pp.exists():
            prefixes.append(pd.read_csv(pp))

    if not summaries:
        raise SystemExit(f'No summary.csv found under {root}')

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / 'all_summary.csv', index=False)
    prefix = pd.concat(prefixes, ignore_index=True) if prefixes else pd.DataFrame()
    if len(prefix):
        prefix.to_csv(out / 'all_prefix_summary.csv', index=False)

    for dataset, g0 in summary.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(7.3, 4.7))
        for model, g in g0.groupby('model'):
            g = g.sort_values('K')
            ax.plot(g['K'], 100 * g['within_problem_acc_delta'], marker='o', label=model)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xscale('log')
        ax.set_xlabel('K')
        ax.set_ylabel('Within-question Δ accuracy (pp)\nviolation − no violation')
        ax.set_title(f'Top-K violation predictiveness · {dataset}')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{safe_name(dataset)}__cross_model_effect.png', dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.3, 4.7))
        for model, g in g0.groupby('model'):
            g = g.sort_values('K')
            ax.plot(g['K'], g['fraction_with_any_violation'], marker='o', label=model)
        ax.set_xscale('log')
        ax.set_xlabel('K')
        ax.set_ylabel('Fraction of rollouts with ≥1 violation')
        ax.set_title(f'Tail-event prevalence · {dataset}')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{safe_name(dataset)}__cross_model_prevalence.png', dpi=220)
        plt.close(fig)

    if len(prefix):
        pg0 = prefix[prefix['K'] == args.primary_k]
        for dataset, g0 in pg0.groupby('dataset'):
            fig, ax = plt.subplots(figsize=(7.3, 4.7))
            for model, g in g0.groupby('model'):
                g = g.sort_values('prefix_tokens')
                ax.plot(g['prefix_tokens'], 100 * g['within_problem_acc_delta'], marker='o', label=model)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xlabel('Observed prefix length (tokens)')
            ax.set_ylabel('Within-question Δ accuracy (pp)\nprefix violation − no prefix violation')
            ax.set_title(f'Early Top-{args.primary_k} violation predictiveness · {dataset}')
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f'{safe_name(dataset)}__cross_model_prefix_K{args.primary_k}.png', dpi=220)
            plt.close(fig)

    print(f'Wrote aggregate CSVs and figures to {out}')


if __name__ == '__main__':
    main()
