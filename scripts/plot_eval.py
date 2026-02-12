#!/usr/bin/env python3
"""
plot_eval.py — Generate evaluation visuals from training results.

Reads metrics produced by the training notebook and saves two plots:
  • docs/visuals/recall_at_k.png   — Recall@K curve
  • docs/visuals/score_hist.png    — Score distribution (positive vs negative)

Usage:
    python scripts/plot_eval.py                          # uses defaults
    python scripts/plot_eval.py --metrics results.json   # custom file

If no metrics file is found, the script creates a small synthetic example
so that the plots are never empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "visuals")
DEFAULT_METRICS = os.path.join(REPO_ROOT, "metrics", "eval_results.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Synthetic fallback ────────────────────────────────────────────────

def _synthetic_metrics() -> dict:
    """Return plausible synthetic metrics so the plots are always valid."""
    rng = np.random.default_rng(42)

    # Recall@K for K = 1..20
    K_values = list(range(1, 21))
    recall_at_k = [min(1.0, 0.45 + 0.03 * k + rng.normal(0, 0.01)) for k in K_values]
    recall_at_k = [round(min(max(v, 0), 1.0), 4) for v in recall_at_k]

    # Score distributions
    pos_scores = rng.beta(5, 2, size=200).tolist()
    neg_scores = rng.beta(2, 5, size=800).tolist()

    return {
        "K_values": K_values,
        "recall_at_k": recall_at_k,
        "positive_scores": pos_scores,
        "negative_scores": neg_scores,
        "_source": "synthetic (placeholder)",
    }


def load_metrics(path: str | None) -> dict:
    """Load metrics JSON or fall back to synthetic data."""
    if path and os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        print(f"✅ Loaded metrics from {path}")
        return data

    print(
        f"⚠️  Metrics file not found ({path or DEFAULT_METRICS}).\n"
        "   Using synthetic placeholder data. To use real results, run the\n"
        "   training notebook and save metrics to metrics/eval_results.json."
    )
    return _synthetic_metrics()


# ── Plots ─────────────────────────────────────────────────────────────

def plot_recall_at_k(metrics: dict, out_path: str) -> None:
    K = metrics["K_values"]
    R = metrics["recall_at_k"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K, R, "o-", color="#1565C0", linewidth=2, markersize=5)
    ax.set_xlabel("K (candidates considered)", fontsize=11)
    ax.set_ylabel("Recall@K", fontsize=11)
    ax.set_title("Recall@K — Trade Matching", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(K)
    ax.grid(axis="y", alpha=0.3)

    # Annotate key points
    for k_val in [1, 5, 10, 20]:
        if k_val in K:
            idx = K.index(k_val)
            ax.annotate(
                f"R@{k_val}={R[idx]:.2f}",
                xy=(k_val, R[idx]),
                xytext=(k_val + 0.5, R[idx] - 0.06),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    if metrics.get("_source", "").startswith("synthetic"):
        ax.text(
            0.98, 0.02, "placeholder data",
            transform=ax.transAxes, fontsize=7, color="gray",
            ha="right", va="bottom", style="italic",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


def plot_score_histogram(metrics: dict, out_path: str) -> None:
    pos = np.asarray(metrics["positive_scores"])
    neg = np.asarray(metrics["negative_scores"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(neg, bins=bins, alpha=0.55, label="Negatives", color="#E53935", density=True)
    ax.hist(pos, bins=bins, alpha=0.55, label="Positives", color="#43A047", density=True)
    ax.set_xlabel("Model Score (σ(logit))", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score Distribution — Positive vs Negative Pairs", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    if metrics.get("_source", "").startswith("synthetic"):
        ax.text(
            0.98, 0.98, "placeholder data",
            transform=ax.transAxes, fontsize=7, color="gray",
            ha="right", va="top", style="italic",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--metrics", type=str, default=None, help="Path to eval_results.json")
    args = parser.parse_args()

    _ensure_dir(OUTPUT_DIR)

    metrics = load_metrics(args.metrics or DEFAULT_METRICS)

    plot_recall_at_k(metrics, os.path.join(OUTPUT_DIR, "recall_at_k.png"))
    plot_score_histogram(metrics, os.path.join(OUTPUT_DIR, "score_hist.png"))

    print("\n✅ All evaluation plots saved to docs/visuals/")


if __name__ == "__main__":
    main()
