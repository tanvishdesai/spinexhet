"""Generate publication-quality figures from XAI benchmark v2 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize XAI benchmark results.")
    p.add_argument("--results-dir", type=Path, required=True, help="Directory with benchmark CSVs")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save figures")
    return p.parse_args()


def plot_agreement_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Generate the 7x7 agreement heatmap (hero figure of the paper)."""
    path = results_dir / "agreement_metrics.csv"
    if not path.exists():
        print(f"Skipping agreement heatmap — {path} not found")
        return

    df = pd.read_csv(path)
    spearman_cols = [c for c in df.columns if c.startswith("spearman_") and "_vs_" in c]
    if not spearman_cols:
        print("No pairwise spearman columns found.")
        return

    methods = set()
    for col in spearman_cols:
        parts = col.replace("spearman_", "").split("_vs_")
        methods.update(parts)
    methods = sorted(methods)
    n = len(methods)

    matrix = np.eye(n)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                continue
            col_a = f"spearman_{m1}_vs_{m2}"
            col_b = f"spearman_{m2}_vs_{m1}"
            col = col_a if col_a in df.columns else col_b if col_b in df.columns else None
            if col and col in df.columns:
                val = df[col].mean()
                matrix[i, j] = val
                matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matrix, xticklabels=methods, yticklabels=methods,
        annot=True, fmt=".2f", cmap="RdYlGn", vmin=-0.5, vmax=1.0,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Spearman Rank Correlation"},
    )
    ax.set_title("Pairwise Agreement Between XAI Methods", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_dir / "agreement_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved agreement heatmap to {output_dir / 'agreement_heatmap.png'}")


def plot_faithfulness_bars(results_dir: Path, output_dir: Path) -> None:
    """Bar chart comparing faithfulness metrics across methods."""
    path = results_dir / "faithfulness_metrics.csv"
    if not path.exists():
        print(f"Skipping faithfulness bars — {path} not found")
        return

    df = pd.read_csv(path)
    summary = df.groupby("method")[["deletion_auc", "insertion_auc"]].agg(["mean", "std"])

    methods = summary.index.tolist()
    del_means = summary[("deletion_auc", "mean")].values
    del_stds = summary[("deletion_auc", "std")].values
    ins_means = summary[("insertion_auc", "mean")].values
    ins_stds = summary[("insertion_auc", "std")].values

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, del_means, width, yerr=del_stds, label="Deletion AUC (↓ better)", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, ins_means, width, yerr=ins_stds, label="Insertion AUC (↑ better)", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("AUC Score")
    ax.set_title("Faithfulness Benchmark: Deletion vs Insertion AUC", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    plt.tight_layout()
    fig.savefig(output_dir / "faithfulness_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved faithfulness comparison to {output_dir / 'faithfulness_comparison.png'}")


def plot_clinical_alignment(results_dir: Path, output_dir: Path) -> None:
    """Clinical alignment by method and condition."""
    path = results_dir / "clinical_alignment.csv"
    if not path.exists():
        print(f"Skipping clinical alignment — {path} not found")
        return

    df = pd.read_csv(path)
    pivot = df.groupby(["method", "condition"])["proxy_roi_alignment"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Clinical Alignment by Method and Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Proxy ROI IoU")
    ax.set_xlabel("")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "clinical_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clinical alignment to {output_dir / 'clinical_alignment.png'}")


def plot_consistency(results_dir: Path, output_dir: Path) -> None:
    """Consistency (augmentation stability) by method."""
    path = results_dir / "consistency_metrics.csv"
    if not path.exists():
        print(f"Skipping consistency — {path} not found")
        return

    df = pd.read_csv(path)
    summary = df.groupby("method")["augmentation_iou"].agg(["mean", "std"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#3498db", alpha=0.8)
    ax.set_xlabel("Augmentation Consistency (Top-20% IoU)")
    ax.set_title("Explanation Stability Under Input Augmentation", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(output_dir / "consistency_by_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved consistency plot to {output_dir / 'consistency_by_method.png'}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_agreement_heatmap(args.results_dir, output_dir)
    plot_faithfulness_bars(args.results_dir, output_dir)
    plot_clinical_alignment(args.results_dir, output_dir)
    plot_consistency(args.results_dir, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
