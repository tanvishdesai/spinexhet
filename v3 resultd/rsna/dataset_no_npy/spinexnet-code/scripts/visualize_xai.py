"""Generate publication-quality figures from XAI benchmark v2 results.

Supports both per-model figures and cross-model comparison figures.

Usage:
    # Per-model figures
    python scripts/visualize_xai.py --results-dir xai/convnext_blackbox

    # Cross-model comparison (the key paper figures)
    python scripts/visualize_xai.py \
        --cross-model-dir xai/ \
        --output-dir figures/cross_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from spine_xnet.evaluation.stats import bootstrap_ci


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize XAI benchmark results.")
    p.add_argument("--results-dir", type=Path, default=None,
                    help="Directory with single-model benchmark CSVs")
    p.add_argument("--cross-model-dir", type=Path, default=None,
                    help="Parent directory containing per-model subdirs for cross-model figures")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Where to save figures")
    return p.parse_args()


# ── Per-model figures ────────────────────────────────────────────────────────


def plot_agreement_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Generate the agreement heatmap (hero figure)."""
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
    """Bar chart with bootstrap error bars comparing faithfulness across methods."""
    path = results_dir / "faithfulness_metrics.csv"
    if not path.exists():
        print(f"Skipping faithfulness bars — {path} not found")
        return

    df = pd.read_csv(path)
    methods = sorted(df["method"].unique())

    del_means, del_cis = [], []
    ins_means, ins_cis = [], []
    for method in methods:
        mdf = df[df["method"] == method]
        del_vals = mdf["deletion_auc"].dropna().values
        ins_vals = mdf["insertion_auc"].dropna().values
        del_means.append(del_vals.mean())
        ins_means.append(ins_vals.mean())
        if len(del_vals) >= 2:
            lo, hi = bootstrap_ci(del_vals, n_resamples=1000)
            del_cis.append(del_vals.mean() - lo)
        else:
            del_cis.append(0)
        if len(ins_vals) >= 2:
            lo, hi = bootstrap_ci(ins_vals, n_resamples=1000)
            ins_cis.append(ins_vals.mean() - lo)
        else:
            ins_cis.append(0)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, del_means, width, yerr=del_cis,
           label="Deletion AUC (↓ better)", color="#e74c3c", alpha=0.8, capsize=3)
    ax.bar(x + width / 2, ins_means, width, yerr=ins_cis,
           label="Insertion AUC (↑ better)", color="#2ecc71", alpha=0.8, capsize=3)
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
    """Clinical alignment by method and condition — now includes expert ROI."""
    path = results_dir / "clinical_alignment.csv"
    if not path.exists():
        print(f"Skipping clinical alignment — {path} not found")
        return

    df = pd.read_csv(path)

    # Expert ROI figure
    if "expert_roi_alignment" in df.columns:
        summary = df.groupby("method")["expert_roi_alignment"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(summary.index, summary["mean"], xerr=summary["std"],
                color="#9b59b6", alpha=0.8, capsize=3)
        ax.set_xlabel("Expert ROI IoU (top-25% saliency vs expert annotation)")
        ax.set_title("Expert ROI Clinical Alignment", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        fig.savefig(output_dir / "expert_roi_alignment.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved expert ROI alignment to {output_dir / 'expert_roi_alignment.png'}")

    # Condition breakdown
    pivot_col = "expert_roi_alignment" if "expert_roi_alignment" in df.columns else "proxy_roi_alignment"
    pivot = df.groupby(["method", "condition"])[pivot_col].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Clinical Alignment by Method and Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("ROI IoU")
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
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#3498db", alpha=0.8, capsize=3)
    ax.set_xlabel("Augmentation Consistency (Top-20% IoU)")
    ax.set_title("Explanation Stability Under Input Augmentation", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(output_dir / "consistency_by_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved consistency plot to {output_dir / 'consistency_by_method.png'}")


# ── Cross-model figures ──────────────────────────────────────────────────────


def plot_cross_model_agreement(cross_dir: Path, output_dir: Path) -> None:
    """Bar chart of mean Spearman agreement across architectures with CIs."""
    model_data: list[dict] = []
    for model_dir in sorted(cross_dir.iterdir()):
        agree_path = model_dir / "agreement_metrics.csv"
        if not agree_path.exists():
            continue
        df = pd.read_csv(agree_path)
        if "mean_spearman" not in df.columns:
            continue
        vals = df["mean_spearman"].dropna().values
        if len(vals) < 2:
            continue
        lo, hi = bootstrap_ci(vals, n_resamples=1000)
        model_data.append({
            "model": model_dir.name,
            "mean_spearman": float(vals.mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        })

    if not model_data:
        print("No cross-model agreement data found.")
        return

    mdf = pd.DataFrame(model_data).sort_values("mean_spearman", ascending=True)
    err_lo = mdf["mean_spearman"] - mdf["ci_lo"]
    err_hi = mdf["ci_hi"] - mdf["mean_spearman"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mdf)))
    ax.barh(mdf["model"], mdf["mean_spearman"],
            xerr=[err_lo.values, err_hi.values],
            color=colors, alpha=0.85, capsize=4, edgecolor="gray")
    ax.set_xlabel("Mean Spearman ρ (inter-method agreement)", fontsize=11)
    ax.set_title("Architecture-Dependent XAI Disagreement (95% CI)",
                 fontsize=13, fontweight="bold")
    ax.axvline(0.0, color="red", linestyle="--", alpha=0.4, label="No agreement")
    ax.set_xlim(-0.2, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "cross_model_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save the data as CSV for the paper tables
    mdf.to_csv(output_dir / "cross_model_agreement.csv", index=False)
    print(f"Saved cross-model agreement to {output_dir / 'cross_model_agreement.png'}")


def plot_cross_model_faithfulness(cross_dir: Path, output_dir: Path) -> None:
    """Grouped bar chart: insertion/deletion AUC across models and methods."""
    all_dfs = []
    for model_dir in sorted(cross_dir.iterdir()):
        faith_path = model_dir / "faithfulness_metrics.csv"
        if not faith_path.exists():
            continue
        df = pd.read_csv(faith_path)
        df["model"] = model_dir.name
        all_dfs.append(df)

    if not all_dfs:
        print("No cross-model faithfulness data found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    pivot_ins = combined.groupby(["model", "method"])["insertion_auc"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_ins.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Insertion AUC Across Architectures and XAI Methods",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Insertion AUC (↑ better)")
    ax.set_xlabel("")
    ax.legend(title="XAI Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "cross_model_faithfulness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-model faithfulness to {output_dir / 'cross_model_faithfulness.png'}")


def plot_cross_model_expert_roi(cross_dir: Path, output_dir: Path) -> None:
    """Expert ROI alignment comparison across architectures."""
    all_dfs = []
    for model_dir in sorted(cross_dir.iterdir()):
        clin_path = model_dir / "clinical_alignment.csv"
        if not clin_path.exists():
            continue
        df = pd.read_csv(clin_path)
        if "expert_roi_alignment" not in df.columns:
            continue
        df["model"] = model_dir.name
        all_dfs.append(df)

    if not all_dfs:
        print("No cross-model expert ROI data found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    summary = combined.groupby(["model", "method"])["expert_roi_alignment"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    summary.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Expert ROI Alignment Across Architectures",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Expert ROI IoU")
    ax.set_xlabel("")
    ax.legend(title="XAI Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "cross_model_expert_roi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cross-model expert ROI to {output_dir / 'cross_model_expert_roi.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Per-model figures
    if args.results_dir is not None:
        output_dir = args.output_dir or args.results_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_agreement_heatmap(args.results_dir, output_dir)
        plot_faithfulness_bars(args.results_dir, output_dir)
        plot_clinical_alignment(args.results_dir, output_dir)
        plot_consistency(args.results_dir, output_dir)
        print(f"\nPer-model figures saved to {output_dir}")

    # Cross-model comparison figures
    if args.cross_model_dir is not None:
        output_dir = args.output_dir or args.cross_model_dir / "cross_model_figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_model_agreement(args.cross_model_dir, output_dir)
        plot_cross_model_faithfulness(args.cross_model_dir, output_dir)
        plot_cross_model_expert_roi(args.cross_model_dir, output_dir)
        print(f"\nCross-model figures saved to {output_dir}")


if __name__ == "__main__":
    main()
