"""Aggregate CUB-200 fold outputs from per-model Kaggle notebooks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


MODELS = ["convnext_blackbox", "resnet50", "densenet121", "efficientnet_b4", "vit_small"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate CUB-200 generalization outputs.")
    p.add_argument("--input-roots", type=Path, nargs="*", default=None)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/cub_aggregate"))
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--copy-inputs", action="store_true", help="Merge cub_eval/cub_xai trees before aggregating.")
    return p.parse_args()


def default_input_roots() -> list[Path]:
    roots = [Path("/kaggle/working")]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        roots += [p for p in input_root.iterdir() if p.is_dir()]
        datasets = input_root / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots


def merge_tree(name: str, roots: list[Path], work: Path) -> Path:
    dest = work / name
    for root in roots:
        src = root / name
        if src.exists():
            try:
                if src.resolve() == dest.resolve():
                    continue
            except Exception:
                pass
            shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_classification(eval_root: Path, output_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for path in sorted((eval_root / model).glob("fold_*/metrics_val.json")):
            metrics = load_json(path)
            rows.append({"model": model, "fold": path.parent.name, **metrics})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No CUB classification metrics found under {eval_root}")
        return df
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(output_dir / "cub_classification_metrics_by_fold.csv", index=False)
    metric_cols = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1", "top5_accuracy"]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(output_dir / "cub_classification_metrics_mean_std.csv")
    print(summary)
    return df


def aggregate_xai(xai_root: Path, output_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(xai_root.glob("fold_*/*/xai_summary_cub.json")):
        model = path.parent.name
        if model not in models:
            continue
        payload = load_json(path)
        summary = payload.get("summary", {})
        rows.append(
            {
                "model": model,
                "fold": path.parents[1].name,
                "mean_spearman": summary.get("mean_spearman"),
                "mean_top20_iou": summary.get("mean_top20_iou"),
                "skipped": json.dumps(summary.get("skipped", {}), sort_keys=True),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No CUB XAI summaries found under {xai_root}")
        return df
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(output_dir / "cub_xai_summary_by_fold.csv", index=False)
    summary = df.groupby("model")[["mean_spearman", "mean_top20_iou"]].agg(["mean", "std"]).round(4)
    summary.to_csv(output_dir / "cub_xai_summary_mean_std.csv")
    print(summary)
    return df


def make_plots(classification: pd.DataFrame, xai: pd.DataFrame, output_dir: Path) -> None:
    if classification.empty and xai.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not classification.empty:
        cls = classification.groupby("model")["accuracy"].agg(["mean", "std"]).reindex(MODELS).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cls.index, cls["mean"], yerr=cls["std"], color="#4e79a7", capsize=4)
        ax.set_ylabel("CUB-200 validation accuracy")
        ax.set_ylim(0, max(1.0, float((cls["mean"] + cls["std"]).max()) * 1.1))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(fig_dir / "cub_classification_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if not xai.empty:
        xai_summary = xai.groupby("model")["mean_spearman"].agg(["mean", "std"]).reindex(MODELS).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(xai_summary.index, xai_summary["mean"], yerr=xai_summary["std"], color="#59a14f", capsize=4)
        ax.set_ylabel("Mean attribution Spearman rho")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(fig_dir / "cub_xai_agreement.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = args.input_roots or default_input_roots()
    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else args.output_dir

    if args.copy_inputs:
        eval_root = merge_tree("cub_eval", roots, work)
        xai_root = merge_tree("cub_xai", roots, work)
    else:
        eval_root = work / "cub_eval"
        xai_root = work / "cub_xai"

    classification = aggregate_classification(eval_root, args.output_dir, args.models)
    xai = aggregate_xai(xai_root, args.output_dir, args.models)
    make_plots(classification, xai, args.output_dir)
    print({"output_dir": str(args.output_dir), "eval_root": str(eval_root), "xai_root": str(xai_root)})


if __name__ == "__main__":
    main()
