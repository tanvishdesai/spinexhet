"""Aggregate RSNA multifold outputs, with optional repaired XAI roots.

Later XAI roots override earlier roots for the same (fold, model). This lets a
V4 repair run recompute only EfficientNet or ViT fold 0/1 and then aggregate a
clean result table without copying the entire original result tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODELS = [
    "convnext_blackbox",
    "resnet50",
    "densenet121",
    "efficientnet_b4",
    "vit_small",
    "deit_small",
    "cbm_nonleaky",
    "cbm_leaky",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate RSNA V4 outputs.")
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--xai-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_mean_std(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(group_col)[metric_cols].agg(["mean", "std"]).round(4)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def aggregate_classification(eval_root: Path, output_dir: Path, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        for path in sorted((eval_root / model).glob("fold_*/metrics_val.json")):
            rows.append({"model": model, "fold": path.parent.name, **load_json(path)})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No classification metrics found under {eval_root}")
        return df
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(output_dir / "classification_metrics_by_fold.csv", index=False)
    metric_cols = [c for c in ["weighted_log_loss", "balanced_accuracy", "macro_f1", "accuracy", "auc_ovr"] if c in df]
    flatten_mean_std(df, "model", metric_cols).to_csv(output_dir / "classification_metrics_mean_std_flat.csv", index=False)
    return df


def iter_xai_summaries(xai_roots: list[Path], models: list[str]) -> dict[tuple[str, str], Path]:
    selected: dict[tuple[str, str], Path] = {}
    for root in xai_roots:
        for path in sorted(root.glob("fold_*/*/xai_summary_v2.json")):
            model = path.parent.name
            fold = path.parents[1].name
            if model in models:
                selected[(fold, model)] = path
    return selected


def aggregate_xai(xai_roots: list[Path], output_dir: Path, models: list[str]) -> pd.DataFrame:
    selected = iter_xai_summaries(xai_roots, models)
    rows: list[dict[str, Any]] = []
    for (fold, model), path in sorted(selected.items()):
        payload = load_json(path)
        summary = payload.get("summary", {})
        skipped = payload.get("skipped", summary.get("skipped", {}))
        faithfulness = summary.get("faithfulness", {})
        methods = sorted(faithfulness) if isinstance(faithfulness, dict) else []
        rows.append(
            {
                "fold": fold,
                "model": model,
                "source": str(path),
                "n_methods": len(methods),
                "methods": " ".join(methods),
                "skipped": json.dumps(skipped, sort_keys=True),
                "mean_spearman": summary.get("mean_spearman"),
                "mean_top20_iou": summary.get("mean_top20_iou"),
                "consensus_insertion_auc_mean": summary.get("consensus_insertion_auc_mean"),
                "consensus_expert_roi_mean": summary.get("consensus_expert_roi_mean"),
                "fw_consensus_insertion_auc_mean": summary.get("fw_consensus_insertion_auc_mean"),
                "uniform_consensus_insertion_auc_mean": summary.get("uniform_consensus_insertion_auc_mean"),
                "topk_consensus_insertion_auc_mean": summary.get("topk_consensus_insertion_auc_mean"),
                "fw_consensus_expert_roi_mean": summary.get("fw_consensus_expert_roi_mean"),
                "uniform_consensus_expert_roi_mean": summary.get("uniform_consensus_expert_roi_mean"),
                "topk_consensus_expert_roi_mean": summary.get("topk_consensus_expert_roi_mean"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No XAI summaries found.")
        return df
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(output_dir / "xai_multifold_summary_by_fold.csv", index=False)
    metric_cols = [
        c
        for c in [
            "mean_spearman",
            "mean_top20_iou",
            "consensus_insertion_auc_mean",
            "consensus_expert_roi_mean",
            "fw_consensus_insertion_auc_mean",
            "uniform_consensus_insertion_auc_mean",
            "topk_consensus_insertion_auc_mean",
            "fw_consensus_expert_roi_mean",
            "uniform_consensus_expert_roi_mean",
            "topk_consensus_expert_roi_mean",
        ]
        if c in df and df[c].notna().any()
    ]
    flatten_mean_std(df, "model", metric_cols).to_csv(output_dir / "xai_multifold_summary_mean_std_flat.csv", index=False)
    return df


def aggregate_faithfulness(xai_roots: list[Path], output_dir: Path, models: list[str]) -> pd.DataFrame:
    summary_paths = iter_xai_summaries(xai_roots, models)
    rows: list[pd.DataFrame] = []
    for (fold, model), summary_path in sorted(summary_paths.items()):
        faith_path = summary_path.parent / "faithfulness_metrics.csv"
        if not faith_path.exists():
            continue
        df = pd.read_csv(faith_path)
        if df.empty or "method" not in df:
            continue
        grouped = (
            df.groupby("method")[["deletion_auc", "insertion_auc"]]
            .mean()
            .reset_index()
        )
        grouped.insert(0, "model", model)
        grouped.insert(1, "fold", fold)
        grouped["source"] = str(faith_path)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output_dir / "faithfulness_by_fold_method.csv", index=False)
    mean = (
        out.groupby(["model", "method"])[["deletion_auc", "insertion_auc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    mean.columns = [f"{metric}_{stat}" for metric, stat in mean.columns]
    mean.reset_index().to_csv(output_dir / "faithfulness_mean_std_by_method_flat.csv", index=False)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classification = aggregate_classification(args.eval_root, args.output_dir, args.models)
    xai = aggregate_xai(args.xai_roots, args.output_dir, args.models)
    faithfulness = aggregate_faithfulness(args.xai_roots, args.output_dir, args.models)
    print(
        {
            "output_dir": str(args.output_dir),
            "classification_rows": int(len(classification)),
            "xai_rows": int(len(xai)),
            "faithfulness_rows": int(len(faithfulness)),
        }
    )


if __name__ == "__main__":
    main()
