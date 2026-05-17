"""Audit V3 reviewer concerns and produce paper-ready statistical tables.

This script is intentionally CPU-only. It reads the existing V3 result folders
and writes compact CSVs for the issues repeatedly raised in mentor review:

  - paired significance tests for architecture-level agreement gaps,
  - CUB classifier exclusion report,
  - consensus-vs-best-individual tests,
  - RSNA/CUB rank-stability table,
  - EfficientNet IG vs GradientSHAP sanity audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, ttest_rel, wilcoxon


DEFAULT_RSNA_ROOT = Path("v3 resultd/rsna/dataset_no_npy")
DEFAULT_CUB_ROOT = Path("v3 resultd/cuba/dataset_no_npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze V3 review gap statistics.")
    p.add_argument("--rsna-root", type=Path, default=DEFAULT_RSNA_ROOT)
    p.add_argument("--cub-root", type=Path, default=DEFAULT_CUB_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("review_gap_analysis"))
    p.add_argument("--cub-min-accuracy", type=float, default=0.70)
    return p.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def paired_test(left: pd.Series, right: pd.Series) -> dict[str, float | int | str]:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    mask = left.notna() & right.notna()
    left = left[mask]
    right = right[mask]
    diff = left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
    out: dict[str, float | int | str] = {
        "n": int(len(diff)),
        "mean_delta": float(np.mean(diff)) if len(diff) else float("nan"),
        "std_delta": float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan"),
    }
    if len(diff) < 2:
        out["note"] = "too few paired observations"
        return out
    try:
        stat, p_value = wilcoxon(left, right, zero_method="wilcox")
        out["wilcoxon_stat"] = float(stat)
        out["wilcoxon_p"] = float(p_value)
    except ValueError as exc:
        out["wilcoxon_note"] = str(exc)
    t_stat, t_p = ttest_rel(left, right)
    out["paired_t_stat"] = float(t_stat)
    out["paired_t_p"] = float(t_p)
    return out


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df:
        return pd.DataFrame()
    out = df.pivot_table(index="fold", columns="model", values=metric, aggfunc="mean")
    return out.sort_index()


def cub_exclusion_report(cub_cls: pd.DataFrame, min_accuracy: float, output_dir: Path) -> pd.DataFrame:
    if cub_cls.empty:
        return pd.DataFrame()
    report = (
        cub_cls.groupby("model")["accuracy"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    report["xai_included"] = report["mean"] >= min_accuracy
    report["reason"] = np.where(
        report["xai_included"],
        "accuracy threshold met",
        "excluded from CUB XAI summary: classifier did not converge",
    )
    report.to_csv(output_dir / "cub_model_exclusion_report.csv", index=False)
    return report


def rank_stability(
    rsna_xai: pd.DataFrame,
    cub_xai: pd.DataFrame,
    cub_report: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not rsna_xai.empty:
        for fold, group in rsna_xai.groupby("fold"):
            ranked = group.sort_values("mean_spearman", ascending=False)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append({"dataset": "RSNA", "fold": fold, "rank": rank, "model": row["model"], "mean_spearman": row["mean_spearman"]})

    eligible = set(cub_report.loc[cub_report["xai_included"], "model"]) if not cub_report.empty else set(cub_xai["model"].unique())
    if not cub_xai.empty:
        for fold, group in cub_xai[cub_xai["model"].isin(eligible)].groupby("fold"):
            ranked = group.sort_values("mean_spearman", ascending=False)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append({"dataset": "CUB", "fold": fold, "rank": rank, "model": row["model"], "mean_spearman": row["mean_spearman"]})

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / "rank_stability_table.csv", index=False)
    return out


def architecture_tests(rsna_xai: pd.DataFrame, cub_xai: pd.DataFrame, cub_report: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, df in [("RSNA", rsna_xai), ("CUB", cub_xai)]:
        pivot = pivot_metric(df, "mean_spearman")
        for left, right in [("densenet121", "vit_small"), ("resnet50", "vit_small"), ("densenet121", "deit_small")]:
            if left in pivot and right in pivot:
                row = {"test": f"{dataset}: {left} vs {right}", "metric": "mean_spearman"}
                row.update(paired_test(pivot[left], pivot[right]))
                rows.append(row)

    if not rsna_xai.empty and not cub_xai.empty:
        cub_eligible = set(cub_report.loc[cub_report["xai_included"], "model"]) if not cub_report.empty else set(cub_xai["model"])
        rsna_mean = rsna_xai.groupby("model")["mean_spearman"].mean()
        cub_mean = cub_xai[cub_xai["model"].isin(cub_eligible)].groupby("model")["mean_spearman"].mean()
        shared = sorted(set(rsna_mean.index) & set(cub_mean.index))
        if len(shared) >= 3:
            rho, rho_p = spearmanr(rsna_mean.loc[shared], cub_mean.loc[shared])
            tau, tau_p = kendalltau(rsna_mean.loc[shared], cub_mean.loc[shared])
            rows.append(
                {
                    "test": "RSNA vs CUB architecture-rank correlation",
                    "metric": "mean_spearman",
                    "n": len(shared),
                    "spearman_rho": float(rho),
                    "spearman_p": float(rho_p),
                    "kendall_tau": float(tau),
                    "kendall_p": float(tau_p),
                    "models": " ".join(shared),
                }
            )
    return rows


def consensus_vs_best(rsna_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    xai_root = rsna_root / "combined_xai_multifold"
    for consensus_path in xai_root.glob("fold_*/*/consensus_metrics.csv"):
        faith_path = consensus_path.parent / "faithfulness_metrics.csv"
        if not faith_path.exists():
            continue
        consensus = pd.read_csv(consensus_path)
        faith = pd.read_csv(faith_path)
        fw_col = "fw_consensus_insertion_auc" if "fw_consensus_insertion_auc" in consensus else "consensus_insertion_auc"
        uniform_col = "uniform_consensus_insertion_auc" if "uniform_consensus_insertion_auc" in consensus else None
        oracle = faith.groupby("sample_id")["insertion_auc"].max().rename("oracle_best_insertion_auc")
        method_means = faith.groupby("method")["insertion_auc"].mean().sort_values(ascending=False)
        best_fixed_method = str(method_means.index[0]) if not method_means.empty else ""
        fixed = (
            faith[faith["method"] == best_fixed_method]
            .set_index("sample_id")["insertion_auc"]
            .rename("best_fixed_method_insertion_auc")
        )
        merged = consensus.merge(oracle, on="sample_id", how="inner").merge(fixed, on="sample_id", how="inner")
        for _, row in merged.iterrows():
            record = {
                "fold": consensus_path.parents[1].name,
                "model": consensus_path.parent.name,
                "sample_id": row["sample_id"],
                "best_fixed_method": best_fixed_method,
                "fw_consensus_insertion_auc": row[fw_col],
                "best_fixed_method_insertion_auc": row["best_fixed_method_insertion_auc"],
                "oracle_best_insertion_auc": row["oracle_best_insertion_auc"],
                "fw_minus_best_fixed_method": row[fw_col] - row["best_fixed_method_insertion_auc"],
                "fw_minus_oracle_best": row[fw_col] - row["oracle_best_insertion_auc"],
            }
            if uniform_col:
                record["uniform_consensus_insertion_auc"] = row[uniform_col]
                record["fw_minus_uniform"] = row[fw_col] - row[uniform_col]
            rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.to_csv(output_dir / "consensus_vs_best_individual.csv", index=False)

    tests: list[dict[str, Any]] = []
    tests.append({"test": "FW consensus vs best fixed individual method", **paired_test(out["fw_consensus_insertion_auc"], out["best_fixed_method_insertion_auc"])})
    tests.append({"test": "FW consensus vs per-sample oracle best method", **paired_test(out["fw_consensus_insertion_auc"], out["oracle_best_insertion_auc"])})
    if "uniform_consensus_insertion_auc" in out:
        tests.append({"test": "FW consensus vs uniform consensus", **paired_test(out["fw_consensus_insertion_auc"], out["uniform_consensus_insertion_auc"])})
    pd.DataFrame(tests).to_csv(output_dir / "consensus_significance_tests.csv", index=False)
    return out


def efficientnet_gradient_audit(rsna_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for faith_path in (rsna_root / "combined_xai_multifold").glob("fold_*/efficientnet_b4/faithfulness_metrics.csv"):
        faith = pd.read_csv(faith_path)
        pivot = faith.pivot_table(index="sample_id", columns="method", values="insertion_auc", aggfunc="mean")
        if {"integrated_gradients", "gradient_shap"}.issubset(pivot.columns):
            diff = pivot["integrated_gradients"] - pivot["gradient_shap"]
            rows.append(
                {
                    "fold": faith_path.parents[1].name,
                    "n": int(diff.notna().sum()),
                    "mean_abs_ig_gradshap_delta": float(diff.abs().mean()),
                    "pearson_corr": float(pivot[["integrated_gradients", "gradient_shap"]].corr().iloc[0, 1]),
                    "identical_fraction": float(np.isclose(diff.fillna(np.inf), 0.0, atol=1e-8).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(output_dir / "efficientnet_gradient_audit.csv", index=False)
    return out


def write_summary(output_dir: Path, stats_rows: list[dict[str, Any]], cub_report: pd.DataFrame) -> None:
    lines = [
        "# V3 Review Gap Analysis",
        "",
        "Generated by `scripts/analyze_v3_review_gaps.py`.",
        "",
        "Key outputs:",
        "",
        "- `statistical_tests.csv`: paired tests for architecture agreement gaps and rank correlation.",
        "- `consensus_significance_tests.csv`: FW consensus against best fixed method, oracle best method, and uniform consensus when available.",
        "- `cub_model_exclusion_report.csv`: CUB classifiers eligible for XAI analysis.",
        "- `rank_stability_table.csv`: per-fold model ranks by attribution agreement.",
        "- `efficientnet_gradient_audit.csv`: IG/GradientSHAP sanity check for EfficientNet.",
        "",
    ]
    if not cub_report.empty:
        excluded = cub_report.loc[~cub_report["xai_included"], "model"].tolist()
        lines.append(f"CUB models excluded at the current threshold: {', '.join(excluded) if excluded else 'none'}.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rsna_xai = read_csv_if_exists(args.rsna_root / "xai_multifold_summary_by_fold.csv")
    cub_xai = read_csv_if_exists(args.cub_root / "cub_aggregate" / "cub_xai_summary_by_fold.csv")
    cub_cls = read_csv_if_exists(args.cub_root / "cub_aggregate" / "cub_classification_metrics_by_fold.csv")

    cub_report = cub_exclusion_report(cub_cls, args.cub_min_accuracy, args.output_dir)
    rank_stability(rsna_xai, cub_xai, cub_report, args.output_dir)
    consensus_vs_best(args.rsna_root, args.output_dir)
    efficientnet_gradient_audit(args.rsna_root, args.output_dir)

    stats_rows = architecture_tests(rsna_xai, cub_xai, cub_report)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(args.output_dir / "statistical_tests.csv", index=False)
    write_summary(args.output_dir, stats_rows, cub_report)
    print({"output_dir": str(args.output_dir), "n_statistical_tests": len(stats_rows)})


if __name__ == "__main__":
    main()
