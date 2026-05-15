"""Build the condition-level RSNA training manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spine_xnet.constants import (
    COLS,
    CONDITIONS,
    LEVELS,
    SEVERITY_TO_INDEX,
    normalize_condition,
    normalize_level,
    normalize_severity,
    target_column,
)


def _dicom_path(data_root: Path, split: str, study_id: int | str, series_id: int | str, instance_number: int | str) -> str:
    return str(data_root / f"{split}_images" / str(study_id) / str(series_id) / f"{int(instance_number)}.dcm")


def _sample_id(study_id: int | str, series_id: int | str, instance_number: int | str, condition: str, level: str) -> str:
    return f"{study_id}_{series_id}_{int(instance_number)}_{condition}_{level}"


def build_train_manifest(data_root: str | Path, drop_missing: bool = True) -> pd.DataFrame:
    """Create one row per labelled coordinate in the RSNA train split."""

    data_root = Path(data_root)
    train_csv = pd.read_csv(data_root / "train.csv")
    coords = pd.read_csv(data_root / "train_label_coordinates.csv")

    series_path = data_root / "train_series_descriptions.csv"
    series = pd.read_csv(series_path) if series_path.exists() else pd.DataFrame()

    rows: list[dict] = []
    for row in coords.itertuples(index=False):
        study_id = getattr(row, "study_id")
        series_id = getattr(row, "series_id")
        instance_number = getattr(row, "instance_number")
        condition = normalize_condition(getattr(row, "condition"))
        level = normalize_level(getattr(row, "level"))
        target_col = target_column(condition, level)

        if target_col not in train_csv.columns:
            continue

        study_match = train_csv.loc[train_csv["study_id"] == study_id]
        if study_match.empty:
            continue

        severity_raw = study_match.iloc[0][target_col]
        if pd.isna(severity_raw):
            if drop_missing:
                continue
            severity = ""
            label = -1
        else:
            severity = normalize_severity(str(severity_raw))
            label = SEVERITY_TO_INDEX.get(severity)
            if label is None:
                severity = str(severity_raw).strip().lower().replace(" ", "_").replace("/", "_")
                label = SEVERITY_TO_INDEX.get(severity, -1)

        x = float(getattr(row, "x"))
        y = float(getattr(row, "y"))
        sample_id = _sample_id(study_id, series_id, instance_number, condition, level)

        rows.append(
            {
                COLS.sample_id: sample_id,
                COLS.study_id: int(study_id),
                COLS.series_id: int(series_id),
                COLS.instance_number: int(instance_number),
                COLS.condition: condition,
                COLS.condition_idx: CONDITIONS.index(condition) if condition in CONDITIONS else -1,
                COLS.level: level,
                COLS.level_idx: LEVELS.index(level) if level in LEVELS else -1,
                COLS.target_col: target_col,
                COLS.severity: severity,
                COLS.label: int(label),
                COLS.x: x,
                COLS.y: y,
                COLS.dicom_path: _dicom_path(data_root, "train", study_id, series_id, instance_number),
            }
        )

    manifest = pd.DataFrame(rows)
    if not series.empty and not manifest.empty:
        manifest = manifest.merge(series, on=[COLS.study_id, COLS.series_id], how="left")

    manifest = manifest.loc[manifest[COLS.label] >= 0].reset_index(drop=True)
    manifest = add_programmatic_concepts(manifest, train_csv)
    manifest = add_nonleaky_concepts(manifest)
    return manifest


def add_programmatic_concepts(manifest: pd.DataFrame, train_csv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add five pseudo-concepts derived from labels, laterality, and adjacent levels.

    WARNING: pathology_present and severe_grade are derived from the target label.
    These are LEAKY concepts — use only for ablation studies, not the main CBM.
    """

    out = manifest.copy()
    out["concept_pseudo_pathology_present"] = (out[COLS.label] > 0).astype(float)
    out["concept_pseudo_severe_grade"] = (out[COLS.label] == 2).astype(float)
    out["concept_pseudo_left_laterality"] = out[COLS.condition].str.startswith("left_").astype(float)
    out["concept_pseudo_right_laterality"] = out[COLS.condition].str.startswith("right_").astype(float)

    density = []
    key_to_label = {
        (int(r[COLS.study_id]), str(r[COLS.condition]), str(r[COLS.level])): int(r[COLS.label])
        for _, r in out.iterrows()
    }
    for _, r in out.iterrows():
        level_idx = int(r[COLS.level_idx])
        neighbours = []
        for delta in (-1, 1):
            idx = level_idx + delta
            if 0 <= idx < len(LEVELS):
                neighbours.append(key_to_label.get((int(r[COLS.study_id]), str(r[COLS.condition]), LEVELS[idx]), 0))
        if not neighbours:
            density.append(0.0)
        else:
            density.append(float(np.mean([n > 0 for n in neighbours])))
    out["concept_pseudo_adjacent_pathology_density"] = density
    return out


def add_nonleaky_concepts(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add seven non-leaky concepts that describe anatomy/condition type ONLY.

    These concepts never encode severity information and are safe for the
    concept bottleneck without creating a label → label shortcut.

    Concepts:
      - left_laterality: Is this a left-sided condition?
      - right_laterality: Is this a right-sided condition?
      - is_stenosis: Is this spinal canal stenosis?
      - is_foraminal: Is this neural foraminal narrowing?
      - is_subarticular: Is this subarticular stenosis?
      - adjacent_pathology_density: Fraction of adjacent levels with any
        pathology (uses labels from OTHER samples, mild indirect leakage only)
      - level_position: Normalized vertebral level position (0.0 = L1/L2, 1.0 = L5/S1)
    """

    out = manifest.copy()
    # Laterality
    out["concept_nonleaky_left_laterality"] = out[COLS.condition].str.startswith("left_").astype(float)
    out["concept_nonleaky_right_laterality"] = out[COLS.condition].str.startswith("right_").astype(float)
    # Condition type (one-hot style)
    out["concept_nonleaky_is_stenosis"] = (out[COLS.condition] == "spinal_canal_stenosis").astype(float)
    out["concept_nonleaky_is_foraminal"] = out[COLS.condition].str.contains("foraminal").astype(float)
    out["concept_nonleaky_is_subarticular"] = out[COLS.condition].str.contains("subarticular").astype(float)

    # Adjacent pathology density — same as programmatic version
    key_to_label = {
        (int(r[COLS.study_id]), str(r[COLS.condition]), str(r[COLS.level])): int(r[COLS.label])
        for _, r in out.iterrows()
    }
    density = []
    for _, r in out.iterrows():
        level_idx = int(r[COLS.level_idx])
        neighbours = []
        for delta in (-1, 1):
            idx = level_idx + delta
            if 0 <= idx < len(LEVELS):
                neighbours.append(key_to_label.get((int(r[COLS.study_id]), str(r[COLS.condition]), LEVELS[idx]), 0))
        density.append(float(np.mean([n > 0 for n in neighbours])) if neighbours else 0.0)
    out["concept_nonleaky_adjacent_pathology_density"] = density

    # Level position — normalized 0..1 scalar encoding vertebral position
    out["concept_nonleaky_level_position"] = out[COLS.level_idx].astype(float) / max(len(LEVELS) - 1, 1)
    return out


def attach_crop_paths(manifest: pd.DataFrame, crop_root: str | Path) -> pd.DataFrame:
    crop_root = Path(crop_root)
    out = manifest.copy()
    out[COLS.crop_path] = out[COLS.sample_id].map(lambda s: str(crop_root / f"{s}.png"))
    return out

