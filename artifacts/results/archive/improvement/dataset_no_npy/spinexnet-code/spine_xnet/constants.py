"""Shared constants for the RSNA lumbar spine classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass


SEVERITY_TO_INDEX = {
    "normal_mild": 0,
    "normal/mild": 0,
    "normal mild": 0,
    "moderate": 1,
    "severe": 2,
}

INDEX_TO_SEVERITY = {
    0: "normal_mild",
    1: "moderate",
    2: "severe",
}

SEVERITY_CLASS_WEIGHTS = [1.0, 2.0, 4.0]

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

CONDITION_ALIASES = {
    "spinal canal stenosis": "spinal_canal_stenosis",
    "left neural foraminal narrowing": "left_neural_foraminal_narrowing",
    "right neural foraminal narrowing": "right_neural_foraminal_narrowing",
    "left subarticular stenosis": "left_subarticular_stenosis",
    "right subarticular stenosis": "right_subarticular_stenosis",
}

LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

LEVEL_ALIASES = {
    "l1/l2": "l1_l2",
    "l2/l3": "l2_l3",
    "l3/l4": "l3_l4",
    "l4/l5": "l4_l5",
    "l5/s1": "l5_s1",
}

PROGRAMMATIC_CONCEPTS = [
    "concept_pseudo_pathology_present",
    "concept_pseudo_severe_grade",
    "concept_pseudo_left_laterality",
    "concept_pseudo_right_laterality",
    "concept_pseudo_adjacent_pathology_density",
]

# Leaky concepts (derived from target labels) — kept for ablation only.
# concept_pseudo_pathology_present and concept_pseudo_severe_grade leak the
# severity label the classifier predicts, creating a shortcut through the
# concept bottleneck.
LEAKY_CONCEPTS = PROGRAMMATIC_CONCEPTS

# Non-leaky concepts: describe WHAT is being examined (anatomy/condition),
# never HOW SEVERE it is. The CBM must learn severity from visual features.
NON_LEAKY_CONCEPTS = [
    "concept_nonleaky_left_laterality",
    "concept_nonleaky_right_laterality",
    "concept_nonleaky_is_stenosis",
    "concept_nonleaky_is_foraminal",
    "concept_nonleaky_is_subarticular",
    "concept_nonleaky_adjacent_pathology_density",
    "concept_nonleaky_level_position",
]

BIOMEDCLIP_CONCEPTS = [
    "concept_biomedclip_disc_height_loss",
    "concept_biomedclip_disc_signal_loss",
    "concept_biomedclip_endplate_change",
    "concept_biomedclip_osteophyte_formation",
    "concept_biomedclip_facet_hypertrophy",
    "concept_biomedclip_ligamentum_flavum_thickening",
    "concept_biomedclip_spinal_canal_narrowing",
    "concept_biomedclip_neural_foraminal_narrowing",
    "concept_biomedclip_disc_bulge_or_herniation",
    "concept_biomedclip_abnormal_vertebral_alignment",
]

# Default for v2 pipeline: use non-leaky concepts
DEFAULT_CONCEPTS = NON_LEAKY_CONCEPTS

# Visual concepts for the improved CBM (Tier 2 Contribution 3).
# These use BiomedCLIP zero-shot scoring — see extract_biomedclip_concepts.py.
# Replaces metadata-derived concepts with clinically grounded visual signals.
VISUAL_CONCEPTS = BIOMEDCLIP_CONCEPTS

# Image-derived visual signal concepts (computed from pixel data, no model needed)
IMAGE_SIGNAL_CONCEPTS = [
    "concept_signal_canal_ratio",          # central vs peripheral intensity
    "concept_signal_bilateral_symmetry",   # L-R asymmetry index
    "concept_signal_dark_fraction",        # fraction of dark pixels (disc dehydration)
    "concept_signal_gradient_variance",    # structural boundary sharpness
    "concept_signal_band_intensity",       # horizontal middle-band intensity (disc level)
    "concept_signal_heterogeneity",        # local variance across sub-regions
]

BIOMEDCLIP_PROMPTS = {
    "concept_biomedclip_disc_height_loss": [
        "lumbar spine MRI showing loss of intervertebral disc height",
        "disc space narrowing in the lumbar spine",
    ],
    "concept_biomedclip_disc_signal_loss": [
        "T2 hypointense degenerative lumbar disc desiccation",
        "loss of disc signal intensity on lumbar MRI",
    ],
    "concept_biomedclip_endplate_change": [
        "Modic endplate changes adjacent to lumbar disc",
        "degenerative vertebral endplate signal change",
    ],
    "concept_biomedclip_osteophyte_formation": [
        "lumbar vertebral osteophyte formation",
        "bone spur at the lumbar disc margin",
    ],
    "concept_biomedclip_facet_hypertrophy": [
        "facet joint hypertrophy in the lumbar spine",
        "degenerative facet arthropathy on lumbar MRI",
    ],
    "concept_biomedclip_ligamentum_flavum_thickening": [
        "thickened ligamentum flavum causing lumbar stenosis",
        "hypertrophy of ligamentum flavum on axial lumbar MRI",
    ],
    "concept_biomedclip_spinal_canal_narrowing": [
        "lumbar spinal canal narrowing stenosis",
        "central canal stenosis on lumbar spine MRI",
    ],
    "concept_biomedclip_neural_foraminal_narrowing": [
        "neural foraminal narrowing in lumbar spine",
        "lumbar foraminal stenosis compressing exiting nerve root",
    ],
    "concept_biomedclip_disc_bulge_or_herniation": [
        "lumbar disc bulge or disc herniation",
        "posterior disc protrusion on lumbar spine MRI",
    ],
    "concept_biomedclip_abnormal_vertebral_alignment": [
        "abnormal lumbar vertebral alignment spondylolisthesis",
        "listhesis of lumbar vertebral body",
    ],
}


@dataclass(frozen=True)
class DataColumns:
    sample_id: str = "sample_id"
    study_id: str = "study_id"
    series_id: str = "series_id"
    instance_number: str = "instance_number"
    condition: str = "condition"
    condition_idx: str = "condition_idx"
    level: str = "level"
    level_idx: str = "level_idx"
    target_col: str = "target_col"
    severity: str = "severity"
    label: str = "label"
    x: str = "x"
    y: str = "y"
    dicom_path: str = "dicom_path"
    crop_path: str = "crop_path"
    fold: str = "fold"


COLS = DataColumns()


def normalize_condition(value: str) -> str:
    key = str(value).strip().lower().replace("_", " ")
    if key in CONDITION_ALIASES:
        return CONDITION_ALIASES[key]
    return key.replace(" ", "_")


def normalize_level(value: str) -> str:
    key = str(value).strip().lower().replace("_", "/")
    if key in LEVEL_ALIASES:
        return LEVEL_ALIASES[key]
    return key.replace("/", "_")


def normalize_severity(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.replace("/", "_")
    if key == "normal_mild":
        return key
    return str(value).strip().lower().replace(" ", "_")


def target_column(condition: str, level: str) -> str:
    return f"{normalize_condition(condition)}_{normalize_level(level)}"

