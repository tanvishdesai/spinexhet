# Repository Map

This repository is organized so paper-writing agents can separate live code, current evidence, archive evidence, and drafting notes quickly.

## Current Paper Inputs

- `docs/reports/current/PROJECT_REPORT_V4.md`: latest project report submitted to mentors.
- `docs/reviews/current/v4-mentor-review.md`: latest mentor feedback.
- `docs/draft_guidance/V4_MENTOR_DRAFTING_GUIDANCE.md`: consolidated writing and positioning guidance extracted from the V4 reviews.
- `artifacts/results/current/v4/`: current V4 result tree.
- `artifacts/analysis/review_gap_analysis/v4_review_gap_analysis/`: V4 diagnostic tables generated from the current results.
- `artifacts/analysis/review_gap_analysis/v4_rsna_aggregate_check/`: flat RSNA aggregate sanity check tables.

## Live Code

- `spine_xnet/`: importable project package with data loading, models, evaluation, XAI, and training utilities.
- `scripts/`: command-line scripts for training, evaluation, XAI, aggregation, audits, and paper diagnostics.
- `configs/`: YAML configs for baselines, ablations, and SpineXNet variants.
- `kaggle_parallel_notebooks/`: RSNA notebooks used for Kaggle execution.
- `cub_200_generalization/`: CUB-200 training, evaluation, XAI, aggregation, and notebooks.
- `concepts/`: concept vocabulary or concept-related local assets.

## Repair Code Added For V4

- `kaggle_parallel_notebooks/repair_v4_rsna_xai.ipynb`: primary repair notebook for EfficientNet gradient explanations, ViT fold 0/1 attention rollout, RSNA repaired aggregation, and Figure 1 disagreement gallery.
- `cub_200_generalization/notebooks/repair_v4_cub_consensus.ipynb`: optional CUB consensus completion notebook for models whose CUB consensus rows were incomplete.
- `scripts/audit_efficientnet_gradients.py`: direct EfficientNet IG/GradientSHAP audit from checkpoint and validation images.
- `scripts/aggregate_rsna_results.py`: RSNA aggregator that can overlay repaired XAI roots on top of original V4 outputs.
- `scripts/generate_disagreement_gallery_v4.py`: qualitative gallery generator for architecture disagreement.
- `scripts/create_v4_repair_notebooks.py`: notebook generator used to create the repair notebooks.

## Documentation Archive

- `docs/reports/archive/`: older project reports.
- `docs/reviews/archive/`: older mentor reviews.
- `docs/planning/`: runbooks, consolidated action plans, and project-planning notes.
- `docs/literature/LR/`: local literature review PDFs and spreadsheet.

## Result Archive

- `artifacts/results/archive/v3/`: archived V3 result tree.
- `artifacts/results/archive/improvement/`: archived intermediate improvement outputs.
- `artifacts/results/archive/early_runs/`: earlier result folder.
- `artifacts/packages/het-spine.zip`: archived project zip.

