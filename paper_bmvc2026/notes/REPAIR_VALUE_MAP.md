# Repair Value Map

This file maps the two pending repair notebooks to exact manuscript locations.
Search the LaTeX source for `REPAIR PLACEHOLDER` to find every visible marker.

## RSNA Repair Notebook

Notebook:
`kaggle_parallel_notebooks/repair_v4_rsna_xai.ipynb`

Purpose:
- Recompute EfficientNet-B4 XAI artifacts from existing checkpoints.
- Recompute ViT-Small folds 0 and 1 with Attention Rollout enabled.
- Aggregate original V4 outputs plus repaired folders.
- Generate a high-resolution Figure 1 disagreement gallery.

Replace in manuscript:
- Abstract consensus sentence if RSNA consensus values change.
- Introduction contribution 3 if final consensus or ViT values change.
- Section 3.4 sentence beginning `For each trained model and fold...`.
- Table 2 ViT-Small RSNA row: Spearman rho and Top-20 IoU.
- Figure 1 image and caption placeholder.
- Any RSNA EfficientNet statements if repaired EfficientNet aggregates differ.

Expected output files to use:
- `v4_repair_rsna_xai/aggregate/xai_multifold_summary_mean_std.csv`
- `v4_repair_rsna_xai/aggregate/xai_multifold_summary_by_fold.csv`
- `v4_repair_rsna_xai/aggregate/faithfulness_mean_std_by_method_flat.csv`
- `v4_repair_rsna_xai/figures/*disagreement_gallery*.png`

## CUB Consensus Repair Notebook

Notebook:
`cub_200_generalization/notebooks/repair_v4_cub_consensus.ipynb`

Purpose:
- Rerun the four older CUB XAI jobs with the updated consensus code.
- Produce uniform, faithfulness-weighted, and top-k consensus rows for all six CUB models.
- Aggregate original CUB results plus repaired consensus outputs.

Replace in manuscript:
- Abstract consensus sentence.
- Section 3.4 final consensus-scope sentence.
- Section 4.4 consensus paragraph.
- Figure 3 / `figures/consensus_ablation.pdf`.
- Limitations sentence that currently restricts CUB top-k claims.

Expected output files to use:
- `v4_repair_cub_consensus/cub_aggregate/cub_xai_summary_mean_std.csv`
- `v4_repair_cub_consensus/cub_aggregate/cub_xai_summary_by_fold.csv`
- `v4_repair_cub_consensus/cub_xai/**/consensus_metrics.csv`

## Placeholder Search Tokens

- `REPAIR PLACEHOLDER: replace consensus sentence after RSNA and CUB repair notebooks finish`
- `REPAIR PLACEHOLDER: refresh consensus and ViT-rollout numbers after repair notebooks finish`
- `REPAIR PLACEHOLDER: update ViT-Small folds 0--1 once Attention Rollout rerun completes`
- `REPAIR PLACEHOLDER: after the CUB repair finishes, replace this sentence with the final global-or-oracle consensus scope`
- `REPAIR PLACEHOLDER: replace ViT-Small RSNA row if RSNA repair changes folds 0--1 Attention Rollout`
- `REPAIR PLACEHOLDER: CUB repair: replace this paragraph and Figure`
- `REPAIR PLACEHOLDER: remove this limitation after repair_v4_cub_consensus.ipynb completes`
- `REPAIR PLACEHOLDER: insert anonymous GitHub/Zenodo URL`
