# V4 Repair Runbook

The V4 mentor-blocking issues do not require retraining. They require rerunning selected XAI jobs from existing checkpoints.

## What To Run First

Run `kaggle_parallel_notebooks/repair_v4_rsna_xai.ipynb`.

This notebook performs four jobs:

1. Audits EfficientNet-B4 Integrated Gradients vs GradientSHAP from the saved checkpoints.
2. Reruns EfficientNet-B4 XAI folds 0-4 with explicit normalized baselines.
3. Reruns ViT-Small folds 0 and 1 so Attention Rollout no longer fails on the `attn_mask` argument.
4. Aggregates repaired RSNA XAI results and generates a qualitative disagreement gallery figure.

Expected Kaggle output root:

```text
/kaggle/working/v4_repair_rsna/
```

Important outputs:

```text
/kaggle/working/v4_repair_rsna/efficientnet_gradient_audit/summary.csv
/kaggle/working/v4_repair_rsna/combined_xai_multifold/fold_*/efficientnet_b4/
/kaggle/working/v4_repair_rsna/combined_xai_multifold/fold_0/vit_small/
/kaggle/working/v4_repair_rsna/combined_xai_multifold/fold_1/vit_small/
/kaggle/working/v4_repair_rsna/rsna_aggregate/xai_multifold_summary_mean_std_flat.csv
/kaggle/working/v4_repair_rsna/figures/rsna_disagreement_gallery.png
```

## Optional CUB Completion

Run `cub_200_generalization/notebooks/repair_v4_cub_consensus.ipynb` only if you want the CUB consensus table to be complete for ResNet-50, DenseNet-121, EfficientNet-B4, and ViT-Small.

This also does not retrain models. It only reruns CUB XAI/consensus from existing checkpoints.

Expected Kaggle output root:

```text
/kaggle/working/v4_repair_cub_consensus/
```

## Local Aggregation After Downloading Repairs

After downloading the Kaggle repair output, either copy the repaired model folders into `artifacts/results/current/v4/rsna/dataset_no_npy/combined_xai_multifold/` or keep them as a separate repair root and aggregate with both roots:

```powershell
python scripts/aggregate_rsna_results.py `
  --eval-root "artifacts/results/current/v4/rsna/dataset_no_npy/combined_eval_multifold" `
  --xai-roots "artifacts/results/current/v4/rsna/dataset_no_npy/combined_xai_multifold" "PATH_TO_DOWNLOADED/v4_repair_rsna/combined_xai_multifold" `
  --output-dir "artifacts/results/current/v4/rsna/dataset_no_npy/repaired_aggregate"
```

For final paper numbers, prefer the repaired aggregate table over the original V4 table.

## Interpretation Checklist

- EfficientNet is acceptable if IG and GradientSHAP are not pixel-identical, have nonzero map differences, and produce distinct but related faithfulness curves.
- ViT is fixed if fold 0 and fold 1 `xai_summary_v2.json` files include `attention_rollout` in the faithfulness methods and no `attn_mask` failure under `skipped`.
- Figure 1 is ready if `rsna_disagreement_gallery.png` shows the same sample with several architecture maps and a consensus map.
- No classification retraining is needed unless a checkpoint cannot be found.

