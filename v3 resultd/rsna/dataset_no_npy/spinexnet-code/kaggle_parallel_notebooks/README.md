# Kaggle Parallel Notebooks

These notebooks are the append pass for the existing SpineXNet Kaggle outputs.
For each model, attach that model's previous notebook-output dataset, then run
the same notebook again. The setup cell copies the prior `outputs/`,
`eval_multifold/`, and `xai_multifold/` trees into `/kaggle/working`, trains
folds 3 and 4, and leaves one combined output dataset containing folds 0-4.

## Per-Model Notebooks

- `train_eval_xai_convnext_blackbox.ipynb`
- `train_eval_xai_cbm_nonleaky.ipynb`
- `train_eval_xai_cbm_leaky.ipynb`
- `train_eval_xai_resnet50.ipynb`
- `train_eval_xai_densenet121.ipynb`
- `train_eval_xai_efficientnet_b4.ipynb`
- `train_eval_xai_vit_small.ipynb`

Each notebook defaults to:

- `FOLDS_TO_TRAIN = [3, 4]`
- `EVAL_FOLDS = [0, 1, 2, 3, 4]`
- `XAI_FOLDS = [0, 1, 2, 3, 4]`

If you only want a quick append pass, change `XAI_FOLDS` to `[3, 4]`.
Leaving it as `0..4` also fills any missing fold-2 XAI output from the previous
run and gives a clean 5-fold XAI table.

## Aggregation

After all seven per-model notebooks finish:

1. Save/update each notebook output as a Kaggle dataset.
2. Attach all seven output datasets to `aggregate_parallel_results.ipynb`.
3. Run the aggregate notebook.

The aggregate notebook now also runs `scripts/feature_map_smoothness.py`, the
replacement for the old input-gradient GLS experiment.

## Output To Keep

Each per-model notebook writes:

- `/kaggle/working/outputs/<experiment>/fold_*/best.pt`
- `/kaggle/working/eval_multifold/<model>/fold_*/metrics_val.json`
- `/kaggle/working/xai_multifold/fold_*/<model>/...`
- `/kaggle/working/classification_metrics_<model>_by_fold.csv`
- `/kaggle/working/xai_multifold_summary_<model>.csv`

The aggregation notebook writes:

- `/kaggle/working/classification_metrics_by_fold.csv`
- `/kaggle/working/classification_metrics_mean_std.csv`
- `/kaggle/working/xai_multifold_summary_by_fold.csv`
- `/kaggle/working/xai_multifold_summary_mean_std.csv`
- `/kaggle/working/feature_map_smoothness/...`
- `/kaggle/working/figures/...`
