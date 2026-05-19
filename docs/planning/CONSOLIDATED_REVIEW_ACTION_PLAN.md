# Consolidated V3 Review Action Plan

Last updated: 2026-05-17

## Current Project State

SpineXNet V3 is a strong empirical benchmark: RSNA spine MRI plus CUB-200, 5-fold CV, seven post-hoc methods on RSNA, expert-coordinate ROI alignment, randomization checks, concept intervention, and a cross-domain architecture-ranking story.

The project is not yet submission-ready for BMVC/MIDL because the current evidence has four review-visible gaps:

1. The Transformer claim rests on ViT-Small only.
2. ConvNeXt-CUB failed to converge and should not be included in CUB XAI tables until fixed.
3. The paper has mean/std tables but not enough paired significance tests.
4. The consensus contribution needs a uniform-consensus baseline.

## Reviewer Consensus

| Theme | Reviewers Raising It | Consolidated Decision |
|---|---|---|
| Add a second Transformer | Gemini, Sonnet, Sam alt, GLM, Kimi | Add `deit_small` on RSNA and CUB. DeiT is the lowest-risk second Transformer because the current ViT Attention Rollout code can support it. |
| ConvNeXt-CUB failure | Gemini, Sonnet, GLM, Kimi | Rerun with a gentler ConvNeXt recipe. Until it reaches competent accuracy, exclude it from CUB XAI summaries with a written exclusion report. |
| ViT fold 0/1 Attention Rollout stale | Gemini, Sonnet, GLM, Kimi | Rerun ViT folds 0 and 1 XAI with the fixed rollout. Do not extrapolate in the final paper if GPU time is available. |
| Significance tests | Sam alt, GLM, Kimi | Add paired Wilcoxon/t-tests across folds for DenseNet vs ViT/DeiT and consensus comparisons. |
| Uniform consensus baseline | Kimi, GLM | Add uniform and top-k consensus baselines next to faithfulness-weighted consensus. |
| Feature coherence null result | Gemini, Sonnet, GLM, Kimi | Keep it as supplementary/exploratory. Report CNN-only correlation separately; do not make it a main mechanistic claim. |
| Clinical alignment is low | Kimi | Reframe as evidence that post-hoc XAI remains clinically weak, not as a success claim. |
| Prior-work positioning | Sonnet | Stop claiming "largest XAI benchmark." Position against LATEC and "Hypothesis Class Determines Explanation" as orthogonal: method-metric reliability vs architecture-linked cross-domain explanation disagreement. |
| Code/data release | Sonnet, GLM | Prepare public code, CUB pipeline, aggregation scripts, and saliency/metric artifacts where data licenses allow. |

## Decisions Made In Code

- Added `configs/baselines/deit_small.yaml`.
- Added `deit_small` to RSNA and CUB notebook generation.
- Added model-specific CUB recipes for ConvNeXt and DeiT.
- Added CUB XAI accuracy-threshold exclusion in `aggregate_cub_results.py`.
- Added faithfulness-weighted, uniform, and top-k consensus outputs for RSNA and CUB XAI.
- Added `scripts/analyze_v3_review_gaps.py` for significance tests, rank stability, consensus-vs-best, CUB exclusion, and EfficientNet IG/GradientSHAP audit.
- Added CNN-only/Transformer-only feature coherence correlation reporting.

## Recommended Experiment Order

1. Rerun RSNA `deit_small` training/evaluation/XAI for folds 0-4.
2. Rerun CUB `deit_small` training/evaluation/XAI for folds 0-4.
3. Rerun CUB `convnext_blackbox` with the updated recipe for folds 0-4.
4. Rerun RSNA `vit_small` XAI for folds 0 and 1 only, because those summaries still show the old `attn_mask` rollout failure.
5. Aggregate RSNA and CUB results.
6. Run `python scripts/analyze_v3_review_gaps.py`.
7. Update report tables and manuscript framing from the generated CSVs.

## Paper Positioning

Use the title "Architecture Shapes Explanation" rather than "Architecture Governs Explanation." The evidence supports strong architecture influence, but EfficientNet rank shifts and ConvNeXt-CUB instability make "governs" too deterministic.

The novelty claim should be:

> We study whether disagreement among post-hoc explanations is stable across architecture and domain, and show that the architecture-linked disagreement hierarchy largely reproduces between clinical spine MRI and fine-grained natural images. Unlike broad XAI metric benchmarks, we focus on cross-domain reproduction, clinical expert-coordinate evaluation, and practical consensus/ante-hoc alternatives.

Do not claim to be larger than LATEC. The differentiator is not scale; it is the cross-domain architecture-disagreement story plus clinical deployment context.
