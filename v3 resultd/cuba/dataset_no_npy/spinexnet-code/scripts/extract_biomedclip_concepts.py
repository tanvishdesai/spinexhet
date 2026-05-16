from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.constants import BIOMEDCLIP_CONCEPTS, BIOMEDCLIP_PROMPTS, COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract label-free concept scores with BiomedCLIP.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=None,
        help="Optional directory containing crop PNGs. Useful when manifest crop_path values are relative.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    return parser.parse_args()


def load_biomedclip(model_name: str, device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def resolve_crop_path(raw_path: str, manifest_path: Path, crop_root: Path | None = None) -> Path:
    """Resolve crop paths after moving manifests/crops into Kaggle Datasets."""

    p = Path(str(raw_path))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
        if crop_root is not None:
            candidates.append(crop_root / p.name)
    else:
        if crop_root is not None:
            candidates.extend([crop_root / p.name, crop_root / p])
        candidates.extend([Path.cwd() / p, p])
        candidates.extend(parent / p for parent in [manifest_path.parent, *manifest_path.parents])
        candidates.extend(parent / p.name for parent in [manifest_path.parent, *manifest_path.parents])

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched = "\n  - ".join(str(c) for c in unique_candidates[:12])
    raise FileNotFoundError(
        f"Could not find crop image for manifest path {raw_path!r}.\n"
        f"Pass --crop-root pointing at the directory containing the PNG crops.\n"
        f"Searched:\n  - {searched}"
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.manifest)
    if COLS.crop_path not in df.columns:
        raise ValueError("Manifest must contain crop_path. Run preprocess_crops.py first.")

    model, preprocess, tokenizer = load_biomedclip(args.model_name, device)
    prompt_groups = [BIOMEDCLIP_PROMPTS[c] for c in BIOMEDCLIP_CONCEPTS]
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    tokens = tokenizer(flat_prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    group_slices = []
    start = 0
    for group in prompt_groups:
        end = start + len(group)
        group_slices.append(slice(start, end))
        start = end

    sample_ids = []
    raw_scores = []
    for start_idx in tqdm(range(0, len(df), args.batch_size), desc="BiomedCLIP"):
        batch_df = df.iloc[start_idx : start_idx + args.batch_size]
        images = []
        for path in batch_df[COLS.crop_path]:
            resolved_path = resolve_crop_path(path, args.manifest, args.crop_root)
            images.append(preprocess(Image.open(resolved_path).convert("RGB")))
        image_tensor = torch.stack(images).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ text_features.t()

        concept_scores = []
        for sl in group_slices:
            concept_scores.append(sims[:, sl].mean(dim=1))
        scores = torch.stack(concept_scores, dim=1)
        raw_scores.append(scores.detach().cpu().numpy())
        sample_ids.extend(batch_df[COLS.sample_id].astype(str).tolist())

    score_arr = np.concatenate(raw_scores, axis=0)
    score_arr = (score_arr - score_arr.mean(axis=0, keepdims=True)) / np.clip(score_arr.std(axis=0, keepdims=True), 1e-6, None)
    score_arr = 1.0 / (1.0 + np.exp(-score_arr))
    out_df = pd.DataFrame({"sample_id": sample_ids})
    for j, concept in enumerate(BIOMEDCLIP_CONCEPTS):
        out_df[concept] = score_arr[:, j].astype(float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote BiomedCLIP concept scores to {args.output}")


if __name__ == "__main__":
    main()
