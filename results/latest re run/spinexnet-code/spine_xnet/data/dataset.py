"""PyTorch datasets for condition-level RSNA crop classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from spine_xnet.constants import COLS, DEFAULT_CONCEPTS
from spine_xnet.data.dicom import crop_around_xy, load_dicom_array


def build_transforms(image_size: int = 224, train: bool = True):
    common = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        aug = []
    post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        post.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)))
    return transforms.Compose(common + aug + post)


def build_tensor_transforms(train: bool = True):
    aug = []
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    post = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        post.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)))
    return transforms.Compose(aug + post)


class RSNACropDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        image_size: int = 224,
        train: bool = True,
        concept_columns: Sequence[str] | None = None,
        crop_size: int = 224,
        crop_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        require_crops: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path: Path | None = Path(manifest_path) if manifest_path is not None else None
        if isinstance(manifest, pd.DataFrame):
            self.df = manifest.reset_index(drop=True).copy()
        else:
            self.manifest_path = Path(manifest)
            self.df = pd.read_csv(manifest).reset_index(drop=True)
        self.image_size = image_size
        self.crop_size = crop_size
        self.crop_root = Path(crop_root) if crop_root is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_images: np.ndarray | None = None
        self.cache_indices: np.ndarray | None = None
        self.transform = build_transforms(image_size=image_size, train=train)
        self.tensor_transform = build_tensor_transforms(train=train)
        self.concept_columns = list(concept_columns or [c for c in DEFAULT_CONCEPTS if c in self.df.columns])
        self.require_crops = require_crops

        self.labels = self.df[COLS.label].astype(np.int64).to_numpy()
        self.condition_indices = self.df[COLS.condition_idx].astype(np.int64).to_numpy()
        self.level_indices = self.df[COLS.level_idx].astype(np.int64).to_numpy()
        self.study_ids = self.df[COLS.study_id].astype(np.int64).to_numpy()
        self.sample_ids = self.df[COLS.sample_id].astype(str).tolist()
        self.conditions = self.df[COLS.condition].astype(str).tolist()
        self.levels = self.df[COLS.level].astype(str).tolist()
        self.dicom_paths = self.df[COLS.dicom_path].astype(str).tolist() if COLS.dicom_path in self.df.columns else []
        self.xs = self.df[COLS.x].astype(float).to_numpy() if COLS.x in self.df.columns else np.zeros(len(self.df))
        self.ys = self.df[COLS.y].astype(float).to_numpy() if COLS.y in self.df.columns else np.zeros(len(self.df))
        self._load_cache()
        self.crop_paths = [None] * len(self.df) if self.cache_images is not None else self._resolve_all_crop_paths()
        self.concepts, self.concept_masks = self._build_concept_arrays()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if self.cache_images is not None:
            image_tensor = self._load_cached_tensor(idx)
        else:
            image = self._load_image(idx)
            image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "condition_idx": torch.tensor(int(self.condition_indices[idx]), dtype=torch.long),
            "level_idx": torch.tensor(int(self.level_indices[idx]), dtype=torch.long),
            "concept_targets": torch.from_numpy(self.concepts[idx]),
            "concept_mask": torch.from_numpy(self.concept_masks[idx]),
            "sample_id": self.sample_ids[idx],
            "study_id": int(self.study_ids[idx]),
            "condition": self.conditions[idx],
            "level": self.levels[idx],
        }

    def _load_cache(self) -> None:
        if self.cache_dir is None:
            return

        image_path = self.cache_dir / "images_uint8.npy"
        index_path = self.cache_dir / "cache_index.csv"
        if not image_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"Cache files not found in {self.cache_dir}. Expected images_uint8.npy and cache_index.csv. "
                "Run scripts/build_image_cache.py first."
            )

        cache_index = pd.read_csv(index_path)
        if "cache_index" not in cache_index.columns or COLS.sample_id not in cache_index.columns:
            raise ValueError(f"{index_path} must contain sample_id and cache_index columns.")
        mapper = dict(zip(cache_index[COLS.sample_id].astype(str), cache_index["cache_index"].astype(int)))
        missing = [sid for sid in self.sample_ids if sid not in mapper]
        if missing:
            raise KeyError(f"{len(missing):,} manifest samples are missing from cache. Example: {missing[:3]}")
        self.cache_indices = np.asarray([mapper[sid] for sid in self.sample_ids], dtype=np.int64)
        self.cache_images = np.load(image_path)
        print(
            f"[RSNACropDataset] Using image cache {image_path} with shape {self.cache_images.shape}.",
            flush=True,
        )

    def _load_cached_tensor(self, idx: int) -> torch.Tensor:
        assert self.cache_images is not None
        assert self.cache_indices is not None
        arr = np.asarray(self.cache_images[int(self.cache_indices[idx])], dtype=np.uint8)
        tensor = torch.from_numpy(arr.copy()).float().div_(255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
        return self.tensor_transform(tensor)

    def _load_image(self, idx: int) -> Image.Image:
        crop_path = self.crop_paths[idx]
        if crop_path is not None:
            return Image.open(crop_path).convert("RGB")

        dicom_path = Path(self.dicom_paths[idx])
        arr = load_dicom_array(dicom_path)
        crop = crop_around_xy(arr, float(self.xs[idx]), float(self.ys[idx]), self.crop_size)
        return Image.fromarray(crop).convert("RGB")

    def _resolve_all_crop_paths(self) -> list[Path | None]:
        if COLS.crop_path not in self.df.columns:
            if self.require_crops:
                raise ValueError("Manifest has no crop_path column, but require_crops=True.")
            return [None] * len(self.df)

        resolved: list[Path | None] = []
        missing_examples: list[str] = []
        for raw_path in self.df[COLS.crop_path].astype(str).tolist():
            path = self._resolve_crop_path(raw_path)
            resolved.append(path)
            if path is None and len(missing_examples) < 5:
                missing_examples.append(raw_path)

        missing = sum(path is None for path in resolved)
        if missing:
            message = (
                f"{missing:,}/{len(resolved):,} crop images were not found. "
                "Training will fall back to slow DICOM decoding for those samples. "
                "Pass --crop-root or set data.crop_root to the directory containing crops_224."
            )
            if missing_examples:
                message += f" Examples: {missing_examples}"
            if self.require_crops:
                raise FileNotFoundError(message)
            print(f"[RSNACropDataset] WARNING: {message}", flush=True)
        else:
            root_note = f" from {self.crop_root}" if self.crop_root is not None else ""
            print(f"[RSNACropDataset] Resolved {len(resolved):,} crop images{root_note}.", flush=True)
        return resolved

    def _resolve_crop_path(self, raw_path: str) -> Path | None:
        p = Path(raw_path)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
            if self.crop_root is not None:
                candidates.append(self.crop_root / p.name)
        else:
            if self.crop_root is not None:
                candidates.extend([self.crop_root / p.name, self.crop_root / p])
            candidates.extend([Path.cwd() / p, p])
            if self.manifest_path is not None:
                candidates.extend(parent / p for parent in [self.manifest_path.parent, *self.manifest_path.parents])
                candidates.extend(parent / p.name for parent in [self.manifest_path.parent, *self.manifest_path.parents])

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
        return None

    def _build_concept_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.concept_columns:
            empty = np.zeros((len(self.df), 0), dtype=np.float32)
            return empty, empty

        concept_df = self.df.reindex(columns=self.concept_columns)
        concept_df = concept_df.apply(pd.to_numeric, errors="coerce")
        mask = concept_df.notna().to_numpy(dtype=np.float32)
        values = concept_df.fillna(0.0).to_numpy(dtype=np.float32)
        return values, mask


def load_manifest_for_fold(
    manifest_path: str | Path,
    fold: int = 0,
    fold_col: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest_path)
    train_df = df.loc[df[fold_col] != fold].reset_index(drop=True)
    val_df = df.loc[df[fold_col] == fold].reset_index(drop=True)
    return train_df, val_df
