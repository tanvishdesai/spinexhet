"""Utilities for the CUB-200 generalization experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CUB_MODELS: dict[str, str] = {
    "convnext_blackbox": "convnext_tiny",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b4": "efficientnet_b4",
    "vit_small": "vit_small_patch16_224",
}


@dataclass(frozen=True)
class CubSplit:
    train: pd.DataFrame
    val: pd.DataFrame


def find_cub_root(cub_root: str | Path | None = None) -> Path:
    if cub_root is not None:
        root = Path(cub_root)
        if (root / "images.txt").exists() and (root / "images").exists():
            return root
        raise FileNotFoundError(f"CUB root does not look valid: {root}")

    candidates: list[Path] = []
    for base in [Path("/kaggle/input"), Path.cwd()]:
        if not base.exists():
            continue
        candidates += list(base.glob("**/CUB_200_2011"))
    for root in candidates:
        if (root / "images.txt").exists() and (root / "images").exists():
            return root
    raise FileNotFoundError("Could not find CUB_200_2011 under /kaggle/input or the current workspace.")


def _read_space_file(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ", header=None, names=names)


def read_cub_metadata(cub_root: str | Path | None = None) -> pd.DataFrame:
    root = find_cub_root(cub_root)
    images = _read_space_file(root / "images.txt", ["image_id", "rel_path"])
    labels = _read_space_file(root / "image_class_labels.txt", ["image_id", "class_id"])
    classes = _read_space_file(root / "classes.txt", ["class_id", "class_name"])
    split_path = root / "train_test_split.txt"
    if split_path.exists():
        official = _read_space_file(split_path, ["image_id", "is_train"])
    else:
        official = pd.DataFrame({"image_id": images["image_id"], "is_train": 1})

    df = images.merge(labels, on="image_id").merge(classes, on="class_id").merge(official, on="image_id")
    df["label"] = df["class_id"].astype(int) - 1
    df["image_path"] = df["rel_path"].map(lambda p: str(root / "images" / p))
    df["sample_id"] = df["image_id"].map(lambda x: f"cub_{int(x):05d}")
    df["class_name"] = df["class_name"].str.replace(r"^\d+\.", "", regex=True)
    return df.reset_index(drop=True)


def make_cub_split(df: pd.DataFrame, fold: int, n_splits: int = 5, seed: int = 42) -> CubSplit:
    from sklearn.model_selection import StratifiedKFold

    if fold < 0 or fold >= n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}], got {fold}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = df["label"].to_numpy()
    splits = list(splitter.split(np.zeros(len(df)), labels))
    train_idx, val_idx = splits[fold]
    return CubSplit(
        train=df.iloc[train_idx].reset_index(drop=True),
        val=df.iloc[val_idx].reset_index(drop=True),
    )


def make_transforms(image_size: int = 224, train: bool = False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class CUBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 224, train: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = make_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "class_name": str(row["class_name"]),
            "image_path": str(row["image_path"]),
        }


def build_cub_model(model_name: str, num_classes: int = 200, pretrained: bool = True) -> torch.nn.Module:
    import timm

    if model_name not in CUB_MODELS:
        known = ", ".join(sorted(CUB_MODELS))
        raise ValueError(f"Unknown CUB model '{model_name}'. Known models: {known}")
    return timm.create_model(CUB_MODELS[model_name], pretrained=pretrained, num_classes=num_classes)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def cub_classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    pred = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    return {
        "log_loss": float(log_loss(labels, probs, labels=list(range(probs.shape[1])))),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "top5_accuracy": float(np.mean([label in row for label, row in zip(labels, top5)])),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model_from_checkpoint(
    checkpoint: str | Path,
    model_name: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = dict(ckpt.get("config", {}))
    resolved_model = model_name or cfg.get("model_name")
    if not resolved_model:
        raise ValueError("model_name must be provided when checkpoint config has no model_name.")
    model = build_cub_model(resolved_model, num_classes=int(cfg.get("num_classes", 200)), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device), cfg
