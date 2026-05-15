"""DICOM loading and crop utilities for RSNA lumbar MRI slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_dicom_array(path: str | Path) -> np.ndarray:
    """Load one DICOM slice as a normalized uint8 array."""

    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
    return (arr * 255.0).astype(np.uint8)


def crop_around_xy(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Crop a square around pixel coordinate (x, y), padding if needed."""

    h, w = image.shape[:2]
    half = size // 2
    cx = int(round(float(x)))
    cy = int(round(float(y)))

    left = cx - half
    top = cy - half
    right = left + size
    bottom = top + size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    return image[top:bottom, left:right]


def save_png(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def load_png_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")

