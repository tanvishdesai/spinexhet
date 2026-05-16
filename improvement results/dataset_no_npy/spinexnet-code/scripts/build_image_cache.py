from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.constants import COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack crop PNGs into one fast local memmap cache.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_crop_path(raw_path: str, crop_root: Path, manifest_path: Path) -> Path:
    p = Path(str(raw_path))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.extend([p, crop_root / p.name])
    else:
        candidates.extend([crop_root / p.name, crop_root / p, Path.cwd() / p, p])
        candidates.extend(parent / p for parent in [manifest_path.parent, *manifest_path.parents])
        candidates.extend(parent / p.name for parent in [manifest_path.parent, *manifest_path.parents])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find crop for {raw_path!r}; tried crop-root {crop_root}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    if COLS.crop_path not in df.columns:
        raise ValueError("Manifest must contain crop_path.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "images_uint8.npy"
    index_path = args.output_dir / "cache_index.csv"

    arr = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(df), args.image_size, args.image_size),
    )

    index_rows = []
    for idx, row in tqdm(list(df.iterrows()), total=len(df), desc="image-cache"):
        crop_path = resolve_crop_path(str(row[COLS.crop_path]), args.crop_root, args.manifest)
        image = Image.open(crop_path).convert("L").resize((args.image_size, args.image_size), Image.BILINEAR)
        arr[len(index_rows)] = np.asarray(image, dtype=np.uint8)
        index_rows.append(
            {
                COLS.sample_id: str(row[COLS.sample_id]),
                "cache_index": len(index_rows),
            }
        )

    arr.flush()
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    print(
        {
            "event": "cache_done",
            "samples": len(df),
            "image_path": str(image_path),
            "index_path": str(index_path),
            "shape": list(arr.shape),
            "size_mb": round(image_path.stat().st_size / (1024 * 1024), 2),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

