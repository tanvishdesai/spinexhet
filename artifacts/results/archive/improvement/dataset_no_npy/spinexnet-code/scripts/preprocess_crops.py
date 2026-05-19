from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.constants import COLS
from spine_xnet.data.dicom import crop_around_xy, load_dicom_array, save_png
from spine_xnet.data.manifest import attach_crop_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess RSNA DICOM slices into centered PNG crops.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_row(row: dict, crop_root: Path, crop_size: int, overwrite: bool) -> tuple[str, str]:
    out_path = crop_root / f"{row[COLS.sample_id]}.png"
    if out_path.exists() and not overwrite:
        return row[COLS.sample_id], str(out_path)
    image = load_dicom_array(row[COLS.dicom_path])
    crop = crop_around_xy(image, float(row[COLS.x]), float(row[COLS.y]), crop_size)
    save_png(crop, out_path)
    return row[COLS.sample_id], str(out_path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    args.crop_root.mkdir(parents=True, exist_ok=True)

    records = df.to_dict("records")
    paths: dict[str, str] = {}
    if args.workers <= 1:
        for row in tqdm(records, desc="crops"):
            sid, path = process_row(row, args.crop_root, args.crop_size, args.overwrite)
            paths[sid] = path
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_row, row, args.crop_root, args.crop_size, args.overwrite) for row in records]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="crops"):
                sid, path = fut.result()
                paths[sid] = path

    out = attach_crop_paths(df, args.crop_root)
    out[COLS.crop_path] = out[COLS.sample_id].map(paths)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_manifest, index=False)
    print(f"Wrote crop manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()

