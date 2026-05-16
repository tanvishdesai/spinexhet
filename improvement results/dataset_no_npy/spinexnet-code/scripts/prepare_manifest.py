from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.data.manifest import build_train_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build condition-level RSNA training manifest.")
    parser.add_argument("--data-root", type=Path, required=True, help="RSNA Kaggle dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--keep-missing", action="store_true", help="Keep missing labels with label=-1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_train_manifest(args.data_root, drop_missing=not args.keep_missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest):,} samples to {args.output}")


if __name__ == "__main__":
    main()

