from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.data.splits import add_site_holdout_split, make_stratified_group_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add CV folds and optional site-holdout split to a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-col", type=str, default=None)
    parser.add_argument("--site-holdout-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = make_stratified_group_folds(df, n_splits=args.folds, seed=args.seed)
    df = add_site_holdout_split(df, site_col=args.site_col, holdout_fraction=args.site_holdout_fraction, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote split manifest to {args.output}")
    print(df["fold"].value_counts().sort_index().to_string())
    print(df["site_split"].value_counts().to_string())


if __name__ == "__main__":
    main()

