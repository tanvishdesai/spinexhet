from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one or more concept CSV files into the manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    for concept_path in args.concepts:
        concept_df = pd.read_csv(concept_path)
        df = df.merge(concept_df, on="sample_id", how="left", suffixes=("", "_new"))
        duplicate_new = [c for c in df.columns if c.endswith("_new")]
        for new_col in duplicate_new:
            old_col = new_col[:-4]
            df[old_col] = df[old_col].fillna(df[new_col])
            df = df.drop(columns=[new_col])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote merged manifest to {args.output}")


if __name__ == "__main__":
    main()

