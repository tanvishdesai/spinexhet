from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect metric JSON files into one CSV table.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="**/metrics_*.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.root.glob(args.pattern):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"path": str(path)}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

