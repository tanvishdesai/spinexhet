from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a sequence of training jobs for configs/folds.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config in args.configs:
        for fold in args.folds:
            out_dir = args.output_root / config.stem / f"fold_{fold}"
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "train.py"),
                "--config",
                str(config),
                "--manifest",
                str(args.manifest),
                "--fold",
                str(fold),
                "--output-dir",
                str(out_dir),
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

