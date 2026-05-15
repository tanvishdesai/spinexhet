"""Config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config type: {path}")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config_with_base(path: str | Path) -> dict[str, Any]:
    cfg = load_config(path)
    base_path = cfg.pop("base_config", None)
    if base_path is None:
        return cfg
    base_path = Path(path).parent / base_path
    base = load_config_with_base(base_path)
    return deep_update(base, cfg)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

