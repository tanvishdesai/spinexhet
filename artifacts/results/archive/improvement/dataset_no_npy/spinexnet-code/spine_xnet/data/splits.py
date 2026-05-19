"""Group-safe train/validation split helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from spine_xnet.constants import COLS


def make_stratified_group_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    fold_col: str = "fold",
) -> pd.DataFrame:
    out = manifest.copy()
    y = out[COLS.condition_idx].astype(str) + "_" + out[COLS.label].astype(str)
    groups = out[COLS.study_id].astype(str)

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(out, y=y, groups=groups)
    except Exception:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(out, y=y, groups=groups)

    out[fold_col] = -1
    for fold, (_, val_idx) in enumerate(splits):
        out.loc[out.index[val_idx], fold_col] = fold
    return out


def add_site_holdout_split(
    manifest: pd.DataFrame,
    site_col: str | None = None,
    holdout_fraction: float = 0.25,
    seed: int = 42,
    split_col: str = "site_split",
) -> pd.DataFrame:
    """Add train/test tags for a site split.

    RSNA public metadata may not expose acquisition sites. If no site column is
    available, this creates deterministic pseudo-sites from study IDs. Treat
    that fallback as a stress test, not as a true institution split.
    """

    out = manifest.copy()
    rng = np.random.default_rng(seed)

    if site_col is not None and site_col in out.columns:
        sites = sorted(out[site_col].dropna().astype(str).unique().tolist())
        source_col = site_col
    else:
        source_col = "pseudo_site"
        out[source_col] = out[COLS.study_id].map(_pseudo_site)
        sites = sorted(out[source_col].unique().tolist())

    n_holdout = max(1, int(round(len(sites) * holdout_fraction)))
    holdout = set(rng.choice(sites, size=n_holdout, replace=False).tolist())
    out[split_col] = np.where(out[source_col].astype(str).isin(holdout), "test", "train")
    out[f"{split_col}_source_col"] = source_col
    return out


def _pseudo_site(study_id: int | str, n_sites: int = 8) -> str:
    digest = hashlib.sha1(str(study_id).encode("utf-8")).hexdigest()
    return f"pseudo_site_{int(digest[:8], 16) % n_sites}"

