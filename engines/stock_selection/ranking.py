"""Ranking maths for the Stock Selection Engine.

Method: for every feature, convert each symbol's value to a PERCENTILE RANK across the eligible
symbols (0..1, ties averaged). The composite score is the weighted average of those percentiles
over the features that are available for that symbol (weights re-normalised). Percentile ranks
are used instead of raw values so that one extreme outlier cannot dominate the ranking and so
that features with different units can be combined.

Signed features (gap, momentum, relative strength, sector strength) are ranked by absolute size,
because the engine looks for stocks that are *moving* (either way), not for a direction.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from engines.stock_selection.features import ABS_FEATURES


def feature_coverage(features: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Share of total ranking weight that is computable (non-NaN) for each symbol."""
    w = pd.Series(weights, dtype=float)
    available = features[list(w.index)].notna()
    return (available * w).sum(axis=1) / w.sum()


def rank_candidates(features: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Return ``features`` plus ``pct_*`` columns, ``composite_score``, ``feature_coverage``, ``rank``.

    ``features`` is indexed by symbol and contains the SIGNED feature values.
    """
    w = pd.Series(weights, dtype=float)
    w = w[w > 0]
    base = features[list(w.index)].astype(float).copy()
    for col in ABS_FEATURES & set(base.columns):
        base[col] = base[col].abs()
    pct = base.rank(pct=True, method="average")
    available = pct.notna()
    wsum = (available * w).sum(axis=1)
    # A missing feature is NOT scored as zero: it is left out and the remaining weights are
    # re-normalised. A row with nothing available gets NaN (sorted last), never a fake 0.
    weighted = pct.mul(w, axis=1).sum(axis=1, skipna=True, min_count=1)
    composite = weighted / wsum.replace(0, np.nan)

    out = features.copy()
    for col in pct.columns:
        out[f"pct_{col}"] = pct[col]
    out["composite_score"] = composite
    out["feature_coverage"] = wsum / w.sum()
    out = out.reset_index().rename(columns={out.index.name or "index": "symbol"})
    out = out.sort_values(["composite_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
