from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.factors import apply_operations


def _mock_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["datetime", "symbol"])
    frame = pd.DataFrame(index=idx)
    frame["close"] = np.arange(len(idx), dtype=float) + 10
    frame["size"] = [10, 11, 9] * len(dates)
    frame["industry"] = ["tech", "finance", "tech"] * len(dates)
    return frame


def test_apply_rolling_rank_zscore() -> None:
    frame = _mock_frame()
    base = frame["close"]
    out = apply_operations(
        base,
        [{"op": "rolling_mean", "window": 2}, {"op": "rank"}, {"op": "zscore"}],
        frame,
    )
    assert out.index.equals(base.index)
    assert out.notna().sum() > 0


def test_apply_neutralize() -> None:
    frame = _mock_frame()
    base = frame["close"]
    out = apply_operations(base, [{"op": "neutralize"}], frame)
    assert out.index.equals(base.index)
    assert np.isfinite(out.dropna()).all()
