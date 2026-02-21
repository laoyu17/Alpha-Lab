from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.config import EvalSpec
from alpha_lab.eval import Evaluator


def test_evaluator_outputs_metrics() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    symbols = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])

    frame = pd.DataFrame(index=idx)
    base = np.arange(len(idx), dtype=float) / 100
    frame["close"] = 100 + base + np.sin(np.arange(len(idx)) / 5) * 0.5
    factor = pd.Series(np.cos(np.arange(len(idx)) / 6), index=idx, dtype=float)

    evaluator = Evaluator(
        EvalSpec(
            quantiles=5,
            rolling_window=5,
            walk_forward_train=10,
            walk_forward_test=5,
        )
    )
    result = evaluator.evaluate(frame, factor)

    assert "ic_mean" in result.metrics
    assert "rank_ic_mean" in result.metrics
    assert "ic_roll_mean_last" in result.metrics
    assert "ic_roll_ir_last" in result.metrics
    assert isinstance(result.quantile_returns, pd.DataFrame)
    assert isinstance(result.long_short_returns, pd.Series)
