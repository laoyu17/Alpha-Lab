from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import EvalSpec
from alpha_lab.eval import Evaluator


def _build_eval_inputs() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.bdate_range("2024-01-02", periods=20)
    symbols = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])

    frame = pd.DataFrame(index=idx)
    base = np.arange(len(idx), dtype=float) / 100
    frame["close"] = 100 + base + np.sin(np.arange(len(idx)) / 5) * 0.5
    factor = pd.Series(np.cos(np.arange(len(idx)) / 6), index=idx, dtype=float)
    return frame, factor


def test_evaluator_outputs_metrics() -> None:
    frame, factor = _build_eval_inputs()

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


def test_evaluator_annualizer_scales_sharpe_by_periods_per_day() -> None:
    frame, factor = _build_eval_inputs()
    base_eval = Evaluator(
        EvalSpec(
            quantiles=5,
            rolling_window=5,
            walk_forward_train=10,
            walk_forward_test=5,
            trading_days_per_year=252,
            periods_per_day=1,
        )
    )
    minute_like_eval = Evaluator(
        EvalSpec(
            quantiles=5,
            rolling_window=5,
            walk_forward_train=10,
            walk_forward_test=5,
            trading_days_per_year=252,
            periods_per_day=8,
        )
    )

    base_result = base_eval.evaluate(frame, factor)
    minute_result = minute_like_eval.evaluate(frame, factor)

    assert base_result.long_short_returns.equals(minute_result.long_short_returns)

    base_sharpe = float(base_result.metrics["ls_sharpe"])
    minute_sharpe = float(minute_result.metrics["ls_sharpe"])
    assert minute_sharpe == pytest.approx(base_sharpe * math.sqrt(8), rel=1e-9, abs=1e-9)


def test_evaluator_default_annualizer_matches_explicit_daily_defaults() -> None:
    implicit = Evaluator(EvalSpec())
    explicit = Evaluator(EvalSpec(trading_days_per_year=252, periods_per_day=1))
    assert implicit.annualizer == pytest.approx(explicit.annualizer)
