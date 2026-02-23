from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.config import GuardSpec
from alpha_lab.guards import GuardSuite


def test_apply_price_adjustment_adjusts_ohlc() -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "AAA"),
            (pd.Timestamp("2024-01-03"), "AAA"),
        ],
        names=["datetime", "symbol"],
    )
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [10.2, 10.8],
            "low": [9.8, 10.2],
            "close": [10.1, 10.6],
            "adj_factor": [1.1, 0.9],
        },
        index=idx,
    )

    suite = GuardSuite(GuardSpec())
    adjusted = suite.apply_price_adjustment(frame)

    factor = frame["adj_factor"].to_numpy(dtype=float)
    for col in ("open", "high", "low", "close"):
        expected = frame[col].to_numpy(dtype=float) * factor
        np.testing.assert_allclose(adjusted[col].to_numpy(dtype=float), expected)


def test_check_operator_rules_blocks_future_like_names() -> None:
    suite = GuardSuite(GuardSpec())
    issue = suite.check_operator_rules([{"op": "future_mean"}, {"op": "rank"}])
    assert issue is not None
    assert issue.level == "error"


def test_check_operator_rules_allows_normal_ops() -> None:
    suite = GuardSuite(GuardSpec())
    issue = suite.check_operator_rules([{"op": "rolling_mean"}, {"op": "zscore"}])
    assert issue is None


def test_apply_tradable_filter_filters_suspended_limit_and_abnormal_move() -> None:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "AAA"),
            (pd.Timestamp("2024-01-03"), "AAA"),
            (pd.Timestamp("2024-01-04"), "AAA"),
            (pd.Timestamp("2024-01-05"), "AAA"),
        ],
        names=["datetime", "symbol"],
    )
    frame = pd.DataFrame(
        {
            "close": [10.0, 10.3, 11.5, 11.6],
            "suspended": [False, True, False, False],
            "limit_hit": [False, False, True, False],
        },
        index=idx,
    )
    factor = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    suite = GuardSuite(GuardSpec(limit_move_threshold=0.095))
    filtered = suite.apply_tradable_filter(factor, frame)

    assert filtered.iloc[0] == 1.0
    assert np.isnan(filtered.iloc[1])
    assert np.isnan(filtered.iloc[2])
    assert filtered.iloc[3] == 4.0


def test_check_future_leakage_threshold_behavior() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=80)
    symbols = [f"S{i:02d}" for i in range(20)]
    returns = rng.normal(loc=0.0, scale=0.02, size=(len(dates), len(symbols)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)

    idx = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "symbol"])
    frame = pd.DataFrame({"close": prices.reshape(-1)}, index=idx)

    same_day_ret = frame["close"].groupby(level=1).pct_change()
    next_day_ret = frame["close"].groupby(level=1).shift(-1) / frame["close"] - 1.0

    suite = GuardSuite(GuardSpec(enable_future_check=True))
    issue_for_same_day_signal = suite.check_future_leakage(same_day_ret, frame)
    issue_for_next_day_signal = suite.check_future_leakage(next_day_ret, frame)

    assert issue_for_same_day_signal is not None
    assert issue_for_same_day_signal.name == "future_leakage"
    assert issue_for_next_day_signal is None
