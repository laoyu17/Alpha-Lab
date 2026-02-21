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
