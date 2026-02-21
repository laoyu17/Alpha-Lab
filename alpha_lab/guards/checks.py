from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.config import GuardSpec
from alpha_lab.types import GuardIssue, GuardReport


@dataclass(slots=True)
class GuardSuite:
    spec: GuardSpec

    def apply_price_adjustment(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "adj_factor" not in frame.columns:
            return frame
        adjusted = frame.copy()
        factor = adjusted["adj_factor"].astype(float)
        for price_col in ("open", "high", "low", "close"):
            if price_col in adjusted.columns:
                adjusted[price_col] = adjusted[price_col].astype(float) * factor
        return adjusted

    def apply_tradable_filter(self, factor: pd.Series, frame: pd.DataFrame) -> pd.Series:
        if not self.spec.enforce_tradable_filter:
            return factor

        mask = pd.Series(True, index=factor.index)
        if "suspended" in frame.columns:
            mask &= ~frame["suspended"].astype(bool)
        if "limit_hit" in frame.columns:
            mask &= ~frame["limit_hit"].astype(bool)
        if "close" in frame.columns:
            prev_close = frame["close"].groupby(level=1).shift(1)
            move = (frame["close"] / prev_close - 1.0).abs()
            mask &= (move <= self.spec.limit_move_threshold) | move.isna()

        filtered = factor.copy()
        filtered[~mask] = np.nan
        return filtered

    def check_future_leakage(self, factor: pd.Series, frame: pd.DataFrame) -> GuardIssue | None:
        if not self.spec.enable_future_check:
            return None
        if "close" not in frame.columns:
            return None

        same_day_ret = frame["close"].groupby(level=1).pct_change()
        next_day_ret = frame["close"].groupby(level=1).shift(-1) / frame["close"] - 1.0

        aligned_same = pd.concat([factor.rename("f"), same_day_ret.rename("r")], axis=1).dropna()
        aligned_next = pd.concat([factor.rename("f"), next_day_ret.rename("r")], axis=1).dropna()

        if aligned_same.empty or aligned_next.empty:
            return None

        same_ic = aligned_same.groupby(level=0).apply(lambda x: x["f"].corr(x["r"]))
        next_ic = aligned_next.groupby(level=0).apply(lambda x: x["f"].corr(x["r"]))

        same_mean = float(np.nanmean(same_ic))
        next_mean = float(np.nanmean(next_ic))

        if abs(same_mean) > abs(next_mean) * 1.5 and abs(same_mean) > 0.08:
            return GuardIssue(
                name="future_leakage",
                level="warning",
                message=(
                    "Potential leakage: same-day IC is significantly stronger than next-day IC "
                    f"(same={same_mean:.4f}, next={next_mean:.4f})."
                ),
            )
        return None

    def check_operator_rules(self, operations: list[dict[str, object]]) -> GuardIssue | None:
        bad_ops: list[str] = []
        for op in operations:
            name = str(op.get("op", "")).lower()
            if "lead" in name or "future" in name:
                bad_ops.append(name)

        if bad_ops:
            return GuardIssue(
                name="operator_rule",
                level="error",
                message=f"Forbidden operators detected: {', '.join(bad_ops)}",
            )
        return None

    def build_report(self, issues: list[GuardIssue]) -> GuardReport:
        return GuardReport(issues=issues)
