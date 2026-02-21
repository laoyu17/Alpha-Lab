from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BaseCostModel:
    def estimate(self, turnover: pd.Series) -> pd.Series:
        raise NotImplementedError


@dataclass(slots=True)
class LinearCostModel(BaseCostModel):
    commission_bps: float = 2.0
    slippage_bps: float = 5.0

    def estimate(self, turnover: pd.Series) -> pd.Series:
        bps = (self.commission_bps + self.slippage_bps) / 10_000.0
        return turnover.fillna(0.0) * bps
