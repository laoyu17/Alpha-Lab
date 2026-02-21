from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.dl.base import DLContext


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return x * 0.0
        return (x - x.mean()) / std

    return series.groupby(level=0, group_keys=False).apply(_z)


def _to_int(value: object, default: int) -> int:
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass(slots=True)
class TemporalMLPStubPlugin:
    """A lightweight placeholder plugin for future DL model replacement.

    This plugin mimics temporal modeling behavior using lag-weighted returns
    and optional factor fusion while keeping dependencies minimal.
    """

    name: str = "temporal_mlp_stub"

    def generate(self, context: DLContext) -> pd.Series:
        close = context.frame["close"].astype(float)
        returns = close.groupby(level=1).pct_change().fillna(0.0)

        lookback = _to_int(context.params.get("lookback", 10), default=10)
        momentum_weight = _to_float(context.params.get("momentum_weight", 0.7), default=0.7)
        fusion_weight = _to_float(context.params.get("fusion_weight", 0.3), default=0.3)

        decay = np.exp(np.linspace(-1.0, 0.0, lookback))
        decay = decay / decay.sum()

        def _temporal_signal(x: pd.Series) -> pd.Series:
            arr = x.to_numpy(dtype=float)
            out = np.full_like(arr, np.nan)
            for i in range(len(arr)):
                start = max(0, i - lookback + 1)
                chunk = arr[start : i + 1]
                w = decay[-len(chunk) :]
                out[i] = float(np.dot(chunk, w))
            return pd.Series(out, index=x.index)

        temporal_signal = returns.groupby(level=1, group_keys=False).apply(_temporal_signal)
        temporal_signal = _cross_sectional_zscore(temporal_signal)

        feature_names = context.feature_factors
        if not feature_names:
            feature_names = context.factor_frame.columns.tolist()

        if feature_names:
            selected = [name for name in feature_names if name in context.factor_frame.columns]
            if selected:
                factor_fusion = context.factor_frame[selected].mean(axis=1).astype(float)
                factor_fusion = _cross_sectional_zscore(factor_fusion)
            else:
                factor_fusion = pd.Series(0.0, index=temporal_signal.index)
        else:
            factor_fusion = pd.Series(0.0, index=temporal_signal.index)

        dl_factor = momentum_weight * temporal_signal + fusion_weight * factor_fusion
        return _cross_sectional_zscore(dl_factor).rename(context.output_name)
