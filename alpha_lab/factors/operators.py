from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

Operator = Callable[[pd.Series, dict[str, Any], pd.DataFrame], pd.Series]


def _ensure_multiindex(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError("factor series must use MultiIndex(datetime, symbol)")
    if series.index.nlevels != 2:
        raise ValueError("factor index must have exactly 2 levels")
    return series


def op_rolling_mean(series: pd.Series, params: dict[str, Any], _: pd.DataFrame) -> pd.Series:
    window = int(params.get("window", 5))
    s = _ensure_multiindex(series)
    out = s.groupby(level=1).rolling(window=window, min_periods=1).mean()
    return out.droplevel(0)


def op_rolling_std(series: pd.Series, params: dict[str, Any], _: pd.DataFrame) -> pd.Series:
    window = int(params.get("window", 5))
    s = _ensure_multiindex(series)
    out = s.groupby(level=1).rolling(window=window, min_periods=1).std(ddof=0)
    return out.droplevel(0)


def op_rank(series: pd.Series, _: dict[str, Any], __: pd.DataFrame) -> pd.Series:
    s = _ensure_multiindex(series)
    return s.groupby(level=0).rank(pct=True)


def op_zscore(series: pd.Series, _: dict[str, Any], __: pd.DataFrame) -> pd.Series:
    s = _ensure_multiindex(series)

    def _z(x: pd.Series) -> pd.Series:
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return x * 0.0
        return (x - x.mean()) / std

    return s.groupby(level=0, group_keys=False).apply(_z)


def op_decay_linear(series: pd.Series, params: dict[str, Any], _: pd.DataFrame) -> pd.Series:
    window = int(params.get("window", 5))
    s = _ensure_multiindex(series)
    weights = np.arange(1, window + 1, dtype=float)

    def _decay(x: pd.Series) -> pd.Series:
        arr = x.to_numpy(dtype=float)
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            chunk = arr[start : i + 1]
            w = weights[-len(chunk) :]
            if np.isnan(chunk).all():
                continue
            valid = ~np.isnan(chunk)
            out[i] = float((chunk[valid] * w[valid]).sum() / w[valid].sum())
        return pd.Series(out, index=x.index)

    return s.groupby(level=1, group_keys=False).apply(_decay)


def op_winsorize(series: pd.Series, params: dict[str, Any], _: pd.DataFrame) -> pd.Series:
    lower = float(params.get("lower", 0.01))
    upper = float(params.get("upper", 0.99))
    s = _ensure_multiindex(series)

    def _clip(x: pd.Series) -> pd.Series:
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lo, hi)

    return s.groupby(level=0, group_keys=False).apply(_clip)


def op_neutralize(series: pd.Series, _: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    s = _ensure_multiindex(series)
    if "industry" not in frame.columns or "size" not in frame.columns:
        return s

    out = pd.Series(index=s.index, dtype=float)

    for _dt, x in s.groupby(level=0):
        idx = x.index
        slice_frame = frame.loc[idx]
        y = x.astype(float)

        if y.isna().all():
            out.loc[idx] = np.nan
            continue

        size = slice_frame["size"].astype(float)
        size = (size - size.mean()) / (size.std(ddof=0) + 1e-12)
        industry_dummies = pd.get_dummies(
            slice_frame["industry"],
            drop_first=True,
            dtype=float,
        )

        X = pd.concat([size.rename("size"), industry_dummies], axis=1)
        X = X.fillna(0.0)
        X.insert(0, "const", 1.0)
        X = X.astype(float)

        valid = (~y.isna()) & np.isfinite(y.to_numpy())
        yv = y[valid]
        Xv = X.loc[valid]
        if len(yv) < 3 or Xv.shape[1] >= len(yv):
            out.loc[idx] = y
            continue

        beta = np.linalg.lstsq(Xv.to_numpy(), yv.to_numpy(), rcond=None)[0]
        fitted = pd.Series(X.to_numpy() @ beta, index=idx)
        out.loc[idx] = y - fitted

    return out


def op_fillna(series: pd.Series, params: dict[str, Any], _: pd.DataFrame) -> pd.Series:
    method = str(params.get("method", "ffill"))
    s = _ensure_multiindex(series)

    if method == "ffill":
        return s.groupby(level=1, group_keys=False).ffill()
    if method == "bfill":
        return s.groupby(level=1, group_keys=False).bfill()
    if method == "zero":
        return s.fillna(0.0)
    return s


OPERATOR_REGISTRY: dict[str, Operator] = {
    "rolling_mean": op_rolling_mean,
    "rolling_std": op_rolling_std,
    "rank": op_rank,
    "zscore": op_zscore,
    "decay_linear": op_decay_linear,
    "winsorize": op_winsorize,
    "neutralize": op_neutralize,
    "fillna": op_fillna,
}


def apply_operations(
    base_series: pd.Series,
    operations: list[dict[str, Any]],
    frame: pd.DataFrame,
) -> pd.Series:
    series = base_series.copy()
    target_index = base_series.index
    for op in operations:
        name = str(op.get("op", "")).strip()
        if not name:
            continue
        if name not in OPERATOR_REGISTRY:
            raise ValueError(f"unknown factor operator: {name}")
        series = OPERATOR_REGISTRY[name](series, op, frame)
        series = series.reindex(target_index)
    return series
