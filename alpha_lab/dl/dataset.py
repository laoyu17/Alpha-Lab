from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from alpha_lab.config import DLSpec


@dataclass(slots=True)
class DLDataset:
    features: np.ndarray
    targets: np.ndarray | None
    index: pd.MultiIndex
    feature_names: list[str]


def resolve_feature_frame(
    spec: DLSpec,
    frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
) -> pd.DataFrame:
    available: dict[str, pd.Series] = {}

    for column in factor_frame.columns:
        available[str(column)] = factor_frame[column].astype(float)

    numeric_frame = frame.select_dtypes(include=[np.number])
    for column in numeric_frame.columns:
        name = str(column)
        if name not in available:
            available[name] = numeric_frame[column].astype(float)

    if "close" in frame.columns and "close_ret_1" not in available:
        close = frame["close"].astype(float)
        available["close_ret_1"] = close.groupby(level=1).pct_change()

    selected = spec.features or spec.feature_factors or factor_frame.columns.tolist()
    if not selected and "close_ret_1" in available:
        selected = ["close_ret_1"]

    if not selected:
        raise ValueError("no DL features available, please set dl.features or factor specs")

    missing = [name for name in selected if str(name) not in available]
    if missing:
        raise ValueError(f"dl feature columns not found: {', '.join(sorted(set(missing)))}")

    columns: dict[str, pd.Series] = {name: available[str(name)] for name in selected}
    feature_frame = pd.DataFrame(columns, index=frame.index).sort_index()
    return feature_frame.astype(float)


def _to_int(params: dict[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _build_labels(frame: pd.DataFrame, horizon: int) -> pd.Series:
    close = frame["close"].astype(float)
    return close.groupby(level=1).shift(-horizon) / close - 1.0


def build_sequence_dataset(
    spec: DLSpec,
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    include_target: bool,
) -> DLDataset:
    lookback = max(2, _to_int(spec.params, "lookback", 20))
    horizon = max(1, _to_int(spec.params, "horizon", 1))

    labels = _build_labels(frame, horizon) if include_target else None
    sample_features: list[np.ndarray] = []
    sample_targets: list[float] = []
    sample_index: list[tuple[pd.Timestamp, str]] = []

    label_series = cast(pd.Series, labels) if include_target else None

    for symbol in feature_frame.index.get_level_values(1).unique():
        symbol_features = feature_frame.xs(symbol, level=1).sort_index()
        if symbol_features.empty:
            continue

        values = symbol_features.to_numpy(dtype=np.float32)
        if include_target:
            symbol_labels = (
                cast(pd.Series, label_series)
                .xs(symbol, level=1)
                .reindex(symbol_features.index)
                .to_numpy(dtype=np.float32)
            )
        else:
            symbol_labels = None

        for i in range(lookback - 1, len(symbol_features)):
            window = values[i - lookback + 1 : i + 1]
            if not np.isfinite(window).all():
                continue
            if include_target:
                assert symbol_labels is not None
                target = symbol_labels[i]
                if not np.isfinite(target):
                    continue
                sample_targets.append(float(target))
            dt = symbol_features.index[i]
            sample_index.append((pd.Timestamp(dt), str(symbol)))
            sample_features.append(window)

    if sample_features:
        features = np.stack(sample_features).astype(np.float32)
    else:
        features = np.empty((0, lookback, feature_frame.shape[1]), dtype=np.float32)

    targets = np.array(sample_targets, dtype=np.float32) if include_target else None
    index = pd.MultiIndex.from_tuples(sample_index, names=["datetime", "symbol"])

    return DLDataset(
        features=features,
        targets=targets,
        index=index,
        feature_names=[str(col) for col in feature_frame.columns],
    )
