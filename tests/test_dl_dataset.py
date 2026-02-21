from __future__ import annotations

import pandas as pd

from alpha_lab.config import DLSpec
from alpha_lab.data import DataPortal
from alpha_lab.dl.dataset import build_sequence_dataset, resolve_feature_frame


def test_resolve_feature_frame_from_factor_names(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    frame = portal.load_daily(symbols=["AAA", "BBB"])
    factor_frame = pd.DataFrame(index=frame.index)
    factor_frame["f1"] = frame["close"].groupby(level=1).pct_change().fillna(0.0)

    spec = DLSpec(enabled=True, feature_factors=["f1"])
    features = resolve_feature_frame(spec, frame, factor_frame)
    assert list(features.columns) == ["f1"]
    assert features.index.equals(frame.index)


def test_build_sequence_dataset_output_contract(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    frame = portal.load_daily(symbols=["AAA", "BBB", "CCC"])
    factor_frame = pd.DataFrame(index=frame.index)
    factor_frame["f1"] = frame["close"].groupby(level=1).pct_change().fillna(0.0)
    spec = DLSpec(enabled=True, feature_factors=["f1"], params={"lookback": 6})

    features = resolve_feature_frame(spec, frame, factor_frame)
    dataset = build_sequence_dataset(spec, frame, features, include_target=True)

    assert dataset.features.ndim == 3
    assert dataset.features.shape[1] == 6
    assert dataset.features.shape[2] == 1
    assert dataset.targets is not None
    assert len(dataset.index) == dataset.features.shape[0]
