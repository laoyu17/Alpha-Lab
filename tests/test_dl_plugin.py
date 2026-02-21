from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.config import DLSpec
from alpha_lab.data import DataPortal
from alpha_lab.dl import run_dl_factor, run_dl_plugin


def test_dl_plugin_output_shape(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    frame = portal.load_daily(symbols=["AAA", "BBB", "CCC"])
    factor_frame = pd.DataFrame(index=frame.index)

    spec = DLSpec(
        enabled=True,
        plugin="temporal_mlp_stub",
        output_name="dl_alpha",
        feature_factors=[],
        params={"lookback": 8},
    )
    series = run_dl_plugin(spec, frame, factor_frame)

    assert isinstance(series.index, pd.MultiIndex)
    assert series.index.equals(frame.index)
    assert series.name == "dl_alpha"


def test_dl_factor_requires_checkpoint_in_infer_mode(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    frame = portal.load_daily(symbols=["AAA", "BBB"])
    factor_frame = pd.DataFrame(index=frame.index)

    spec = DLSpec(
        enabled=True,
        mode="infer",
        model_type="tcn",
        output_name="dl_alpha",
    )
    with pytest.raises(ValueError, match="dl.checkpoint_path is required"):
        run_dl_factor(spec, frame, factor_frame)


def test_dl_factor_train_mode_is_rejected(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    frame = portal.load_daily(symbols=["AAA", "BBB"])
    factor_frame = pd.DataFrame(index=frame.index)

    spec = DLSpec(
        enabled=True,
        mode="train",
        model_type="transformer",
        output_name="dl_alpha",
    )
    with pytest.raises(ValueError, match="dl.mode=train should use alpha-lab dl-train"):
        run_dl_factor(spec, frame, factor_frame)
