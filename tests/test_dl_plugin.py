from __future__ import annotations

import pandas as pd

from alpha_lab.config import DLSpec
from alpha_lab.data import DataPortal
from alpha_lab.dl import run_dl_plugin


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
