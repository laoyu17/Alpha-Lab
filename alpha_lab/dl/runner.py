from __future__ import annotations

import pandas as pd

from alpha_lab.config import DLSpec
from alpha_lab.dl.base import DLContext
from alpha_lab.dl.registry import get_dl_plugin


def run_dl_plugin(spec: DLSpec, frame: pd.DataFrame, factor_frame: pd.DataFrame) -> pd.Series:
    plugin = get_dl_plugin(spec.plugin)
    context = DLContext(
        frame=frame,
        factor_frame=factor_frame,
        feature_factors=spec.feature_factors,
        params=spec.params,
        output_name=spec.output_name,
    )
    series = plugin.generate(context)
    if not isinstance(series.index, pd.MultiIndex):
        raise ValueError("DL plugin must return MultiIndex(datetime, symbol) series.")
    return series.reindex(frame.index)
