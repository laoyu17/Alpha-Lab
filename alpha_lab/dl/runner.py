from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.config import DLSpec
from alpha_lab.dl.base import DLContext
from alpha_lab.dl.registry import get_dl_plugin
from alpha_lab.dl.training import DLTrainResult, infer_model, train_model


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


def run_dl_factor(spec: DLSpec, frame: pd.DataFrame, factor_frame: pd.DataFrame) -> pd.Series:
    if spec.mode == "train":
        raise ValueError("dl.mode=train should use alpha-lab dl-train first.")
    if spec.model_type in {"tcn", "transformer"}:
        return infer_model(spec, frame, factor_frame)
    return run_dl_plugin(spec, frame, factor_frame)


def train_dl_factor(
    spec: DLSpec,
    frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
    artifact_dir: str | Path,
) -> DLTrainResult:
    if spec.mode != "train":
        raise ValueError("dl-train requires dl.mode=train")
    if spec.model_type not in {"tcn", "transformer"}:
        raise ValueError("dl-train requires dl.model_type to be tcn or transformer")
    return train_model(spec, frame, factor_frame, artifact_dir=artifact_dir)
