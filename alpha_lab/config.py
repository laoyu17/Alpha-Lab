from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class FactorSpec:
    name: str
    source: str
    operations: list[dict[str, Any]] = field(default_factory=list)
    fillna: str | None = "ffill"


@dataclass(slots=True)
class EvalSpec:
    forward_period: int = 1
    quantiles: int = 5
    rolling_window: int = 120
    walk_forward_train: int = 252
    walk_forward_test: int = 63
    trading_days_per_year: int = 252
    periods_per_day: int = 1


@dataclass(slots=True)
class GuardSpec:
    enable_future_check: bool = True
    enforce_tradable_filter: bool = True
    limit_move_threshold: float = 0.095


@dataclass(slots=True)
class CostSpec:
    commission_bps: float = 2.0
    slippage_bps: float = 5.0


@dataclass(slots=True)
class DLTrainSpec:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    val_split: float = 0.2
    seed: int = 7


@dataclass(slots=True)
class DLSpec:
    enabled: bool = False
    plugin: str = "temporal_mlp_stub"
    output_name: str = "dl_alpha"
    feature_factors: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    mode: str = "infer"
    model_type: str | None = None
    checkpoint_path: str | None = None
    features: list[str] = field(default_factory=list)
    train: DLTrainSpec = field(default_factory=DLTrainSpec)


@dataclass(slots=True)
class TaskConfig:
    task_name: str
    data_dir: Path
    output_dir: Path
    frequency: str = "daily"
    symbols: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    factors: list[FactorSpec] = field(default_factory=list)
    eval: EvalSpec = field(default_factory=EvalSpec)
    guards: GuardSpec = field(default_factory=GuardSpec)
    costs: CostSpec = field(default_factory=CostSpec)
    dl: DLSpec = field(default_factory=DLSpec)


def _parse_factor(item: dict[str, Any]) -> FactorSpec:
    if "name" not in item:
        raise ValueError("factor config requires name")
    source = item.get("source", "close")
    operations = item.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("factor.operations must be a list")
    return FactorSpec(
        name=str(item["name"]),
        source=str(source),
        operations=operations,
        fillna=item.get("fillna", "ffill"),
    )


def _parse_dl_spec(raw_dl: Any) -> DLSpec:
    if raw_dl is None:
        return DLSpec()
    if not isinstance(raw_dl, dict):
        raise ValueError("dl config must be a mapping")

    train_raw = raw_dl.get("train", {})
    if train_raw is None:
        train_raw = {}
    if not isinstance(train_raw, dict):
        raise ValueError("dl.train must be a mapping")
    train_spec = DLTrainSpec(**train_raw)

    mode = str(raw_dl.get("mode", "infer")).lower()
    if mode not in {"train", "infer", "skip"}:
        raise ValueError("dl.mode must be one of: train, infer, skip")

    model_type_raw = raw_dl.get("model_type")
    model_type: str | None = None
    if model_type_raw is not None:
        model_type = str(model_type_raw).lower()
        if model_type not in {"tcn", "transformer"}:
            raise ValueError("dl.model_type must be one of: tcn, transformer")

    features_raw = raw_dl.get("features", [])
    if features_raw is None:
        features_raw = []
    if not isinstance(features_raw, list):
        raise ValueError("dl.features must be a list")

    feature_factors_raw = raw_dl.get("feature_factors", [])
    if feature_factors_raw is None:
        feature_factors_raw = []
    if not isinstance(feature_factors_raw, list):
        raise ValueError("dl.feature_factors must be a list")

    params_raw = raw_dl.get("params", {})
    if params_raw is None:
        params_raw = {}
    if not isinstance(params_raw, dict):
        raise ValueError("dl.params must be a mapping")

    checkpoint_raw = raw_dl.get("checkpoint_path")
    checkpoint_path = None if checkpoint_raw is None else str(checkpoint_raw)

    return DLSpec(
        enabled=bool(raw_dl.get("enabled", False)),
        plugin=str(raw_dl.get("plugin", "temporal_mlp_stub")),
        output_name=str(raw_dl.get("output_name", "dl_alpha")),
        feature_factors=[str(x) for x in feature_factors_raw],
        params=params_raw,
        mode=mode,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        features=[str(x) for x in features_raw],
        train=train_spec,
    )


def load_task_config(path: str | Path) -> TaskConfig:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    factors = [_parse_factor(x) for x in raw.get("factors", [])]
    eval_spec = EvalSpec(**raw.get("eval", {}))
    guard_spec = GuardSpec(**raw.get("guards", {}))
    cost_spec = CostSpec(**raw.get("costs", {}))
    dl_spec = _parse_dl_spec(raw.get("dl", {}))

    data_dir = Path(raw.get("data_dir", "data/demo"))
    output_dir = Path(raw.get("output_dir", "outputs"))

    return TaskConfig(
        task_name=str(raw.get("task_name", path_obj.stem)),
        data_dir=data_dir,
        output_dir=output_dir,
        frequency=str(raw.get("frequency", "daily")),
        symbols=[str(s) for s in raw.get("symbols", [])],
        start=raw.get("start"),
        end=raw.get("end"),
        factors=factors,
        eval=eval_spec,
        guards=guard_spec,
        costs=cost_spec,
        dl=dl_spec,
    )
