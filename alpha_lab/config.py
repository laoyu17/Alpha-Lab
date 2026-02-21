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
class DLSpec:
    enabled: bool = False
    plugin: str = "temporal_mlp_stub"
    output_name: str = "dl_alpha"
    feature_factors: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


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
    dl_spec = DLSpec(**raw.get("dl", {}))

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
