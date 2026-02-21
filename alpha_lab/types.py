from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class GuardIssue:
    name: str
    level: str
    message: str


@dataclass(slots=True)
class GuardReport:
    issues: list[GuardIssue] = field(default_factory=list)

    @property
    def has_blocker(self) -> bool:
        return any(issue.level.lower() == "error" for issue in self.issues)

    def as_rows(self) -> list[dict[str, str]]:
        return [{"name": i.name, "level": i.level, "message": i.message} for i in self.issues]


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    ic_series: pd.Series
    rank_ic_series: pd.Series
    quantile_returns: pd.DataFrame
    long_short_returns: pd.Series
    turnover: pd.Series
    stability: pd.DataFrame
    attribution: dict[str, float]


@dataclass(slots=True)
class PipelineResult:
    task_name: str
    factor_values: pd.DataFrame
    evaluation: dict[str, EvaluationResult]
    guard_report: GuardReport
    report_path: Path | None = None
