from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import TaskConfig, load_task_config
from alpha_lab.costs import LinearCostModel
from alpha_lab.data import DataPortal
from alpha_lab.dl import run_dl_factor, train_dl_factor
from alpha_lab.eval import Evaluator
from alpha_lab.factors import apply_operations
from alpha_lab.guards import GuardSuite
from alpha_lab.report import ReportBuilder
from alpha_lab.types import EvaluationResult, GuardIssue, PipelineResult


def _build_factor(
    frame: pd.DataFrame,
    source: str,
    operations: list[dict[str, object]],
) -> pd.Series:
    if source not in frame.columns:
        raise ValueError(f"source column not found: {source}")
    base = frame[source].astype(float)
    return apply_operations(base, operations, frame)


def _attach_net_metrics(
    result: EvaluationResult,
    cost_model: LinearCostModel,
    annualizer: float,
) -> EvaluationResult:
    costs = cost_model.estimate(result.turnover)
    costs = costs.reindex(result.long_short_returns.index).ffill().fillna(0.0)
    net_ls = result.long_short_returns - costs
    result.metrics["ls_net_mean"] = float(net_ls.mean()) if not net_ls.empty else float(np.nan)
    result.metrics["ls_net_sharpe"] = (
        float(net_ls.mean() / (net_ls.std(ddof=0) + 1e-12) * annualizer)
        if (not net_ls.empty and np.isfinite(annualizer))
        else float(np.nan)
    )
    return result


def _load_frame(config: TaskConfig) -> pd.DataFrame:
    portal = DataPortal(config.data_dir)
    frame = portal.load(
        frequency=config.frequency,
        symbols=config.symbols or None,
        start=config.start,
        end=config.end,
    )
    return frame


def _build_factor_values(
    config: TaskConfig,
    frame: pd.DataFrame,
    guard_suite: GuardSuite,
) -> tuple[dict[str, pd.Series], list[GuardIssue]]:
    factor_values: dict[str, pd.Series] = {}
    issues: list[GuardIssue] = []
    for spec in config.factors:
        operator_issue = guard_suite.check_operator_rules(spec.operations)
        if operator_issue is not None:
            issues.append(operator_issue)
            continue
        factor = _build_factor(frame, spec.source, spec.operations)
        if spec.fillna:
            factor = apply_operations(factor, [{"op": "fillna", "method": spec.fillna}], frame)
        factor = guard_suite.apply_tradable_filter(factor, frame)

        leakage_issue = guard_suite.check_future_leakage(factor, frame)
        if leakage_issue is not None:
            issues.append(leakage_issue)

        factor_values[spec.name] = factor
    return factor_values, issues


def run_pipeline(config_or_path: TaskConfig | str | Path) -> PipelineResult:
    config = (
        load_task_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )

    frame = _load_frame(config)

    guard_suite = GuardSuite(config.guards)
    frame = guard_suite.apply_price_adjustment(frame)
    evaluator = Evaluator(config.eval)
    annualizer = evaluator.annualizer
    cost_model = LinearCostModel(
        commission_bps=config.costs.commission_bps,
        slippage_bps=config.costs.slippage_bps,
    )

    factor_values, issues = _build_factor_values(config, frame, guard_suite)
    evaluations: dict[str, EvaluationResult] = {}

    for factor_name, factor in factor_values.items():
        evaluation = evaluator.evaluate(frame, factor)
        evaluation = _attach_net_metrics(evaluation, cost_model, annualizer)
        evaluations[factor_name] = evaluation

    factor_frame = pd.DataFrame(factor_values).sort_index() if factor_values else pd.DataFrame()
    if config.dl.enabled and config.dl.mode != "skip":
        dl_factor = run_dl_factor(config.dl, frame, factor_frame)
        dl_factor = guard_suite.apply_tradable_filter(dl_factor, frame)

        leakage_issue = guard_suite.check_future_leakage(dl_factor, frame)
        if leakage_issue is not None:
            issues.append(leakage_issue)

        factor_values[config.dl.output_name] = dl_factor
        factor_frame = pd.DataFrame(factor_values).sort_index()

        dl_evaluation = evaluator.evaluate(frame, dl_factor)
        dl_evaluation = _attach_net_metrics(dl_evaluation, cost_model, annualizer)
        evaluations[config.dl.output_name] = dl_evaluation

    guard_report = guard_suite.build_report(issues)
    if guard_report.has_blocker:
        raise ValueError("Guard checks found blocker issues, please fix config/operators first.")

    report_builder = ReportBuilder(config.output_dir)
    report_path = report_builder.build(config.task_name, evaluations, guard_report)

    return PipelineResult(
        task_name=config.task_name,
        factor_values=factor_frame,
        evaluation=evaluations,
        guard_report=guard_report,
        report_path=report_path,
    )


def train_dl_pipeline(config_or_path: TaskConfig | str | Path):
    config = (
        load_task_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    if not config.dl.enabled:
        raise ValueError("dl.enabled must be true for dl-train")
    if config.dl.mode != "train":
        raise ValueError("dl-train requires dl.mode=train")

    frame = _load_frame(config)
    guard_suite = GuardSuite(config.guards)
    frame = guard_suite.apply_price_adjustment(frame)

    factor_values, issues = _build_factor_values(config, frame, guard_suite)
    guard_report = guard_suite.build_report(issues)
    if guard_report.has_blocker:
        raise ValueError("Guard checks found blocker issues, please fix config/operators first.")
    factor_frame = pd.DataFrame(factor_values).sort_index() if factor_values else pd.DataFrame()
    artifact_dir = config.output_dir / config.task_name / "dl"
    return train_dl_factor(config.dl, frame, factor_frame, artifact_dir=artifact_dir)
