from __future__ import annotations

import copy
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha_lab.pipeline import run_pipeline


def _single_factor_config(
    demo_data_dir: Path,
    output_dir: Path,
    frequency: str,
) -> dict[str, object]:
    return {
        "task_name": f"unit_test_task_{frequency}",
        "data_dir": str(demo_data_dir),
        "output_dir": str(output_dir),
        "frequency": frequency,
        "symbols": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "factors": [
            {
                "name": "price_factor",
                "source": "close",
                "operations": [
                    {"op": "rolling_mean", "window": 10},
                    {"op": "zscore"},
                    {"op": "winsorize", "lower": 0.05, "upper": 0.95},
                    {"op": "neutralize"},
                ],
                "fillna": "ffill",
            }
        ],
        "eval": {
            "forward_period": 1,
            "quantiles": 5,
            "walk_forward_train": 80,
            "walk_forward_test": 20,
        },
        "guards": {"enable_future_check": True, "enforce_tradable_filter": True},
        "costs": {"commission_bps": 2.0, "slippage_bps": 5.0},
    }


def test_pipeline_end_to_end(demo_data_dir: Path, tmp_path: Path) -> None:
    config = _single_factor_config(demo_data_dir, tmp_path / "outputs", frequency="daily")
    config["dl"] = {
        "enabled": True,
        "plugin": "temporal_mlp_stub",
        "output_name": "dl_alpha",
        "feature_factors": ["price_factor"],
        "params": {"lookback": 8, "momentum_weight": 0.6, "fusion_weight": 0.4},
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_pipeline(config_path)
    assert "price_factor" in result.factor_values.columns
    assert "dl_alpha" in result.factor_values.columns
    assert result.report_path is not None
    assert result.report_path.exists()
    gross_mean = float(result.evaluation["price_factor"].long_short_returns.mean())
    assert float(result.evaluation["price_factor"].metrics["ls_mean"]) == pytest.approx(gross_mean)
    assert float(result.evaluation["price_factor"].metrics["ls_net_mean"]) <= gross_mean + 1e-12

    summary_path = result.report_path.parent / "summary.csv"
    summary = pd.read_csv(summary_path)
    assert "ls_net_sharpe" in summary.columns
    assert "ls_net_mean" in summary.columns


def test_pipeline_daily_defaults_keep_backward_compatible_metrics(
    demo_data_dir: Path,
    tmp_path: Path,
) -> None:
    legacy_config = _single_factor_config(
        demo_data_dir,
        tmp_path / "legacy_outputs",
        frequency="daily",
    )
    legacy_path = tmp_path / "legacy_daily.yaml"
    legacy_path.write_text(yaml.safe_dump(legacy_config), encoding="utf-8")
    legacy_result = run_pipeline(legacy_path)

    explicit_config = copy.deepcopy(legacy_config)
    explicit_config["task_name"] = "unit_test_task_daily_explicit"
    explicit_config["output_dir"] = str(tmp_path / "explicit_outputs")
    explicit_eval = explicit_config["eval"]
    assert isinstance(explicit_eval, dict)
    explicit_eval["trading_days_per_year"] = 252
    explicit_eval["periods_per_day"] = 1
    explicit_path = tmp_path / "explicit_daily.yaml"
    explicit_path.write_text(yaml.safe_dump(explicit_config), encoding="utf-8")
    explicit_result = run_pipeline(explicit_path)

    legacy_metrics = legacy_result.evaluation["price_factor"].metrics
    explicit_metrics = explicit_result.evaluation["price_factor"].metrics
    assert float(legacy_metrics["ls_sharpe"]) == pytest.approx(float(explicit_metrics["ls_sharpe"]))
    assert float(legacy_metrics["ls_net_sharpe"]) == pytest.approx(
        float(explicit_metrics["ls_net_sharpe"])
    )


def test_pipeline_minute_sharpe_scales_with_periods_per_day(
    demo_data_dir: Path,
    tmp_path: Path,
) -> None:
    minute_base = _single_factor_config(
        demo_data_dir,
        tmp_path / "minute_p1_outputs",
        frequency="minute",
    )
    minute_eval = minute_base["eval"]
    assert isinstance(minute_eval, dict)
    minute_eval["trading_days_per_year"] = 252
    minute_eval["periods_per_day"] = 1
    minute_base_path = tmp_path / "minute_p1.yaml"
    minute_base_path.write_text(yaml.safe_dump(minute_base), encoding="utf-8")
    result_p1 = run_pipeline(minute_base_path)

    minute_p8 = copy.deepcopy(minute_base)
    minute_p8["task_name"] = "unit_test_task_minute_p8"
    minute_p8["output_dir"] = str(tmp_path / "minute_p8_outputs")
    minute_p8_eval = minute_p8["eval"]
    assert isinstance(minute_p8_eval, dict)
    minute_p8_eval["periods_per_day"] = 8
    minute_p8_path = tmp_path / "minute_p8.yaml"
    minute_p8_path.write_text(yaml.safe_dump(minute_p8), encoding="utf-8")
    result_p8 = run_pipeline(minute_p8_path)

    base_eval = result_p1.evaluation["price_factor"]
    scaled_eval = result_p8.evaluation["price_factor"]
    pd.testing.assert_series_equal(base_eval.long_short_returns, scaled_eval.long_short_returns)

    assert float(scaled_eval.metrics["ls_sharpe"]) == pytest.approx(
        float(base_eval.metrics["ls_sharpe"]) * math.sqrt(8),
        rel=1e-9,
        abs=1e-9,
    )
    assert float(scaled_eval.metrics["ls_net_sharpe"]) == pytest.approx(
        float(base_eval.metrics["ls_net_sharpe"]) * math.sqrt(8),
        rel=1e-9,
        abs=1e-9,
    )


def test_pipeline_blocks_forbidden_operator(demo_data_dir: Path, tmp_path: Path) -> None:
    config = {
        "task_name": "guard_blocker_task",
        "data_dir": str(demo_data_dir),
        "output_dir": str(tmp_path / "outputs"),
        "frequency": "daily",
        "symbols": ["AAA", "BBB", "CCC"],
        "factors": [
            {
                "name": "bad_factor",
                "source": "close",
                "operations": [{"op": "future_mean", "window": 3}],
                "fillna": "ffill",
            }
        ],
        "guards": {"enable_future_check": True, "enforce_tradable_filter": True},
    }
    config_path = tmp_path / "task_blocker.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Guard checks found blocker issues"):
        run_pipeline(config_path)


def test_pipeline_rejects_dl_train_mode(demo_data_dir: Path, tmp_path: Path) -> None:
    config = {
        "task_name": "dl_train_mode_task",
        "data_dir": str(demo_data_dir),
        "output_dir": str(tmp_path / "outputs"),
        "frequency": "daily",
        "symbols": ["AAA", "BBB", "CCC"],
        "factors": [
            {
                "name": "price_factor",
                "source": "close",
                "operations": [{"op": "rolling_mean", "window": 5}],
                "fillna": "ffill",
            }
        ],
        "dl": {
            "enabled": True,
            "mode": "train",
            "model_type": "tcn",
            "feature_factors": ["price_factor"],
        },
    }
    config_path = tmp_path / "task_dl_train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="dl.mode=train should use alpha-lab dl-train"):
        run_pipeline(config_path)
