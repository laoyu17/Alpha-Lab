from __future__ import annotations

from pathlib import Path

import yaml

from alpha_lab.pipeline import run_pipeline


def test_pipeline_end_to_end(demo_data_dir: Path, tmp_path: Path) -> None:
    config = {
        "task_name": "unit_test_task",
        "data_dir": str(demo_data_dir),
        "output_dir": str(tmp_path / "outputs"),
        "frequency": "daily",
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
        "dl": {
            "enabled": True,
            "plugin": "temporal_mlp_stub",
            "output_name": "dl_alpha",
            "feature_factors": ["price_factor"],
            "params": {"lookback": 8, "momentum_weight": 0.6, "fusion_weight": 0.4},
        },
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_pipeline(config_path)
    assert "price_factor" in result.factor_values.columns
    assert "dl_alpha" in result.factor_values.columns
    assert result.report_path is not None
    assert result.report_path.exists()
