from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alpha_lab.config import load_task_config


def test_load_config_legacy_dl_fields(tmp_path: Path) -> None:
    config = {
        "task_name": "legacy_dl",
        "data_dir": "data/demo",
        "output_dir": "outputs",
        "factors": [{"name": "f1", "source": "close", "operations": []}],
        "dl": {
            "enabled": True,
            "plugin": "temporal_mlp_stub",
            "output_name": "dl_alpha",
            "feature_factors": ["f1"],
        },
    }
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    parsed = load_task_config(path)
    assert parsed.dl.enabled is True
    assert parsed.dl.mode == "infer"
    assert parsed.dl.model_type is None
    assert parsed.dl.feature_factors == ["f1"]


def test_load_config_new_dl_fields(tmp_path: Path) -> None:
    config = {
        "task_name": "new_dl",
        "data_dir": "data/demo",
        "output_dir": "outputs",
        "factors": [{"name": "f1", "source": "close", "operations": []}],
        "dl": {
            "enabled": True,
            "mode": "train",
            "model_type": "tcn",
            "checkpoint_path": "outputs/new_dl/dl/tcn.pt",
            "features": ["f1", "close_ret_1"],
            "train": {
                "epochs": 3,
                "batch_size": 8,
                "lr": 0.001,
                "val_split": 0.25,
                "seed": 42,
            },
        },
    }
    path = tmp_path / "new.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    parsed = load_task_config(path)
    assert parsed.dl.mode == "train"
    assert parsed.dl.model_type == "tcn"
    assert parsed.dl.checkpoint_path == "outputs/new_dl/dl/tcn.pt"
    assert parsed.dl.features == ["f1", "close_ret_1"]
    assert parsed.dl.train.epochs == 3
    assert parsed.dl.train.batch_size == 8


def test_load_config_invalid_dl_mode_raises(tmp_path: Path) -> None:
    config = {
        "task_name": "bad_dl",
        "data_dir": "data/demo",
        "output_dir": "outputs",
        "factors": [{"name": "f1", "source": "close", "operations": []}],
        "dl": {"enabled": True, "mode": "unknown"},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="dl.mode must be one of"):
        load_task_config(path)
