from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.data import generate_demo_data


@pytest.fixture()
def demo_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "demo_data"
    generate_demo_data(data_dir, seed=11)
    return data_dir
