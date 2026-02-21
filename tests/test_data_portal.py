from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.data import DataPortal


def test_data_portal_load_daily(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    daily = portal.load_daily(symbols=["AAA", "BBB"], start="2023-03-01", end="2023-06-30")
    assert not daily.empty
    assert daily.index.nlevels == 2
    assert set(daily.index.get_level_values(1).unique().tolist()) <= {"AAA", "BBB"}
    assert {"open", "close", "volume"}.issubset(daily.columns)


def test_data_portal_load_minute(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    minute = portal.load_minute(symbols=["AAA"])
    assert not minute.empty
    assert minute.index.nlevels == 2
    assert minute.index.get_level_values(1).unique().tolist() == ["AAA"]


def test_data_portal_minute_end_date_includes_whole_day(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    minute = portal.load_minute(symbols=["AAA"])
    target_day = minute.index.get_level_values(0).max().date().isoformat()

    filtered = portal.load_minute(symbols=["AAA"], start=target_day, end=target_day)
    assert not filtered.empty
    assert filtered.index.get_level_values(0).date.min().isoformat() == target_day
    assert filtered.index.get_level_values(0).date.max().isoformat() == target_day


def test_data_portal_invalid_frequency_raises(demo_data_dir) -> None:
    portal = DataPortal(demo_data_dir)
    with pytest.raises(ValueError, match="unsupported frequency"):
        portal.load("daliy")


def test_data_portal_load_feather_when_parquet_missing(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-03"],
            "symbol": ["AAA", "AAA"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.6],
            "low": [9.8, 10.1],
            "close": [10.3, 10.4],
            "adj_factor": [1.0, 1.0],
        }
    )
    data_dir = tmp_path / "arrow_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_feather(data_dir / "daily.feather")

    portal = DataPortal(data_dir)
    loaded = portal.load_daily()
    assert not loaded.empty
    assert loaded.index.nlevels == 2
    assert loaded.index.get_level_values(1).unique().tolist() == ["AAA"]
