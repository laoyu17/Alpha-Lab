from __future__ import annotations

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
