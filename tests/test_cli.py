from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import alpha_lab.cli as cli
from alpha_lab.cli import build_parser


def test_cli_parser_includes_dl_subcommands() -> None:
    parser = build_parser()
    train_args = parser.parse_args(["dl-train", "--config", "task.yaml"])
    infer_args = parser.parse_args(["dl-infer", "--config", "task.yaml"])

    assert train_args.command == "dl-train"
    assert infer_args.command == "dl-infer"


def test_cmd_run_success_returns_zero(monkeypatch, capsys, tmp_path: Path) -> None:
    fake_result = SimpleNamespace(
        task_name="demo",
        report_path=tmp_path / "report.html",
        guard_report=SimpleNamespace(issues=[]),
    )
    monkeypatch.setattr(cli, "run_pipeline", lambda _: fake_result)

    code = cli._cmd_run(argparse.Namespace(config="configs/example.yaml"))

    assert code == 0
    assert "[alpha-lab] task completed: demo" in capsys.readouterr().out


def test_cmd_run_failure_returns_one(monkeypatch, capsys) -> None:
    def _raise(_: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_pipeline", _raise)

    code = cli._cmd_run(argparse.Namespace(config="configs/example.yaml"))

    assert code == 1
    assert "[alpha-lab] run failed: boom" in capsys.readouterr().out


def test_cmd_dl_train_and_dl_infer_failure_return_one(monkeypatch, capsys) -> None:
    def _raise(_: str) -> None:
        raise RuntimeError("failed")

    monkeypatch.setattr(cli, "train_dl_pipeline", _raise)
    train_code = cli._cmd_dl_train(argparse.Namespace(config="configs/example.yaml"))
    assert train_code == 1
    assert "[alpha-lab] dl-train failed: failed" in capsys.readouterr().out

    monkeypatch.setattr(cli, "run_pipeline", _raise)
    infer_code = cli._cmd_dl_infer(argparse.Namespace(config="configs/example.yaml"))
    assert infer_code == 1
    assert "[alpha-lab] dl-infer failed: failed" in capsys.readouterr().out
