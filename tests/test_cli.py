from __future__ import annotations

from alpha_lab.cli import build_parser


def test_cli_parser_includes_dl_subcommands() -> None:
    parser = build_parser()
    train_args = parser.parse_args(["dl-train", "--config", "task.yaml"])
    infer_args = parser.parse_args(["dl-infer", "--config", "task.yaml"])

    assert train_args.command == "dl-train"
    assert infer_args.command == "dl-infer"
