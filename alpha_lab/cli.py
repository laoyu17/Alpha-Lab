from __future__ import annotations

import argparse
from pathlib import Path

from alpha_lab.data import generate_demo_data
from alpha_lab.pipeline import run_pipeline, train_dl_pipeline


def _cmd_generate_demo_data(args: argparse.Namespace) -> int:
    output = Path(args.output)
    generate_demo_data(output, seed=args.seed)
    print(f"[alpha-lab] demo data generated at: {output}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_pipeline(args.config)
    print(f"[alpha-lab] task completed: {result.task_name}")
    print(f"[alpha-lab] report: {result.report_path}")
    if result.guard_report.issues:
        print("[alpha-lab] guard issues:")
        for issue in result.guard_report.issues:
            print(f"  - [{issue.level}] {issue.name}: {issue.message}")
    return 0


def _cmd_dl_train(args: argparse.Namespace) -> int:
    try:
        result = train_dl_pipeline(args.config)
    except Exception as exc:
        print(f"[alpha-lab] dl-train failed: {exc}")
        return 1
    print("[alpha-lab] DL training completed.")
    print(f"[alpha-lab] checkpoint: {result.checkpoint_path}")
    print(f"[alpha-lab] metadata: {result.metadata_path}")
    print(f"[alpha-lab] train_size={result.train_size} val_size={result.val_size}")
    return 0


def _cmd_dl_infer(args: argparse.Namespace) -> int:
    try:
        result = run_pipeline(args.config)
    except Exception as exc:
        print(f"[alpha-lab] dl-infer failed: {exc}")
        return 1
    print(f"[alpha-lab] DL inference completed: {result.task_name}")
    print(f"[alpha-lab] report: {result.report_path}")
    return 0


def _cmd_gui(_: argparse.Namespace) -> int:
    from alpha_lab.gui import launch_gui

    return launch_gui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha-lab", description="Alpha factor research toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-demo-data", help="Generate reproducible demo parquet data")
    generate.add_argument("--output", default="data/demo", help="Output folder for demo data")
    generate.add_argument("--seed", type=int, default=7, help="Random seed")
    generate.set_defaults(func=_cmd_generate_demo_data)

    run = sub.add_parser("run", help="Run factor research pipeline from a YAML config")
    run.add_argument("--config", required=True, help="Path to task yaml config")
    run.set_defaults(func=_cmd_run)

    dl_train = sub.add_parser("dl-train", help="Train DL model defined in YAML config")
    dl_train.add_argument("--config", required=True, help="Path to task yaml config")
    dl_train.set_defaults(func=_cmd_dl_train)

    dl_infer = sub.add_parser(
        "dl-infer",
        help="Run pipeline with DL inference config and generate report",
    )
    dl_infer.add_argument("--config", required=True, help="Path to task yaml config")
    dl_infer.set_defaults(func=_cmd_dl_infer)

    gui = sub.add_parser("gui", help="Launch PyQt6 visual dashboard")
    gui.set_defaults(func=_cmd_gui)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
