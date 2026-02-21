#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-python}"
export ROOT_DIR

run_cmd() {
  echo "[dl-regression] $*"
  "$@"
}

run_cmd "$PY_BIN" -m alpha_lab.cli generate-demo-data --output "$ROOT_DIR/data/demo"
run_cmd "$PY_BIN" -m alpha_lab.cli dl-train --config "$ROOT_DIR/configs/example_dl_train_tcn.yaml"
run_cmd "$PY_BIN" -m alpha_lab.cli dl-infer --config "$ROOT_DIR/configs/example_dl_infer_tcn.yaml"
run_cmd "$PY_BIN" -m alpha_lab.cli dl-train --config "$ROOT_DIR/configs/example_dl_train_transformer.yaml"
run_cmd "$PY_BIN" -m alpha_lab.cli dl-infer --config "$ROOT_DIR/configs/example_dl_infer_transformer.yaml"

run_cmd "$PY_BIN" - <<"PY"
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["ROOT_DIR"])
required_summary_cols = {
    "factor",
    "ic_mean",
    "rank_ic_mean",
    "ls_sharpe",
    "ls_net_mean",
    "ls_net_sharpe",
    "turnover_mean",
}

train_artifacts = [
    root / "outputs/demo_dl_train_tcn/dl/tcn_checkpoint.pt",
    root / "outputs/demo_dl_train_tcn/dl/tcn_checkpoint.json",
    root / "outputs/demo_dl_train_transformer/dl/transformer_checkpoint.pt",
    root / "outputs/demo_dl_train_transformer/dl/transformer_checkpoint.json",
]
for artifact in train_artifacts:
    if not artifact.exists():
        raise SystemExit(f"[dl-regression] missing artifact: {artifact}")

infer_tasks = ["demo_dl_infer_tcn", "demo_dl_infer_transformer"]
for task in infer_tasks:
    report = root / "outputs" / task / "report.html"
    summary = root / "outputs" / task / "summary.csv"
    if not report.exists():
        raise SystemExit(f"[dl-regression] missing report: {report}")
    if not summary.exists():
        raise SystemExit(f"[dl-regression] missing summary: {summary}")

    summary_df = pd.read_csv(summary)
    if summary_df.empty:
        raise SystemExit(f"[dl-regression] empty summary: {summary}")
    if not required_summary_cols.issubset(summary_df.columns):
        missing = required_summary_cols - set(summary_df.columns)
        raise SystemExit(f"[dl-regression] summary schema mismatch ({task}): {sorted(missing)}")

    factor_dirs = [p for p in (root / "outputs" / task).iterdir() if p.is_dir()]
    if not factor_dirs:
        raise SystemExit(f"[dl-regression] no factor directories: {task}")
    for factor_dir in factor_dirs:
        metrics_path = factor_dir / "metrics.csv"
        if not metrics_path.exists():
            raise SystemExit(f"[dl-regression] missing metrics: {metrics_path}")
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            raise SystemExit(f"[dl-regression] empty metrics: {metrics_path}")

print("[dl-regression] DL train/infer artifacts and report outputs verified.")
PY
