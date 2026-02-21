# Changelog

## [0.3.1] - 2026-02-21

### Fixed

- `dl-train` now saves the best-validation checkpoint weights instead of the last-epoch weights, and records `best_epoch` in metadata
- `alpha-lab run` now returns a non-zero exit code with readable error output when pipeline execution fails

### Changed

- `summary.csv` now includes cost-adjusted net metrics: `ls_net_mean` and `ls_net_sharpe`
- Expanded regression coverage for guard threshold behavior, CLI failure paths, summary net-metric schema, and DL best-checkpoint behavior

## [0.3.0] - 2026-02-21

### Added

- New DL workflow with `dl.mode=train|infer|skip` and `dl.model_type=tcn|transformer`
- New CLI commands: `alpha-lab dl-train` and `alpha-lab dl-infer`
- Sequence dataset builder for DL features/labels with configurable lookback/horizon
- PyTorch-based TCN and Transformer regressors for DL factor generation
- New example configs for TCN/Transformer train-infer pipelines

### Changed

- `fillna` now raises explicit error on unsupported methods (no silent fallback)
- Guard operator rule check now aligns with current operator registry
- Pipeline now skips DL stage when `dl.mode=skip` and rejects `dl.mode=train` in `run`

## [0.2.1] - 2026-02-21

### Changed

- DataPortal now validates `frequency` strictly and supports Parquet / Arrow IPC(含 Feather) dataset loading
- Price adjustment now applies `adj_factor` to OHLC for consistent evaluation basis
- Evaluator now materializes rolling IC diagnostics from `eval.rolling_window`
- GUI Report Viewer now supports factor switching for per-factor metric inspection
- Regression docs updated for new validation and guard scenarios

## [0.2.0] - 2026-02-21

### Added

- Phase-2 deep learning plugin interface (`alpha_lab/dl`)
- Configurable `dl` section in YAML task config
- Builtin `temporal_mlp_stub` plugin for temporal signal + factor fusion
- DL-enabled example config `configs/example_with_dl.yaml`
- DL plugin unit test and pipeline coverage extension
- Portfolio-oriented README with architecture/demo placeholders

## [0.1.0] - 2026-02-21

### Added

- Project scaffold, packaging config, and CI workflow
- Daily/minute `DataPortal` and demo data generator
- Factor operator chain with neutralization and winsorization
- GuardSuite for future leakage and tradability checks
- Evaluator for IC/RankIC/quantile/turnover/stability/walk-forward/attribution
- Linear cost model and net performance metrics
- HTML report builder with chart outputs
- PyQt6 lightweight GUI dashboard
- CLI commands for data generation, pipeline run, and GUI launch
- Tests and engineering docs
