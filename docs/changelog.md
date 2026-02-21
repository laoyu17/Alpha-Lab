# Changelog

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
