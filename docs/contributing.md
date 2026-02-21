# Contributing

## Branch & PR

- Base branch: `main`
- Feature branch: `feature/<topic>`
- Fix branch: `fix/<topic>`
- Merge strategy: Pull Request + squash or rebase

## Commit Convention

Use Conventional Commits:

```
feat(scope): short summary
fix(scope): short summary
refactor(scope): short summary
docs(scope): short summary
test(scope): short summary
chore(scope): short summary
```

## Local Checklist

Before opening PR, run:

```bash
ruff check .
mypy alpha_lab
pytest
```

If pipeline/report logic changed, run:

```bash
alpha-lab generate-demo-data --output data/demo
alpha-lab run --config configs/example.yaml
```

If DL training/inference logic changed, additionally run:

```bash
pip install -e .[dl]
alpha-lab dl-train --config configs/example_dl_train_tcn.yaml
alpha-lab dl-infer --config configs/example_dl_infer_tcn.yaml
# one-command full DL regression (TCN + Transformer + artifact checks)
./scripts/regression_dl.sh
```

## Pull Request Template

Please follow `.github/pull_request_template.md` and include:

- What changed
- Why it changed
- How it was tested
- Risks and rollback plan
