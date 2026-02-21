# AGENTS (Alpha-Lab)

> 版本: 0.1  
> 更新日期: 2026-02-21

## 1. 仓库信息

- GitHub 仓库地址：`https://github.com/laoyu17/Alpha-Lab.git`
- 项目定位：量化因子研究工程化工具链（面向实习/求职展示）
- 技术栈：Python 3.11、Pandas/Numpy、PyQt6、Pytest、Ruff、Mypy

## 2. 开发流程（必须遵守）

1. 从 `main` 拉取最新代码，创建 `feature/<topic>` 分支开发。
2. 每个功能使用小步提交，提交信息遵循 Conventional Commits。
3. 提交前必须通过本地回归（见第 4 节）。
4. 通过 PR 合并到 `main`，禁止直接向 `main` 推送功能代码。
5. 合并后更新 `docs/changelog.md`。

## 3. 提交规范

提交格式：

`<type>(<scope>): <subject>`

允许类型：

- `feat`: 新功能
- `fix`: 缺陷修复
- `refactor`: 重构（无行为变化）
- `test`: 测试
- `docs`: 文档
- `chore`: 构建/脚手架/依赖

示例：

- `feat(pipeline): add guard checks for future leakage`
- `fix(report): handle empty quantile returns`
- `docs(readme): add quick start for demo dataset`

## 4. 每次开发测试回归要求

每次提交前至少执行：

```bash
ruff check .
mypy alpha_lab
pytest
```

涉及核心流程变更（data/factor/eval/guards/report/gui）时，还需执行：

```bash
alpha-lab generate-demo-data --output data/demo
alpha-lab run --config configs/example.yaml
```

并确认：

- `outputs/<task_name>/report.html` 可打开；
- `summary.csv`、`metrics.csv` 已生成且指标非空。

## 5. 文档更新要求

出现以下情况必须同步更新文档：

- **需求/范围变化**：更新 `docs/requirements.md`
- **接口/模块变化**：更新 `docs/design.md`
- **计划调整**：更新 `docs/dev-plan.md`
- **测试策略变化**：更新 `docs/testing-regression.md`
- **协作流程变化**：更新 `docs/contributing.md`

每次版本合并必须追加 `docs/changelog.md`。

## 6. 代码与设计约束

- 遵循 KISS/YAGNI，优先复用现有模块。
- 未经明确批准，不破坏现有 CLI、配置字段、输出目录结构。
- Guard 相关逻辑（未来函数、可交易性）是项目卖点，不得跳过。
- GUI 仅做配置/运行/查看，不承载核心计算逻辑。

## 7. 安全与操作边界

- 禁止提交密钥、token、私有数据。
- 禁止使用破坏性命令重写历史（如 `git reset --hard`）处理他人改动。
- 数据样例需可开源复现，默认使用公开/仿真数据。
