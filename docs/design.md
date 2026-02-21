# Alpha-Lab 设计文档

## 1. 架构总览

```
DataPortal -> Factor Pipeline -> GuardSuite -> Evaluator -> CostModel -> ReportBuilder
                                   |                                 |
                                   +------------- GUI/CLI -----------+
                         DL Plugin Interface ---------^
```

## 2. 模块说明

### 2.1 `alpha_lab/data`

- `DataPortal`：统一加载 `daily|minute` 的 Parquet / Arrow IPC(含 Feather) 数据文件
- 频率字段仅允许 `daily` / `minute`，非法值直接报错，避免静默读错数据
- `generate_demo_data`：生成可复现的公开演示数据

### 2.2 `alpha_lab/factors`

- `OPERATOR_REGISTRY`：算子注册表
- `apply_operations`：按 YAML 顺序执行算子链
- 算子均基于 `MultiIndex(datetime, symbol)` 输入

### 2.3 `alpha_lab/guards`

- `apply_price_adjustment`：基于 `adj_factor` 对 OHLC 统一复权
- `apply_tradable_filter`：停牌/涨跌停/波动阈值过滤
- `check_future_leakage`：同日 IC 异常强度告警
- `check_operator_rules`：禁止前视算子

### 2.4 `alpha_lab/eval`

- forward return 计算
- IC/RankIC
- quantile 分层收益、多空收益
- top bucket turnover
- 年度稳定性统计
- walk-forward 样本外 IC 统计
- 滚动窗口 IC 诊断（`rolling_window`）
- 因子收益对市场收益回归归因（alpha/beta/r2）

### 2.5 `alpha_lab/costs`

- `LinearCostModel`：线性交易成本估计（commission + slippage）

### 2.6 `alpha_lab/report`

- 生成指标图（IC、RankIC、多空收益、换手、分层累计）
- 输出 HTML 报告与 CSV 工件

### 2.7 `alpha_lab/gui`

- PyQt6 轻量三页签：
  - Task Config
  - Task Runner
  - Report Viewer（支持多因子指标切换查看）

### 2.8 `alpha_lab/dl`

- `DLSpec`：配置化深度学习插件参数
- `DLContext`：统一插件输入上下文（行情 + 因子 + 参数）
- `run_dl_plugin`：兼容历史轻量插件入口（stub / 自定义路径）
- `run_dl_factor`：统一 DL 推理入口（兼容 plugin + tcn/transformer）
- `train_dl_factor`：真实 DL 训练入口（TCN/Transformer）
- `dataset`：统一序列样本构建（lookback/horizon/feature 对齐）
- `models`：PyTorch TCN 与 Transformer 回归模型
- `registry`：内置/外部插件发现
- 内置 `temporal_mlp_stub` 继续保留，便于无重依赖快速演示

## 3. 对外接口

### 3.1 CLI

- `alpha-lab generate-demo-data --output data/demo`
- `alpha-lab run --config configs/example.yaml`
- `alpha-lab dl-train --config configs/example_dl_train_tcn.yaml`
- `alpha-lab dl-infer --config configs/example_dl_infer_tcn.yaml`
- `alpha-lab gui`

### 3.2 配置文件（YAML）

核心字段：

- `task_name`, `data_dir`, `output_dir`, `frequency`
- `symbols`, `start`, `end`
- `factors[]`: `name`, `source`, `operations[]`, `fillna`
- `eval`, `guards`, `costs`
- `dl`: `enabled`, `mode`, `model_type`, `checkpoint_path`, `features`, `train`, `params`
- 兼容字段：`plugin`, `feature_factors`（历史配置仍可运行）

## 4. 输出约定

- `outputs/<task_name>/report.html`
- `outputs/<task_name>/summary.csv`
- `outputs/<task_name>/<factor>/metrics.csv`
- `outputs/<task_name>/<factor>/stability.csv`
- `outputs/<task_name>/<factor>/*.png`

## 5. 兼容性策略

- 配置字段只增不删；废弃字段需保留兼容读取
- CLI 命令和输出结构保持稳定
