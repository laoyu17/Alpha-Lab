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

- `DataPortal`：统一加载 `daily.parquet` / `minute.parquet`
- `generate_demo_data`：生成可复现的公开演示数据

### 2.2 `alpha_lab/factors`

- `OPERATOR_REGISTRY`：算子注册表
- `apply_operations`：按 YAML 顺序执行算子链
- 算子均基于 `MultiIndex(datetime, symbol)` 输入

### 2.3 `alpha_lab/guards`

- `apply_price_adjustment`：复权处理
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
  - Report Viewer

### 2.8 `alpha_lab/dl`

- `DLSpec`：配置化深度学习插件参数
- `DLContext`：统一插件输入上下文（行情 + 因子 + 参数）
- `run_dl_plugin`：插件执行入口
- `registry`：内置/外部插件发现
- 当前内置 `temporal_mlp_stub`，用于演示接口与数据流，后续可替换真实模型

## 3. 对外接口

### 3.1 CLI

- `alpha-lab generate-demo-data --output data/demo`
- `alpha-lab run --config configs/example.yaml`
- `alpha-lab gui`

### 3.2 配置文件（YAML）

核心字段：

- `task_name`, `data_dir`, `output_dir`, `frequency`
- `symbols`, `start`, `end`
- `factors[]`: `name`, `source`, `operations[]`, `fillna`
- `eval`, `guards`, `costs`
- `dl`: `enabled`, `plugin`, `output_name`, `feature_factors`, `params`

## 4. 输出约定

- `outputs/<task_name>/report.html`
- `outputs/<task_name>/summary.csv`
- `outputs/<task_name>/<factor>/metrics.csv`
- `outputs/<task_name>/<factor>/stability.csv`
- `outputs/<task_name>/<factor>/*.png`

## 5. 兼容性策略

- 配置字段只增不删；废弃字段需保留兼容读取
- CLI 命令和输出结构保持稳定
