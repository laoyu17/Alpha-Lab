# Alpha-Lab 需求文档

## 1. 背景与目标

Alpha-Lab 旨在把“因子挖掘”做成标准工程流水线，服务于量化研究实习岗位投递，重点匹配：

- 因子构建（多源数据、算子组合、因子验证）
- 研究工程化（可复现、可扩展、可回归）
- 量化常见风险规避（未来函数、可交易性、成本）

## 2. 核心用户

- 求职候选人（软件/非金融背景）
- 量化研究面试官（看重研究流程完整性与工程质量）

## 3. 功能需求

### 3.1 数据层

- 支持 `daily`、`minute` 两种频率
- 统一 Parquet / Arrow IPC(含 Feather) 存储接口
- 支持按标的和时间过滤

### 3.2 因子层

- 提供基础算子：`rolling_mean`、`rolling_std`、`rank`、`zscore`、`decay_linear`、`winsorize`、`neutralize`
- 支持 YAML 配置化算子链
- 支持缺失值处理（`ffill`/`bfill`/`zero`）

### 3.3 评估层

- IC / RankIC
- 分层收益（Quantile）
- 多空收益与换手
- 稳定性（按年度分段）
- Walk-forward 样本外指标
- 滚动窗口 IC 诊断指标（由 `eval.rolling_window` 控制）
- 成本后净收益指标

### 3.4 防坑与风控工程

- 未来函数检测（算子规则 + IC 特征检测）
- 可交易性过滤（停牌、涨跌停、异常波动）
- 复权价格处理（`adj_factor`）

### 3.5 报告与展示

- 自动生成 HTML 报告与图表
- 输出 summary / metrics / stability CSV
- 提供 PyQt6 GUI：任务配置、运行日志、报告查看（支持因子切换）

### 3.6 Phase-2 深度学习扩展接口

- 提供配置化 `dl` 区块（启用开关、模式、模型类型、特征、参数、输出因子名）
- 支持 `train/infer/skip` 模式，训练与推理链路解耦
- 支持真实 PyTorch `TCN` 与 `Transformer` 两类序列模型
- 兼容历史轻量插件入口，输出统一为 `MultiIndex(datetime, symbol)` 序列

## 4. 非功能需求

- 可复现：提供 demo 数据生成命令
- 可维护：模块化目录、单测覆盖核心流程
- 可验证：CI 至少覆盖 lint/type/test

## 5. 不在本期范围

- 实盘交易接入
- 全量 Level2 真实行情接入
