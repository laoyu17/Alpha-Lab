# Alpha-Lab 测试与回归规范

## 1. 测试金字塔

- 单元测试：算子、评估函数、guard 规则
- 集成测试：pipeline 端到端（demo 数据 + 配置）
- 冒烟测试：CLI 任务执行与报告产出

## 2. 必跑命令

```bash
ruff check .
mypy alpha_lab
pytest
```

## 3. 关键回归场景

1. **数据接口**：daily/minute 加载、时间和标的过滤正确
2. **参数校验**：`frequency` 非法值可被显式拦截，不允许静默回退
3. **因子算子**：rolling/rank/zscore/winsorize/neutralize 输出有效
4. **评估指标**：IC/RankIC、分层收益、多空收益、换手、滚动 IC 指标可用
5. **防坑逻辑**：前视算子拦截、可交易性过滤、OHLC 复权生效
6. **报告产出**：HTML + CSV + PNG 图表完整生成
7. **GUI 链路**：可加载配置并触发任务，报告路径可打开，多因子指标可切换查看
8. **DL 配置校验**：`dl.mode/model_type/checkpoint_path/features` 错误可被显式拦截
9. **DL 训练推理**：`dl-train -> dl-infer -> report` 产出链路可执行（安装 `.[dl]` 后）
10. **成本后指标口径**：`summary.csv` 包含净收益字段（`ls_net_mean`、`ls_net_sharpe`）
11. **CLI 异常路径**：`run`/`dl-train`/`dl-infer` 失败时返回非 0 并输出可读错误信息

## 4. 回归记录模板

- 变更范围：
- 执行命令：
- 结果（pass/fail）：
- 风险点：
- 回滚策略：
