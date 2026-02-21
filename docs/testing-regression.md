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
2. **因子算子**：rolling/rank/zscore/winsorize/neutralize 输出有效
3. **评估指标**：IC/RankIC、分层收益、多空收益、换手非空
4. **防坑逻辑**：前视算子拦截、可交易性过滤生效
5. **报告产出**：HTML + CSV + PNG 图表完整生成
6. **GUI 链路**：可加载配置并触发任务，报告路径可打开

## 4. 回归记录模板

- 变更范围：
- 执行命令：
- 结果（pass/fail）：
- 风险点：
- 回滚策略：
