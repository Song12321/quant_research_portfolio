# 因子研究

当前模块只负责盈利系统的研究前半段：验证因子有效性，并产出可供后续选因子套装和回测读取的正式信号。

## 正式流程

1. 按 `factory/experiments.yaml` 读取因子与股票池。
2. 单因子统一经过现有的去极值、中性化、标准化，正式信号只保留 `processed`。
3. 复合因子的每个子因子先走同一套 processed 流程，再严格等权平均并最终标准化。
4. 对正式信号计算 Spearman IC、分层周期收益、分层每日收益和头部分组换手率。
5. 保存 `processed_factor.parquet`、IC/分层序列和精简后的 `summary_stats.json`。

正式入口：

```python
from projects._03_factor_selection.factory import run_test_by_config

run_test_by_config("processed 因子研究")
```

## 明确不包含

- raw 信号的重复评价；
- IC 加权、滚动 IC 选权和正交化合成；
- Fama-MacBeth、风格相关性、综合打分和研究报告绘图；
- 自动选因子套装以及 Inner / Out / Finalout；
- 投资组合回测。

选套装与回测将在后续阶段基于本阶段保存的 processed 产物单独实现。
