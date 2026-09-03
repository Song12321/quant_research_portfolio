# 因子研究

当前模块负责 Inner 单因子有效性研究，并为后续 Out 阶段增量冻结因子方向。

## 配置

- `configs/factors/definitions/*.yaml`：按 `style_category` 拆分的因子定义；运行时按文件名顺序严格合并。
- `configs/research/inner.yaml`：Inner 日期、唯一股票池、实验名单、预处理和评价周期。`stock_pool_name` 是顶层必填字段，`experiments` 每项只配置 `factor_name`。
- `configs/factors/inner_resolved_directions.yaml`：Inner 完成后增量写入的方向配置。

正式入口只读取 `inner.yaml`，不会再由 Python 运行模式覆盖日期、股票池或评价参数。

## 正式流程

1. 创建不可覆盖的 `runs/inner/<run_id>` 目录并保存完整生效配置。
2. 只常驻构建股票池所需的基础数据；每个因子的其他原料在计算时临时读取。
3. 单个因子内部允许复用原料和中间因子；该因子结束后立即清空缓存。
4. 单因子统一执行去极值、中性化、标准化、Spearman IC、分层收益和换手研究。
5. 复合因子的子因子必须先在同一次 Inner 完成并冻结方向；同次运行天然共享顶层股票池，分别 processed 后乘该方向、等权平均，再标准化。
6. 对 `inner.yaml.evaluation.forward_periods` 中每个周期取得 `ic_mean` 和非重叠 IC 节点数 `ic_Valid Days`，按节点数加权计算方向分数：`sum(ic_mean * ic_Valid Days) / sum(ic_Valid Days)`。
7. 加权分数大于零记为 `direction: 1`，小于零记为 `direction: -1`；任一周期无效、缺少有效节点数或最终为零时停止。冻结结果同时记录各周期节点数和归一化权重。
8. 方向增量写入 `inner_resolved_directions.yaml`；已有同名因子禁止覆盖。

运行目录包含：

```text
runs/inner/<run_id>/
├─ effective_config.yaml
├─ experiments.yaml
├─ manifest.json
├─ resolved_factors.yaml
├─ summary.json
└─ artifacts/
```

`resolved_factors.yaml` 是本次运行完成后的方向配置副本。Out 可以读取共享方向配置；Finalout 必须读取 Out 冻结的唯一套装，不得直接读取持续变化的 Inner 方向文件。

## 调用

```python
from projects._03_factor_selection.factory import run_test_by_config

run_test_by_config("Inner processed 因子研究")
```

## 当前边界

- 不评价 raw 与 processed 两套重复信号；单因子使用未预设方向的 processed 流程，复合因子只使用本次 Inner 已冻结的子因子方向。
- 不在本模块实现 Out 套装选择、Finalout 验收或投资组合回测。
- 不自动覆盖已有方向，不复用历史运行目录，不在关键失败后继续。
