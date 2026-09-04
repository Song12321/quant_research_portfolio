# Inner / Out / Finalout 时间划分规则

## 一、最终时间划分

本规则适用于候选持有和评价周期 `5D`、`10D`、`20D`。三个周期使用完全相同的连续日期，便于一次运行并进行公平比较。

| 候选周期 | Inner | Out | Finalout | 总评价期 |
| --- | ---: | ---: | ---: | ---: |
| 5D | 2年 | 1年 | 1年 | 4年 |
| 10D | 2年 | 1年 | 1年 | 4年 |
| 20D | 2年 | 1年 | 1年 | 4年 |

统一结论：

\[
\boxed{5D、10D、20D：2年\ Inner+1年\ Out+1年\ Finalout}
\]

四年必须按时间连续排列，顺序只能是 `Inner → Out → Finalout`，不得随机划分或相互重叠。

## 二、预热期

预热期不固定为五年，也不计入四年评价期。

\[
\boxed{预热长度=候选因子的实际最大历史回看窗口}
\]

例如，20日均值需要至少20个交易日历史，252日分位需要至少252个交易日历史。只有因子公式明确使用五年滚动统计时，才需要相应的五年原始历史。预热数据只能用于计算评价期首日的因子值，不得用于选择周期、参数或判断收益表现。

## 三、三个阶段的职责

### Inner

- 允许研究因子定义、方向、预处理、组合方法和交易成本。
- 同时评价 `5D`、`10D`、`20D`，并且只能根据 Inner 结果确定最终周期。
- 必须记录全部成功和失败的因子、参数、周期及组合尝试。

### Out

- 进入 Out 前必须冻结候选全集、周期选择规则、因子方向、预处理、费用和通过标准。
- Out 可以按照预先确定的规则选择唯一最终套装。
- 查看 Out 后发生任何模型、周期、方向或阈值修改，原 Out 立即归入 Inner，不得继续称为样本外验证。

### Finalout

- 只允许评价 Out 冻结的唯一套装。
- 不得在 Finalout 中重新选择因子、周期、方向、参数或费用设置。
- Finalout 失败后不得修改方案并继续复用同一时间段。

## 四、各周期的样本数量

按每年约252个交易日、收益标签互不重叠估算：

\[
N\approx\frac{评价年数\times252}{持有周期}
\]

| 周期 | 2年 Inner | 1年 Out | 1年 Finalout |
| --- | ---: | ---: | ---: |
| 5D | 约100次 | 约50次 | 约50次 |
| 10D | 约50次 | 约25次 | 约25次 |
| 20D | 约25次 | 约12次 | 约12次 |

这些数值只是按持有周期隔开的时间锚点数量，不代表已经证明相互独立。共同市场冲击、行业相关性和因子持续性仍会造成时间依赖。

## 五、20D 的统计边界

20D 在一年 Out 或 Finalout 中只有约12个非重叠周期，统计把握度明显弱于5D和10D。因此：

- 一年 Out 对20D主要用于淘汰明显失败的因子，不作为充分有效性的独立证明。
- 应保留全部日度20D RankIC，并使用预先固定的 Newey-West/HAC 规则处理重叠标签造成的自相关；不能把日度IC数量直接当作独立样本数。
- 应检查不同20D起始偏移下的结果是否一致，但不同偏移结果高度相关，不得累加为独立证据。
- 20D 通过 Finalout 后，继续用新增模拟实盘或真实实盘数据积累证据，不通过扩张到更早历史来改善结果。

本时间方案明确选择“近期有效性和统一运行便利”，同时接受20D统计把握度较弱这一限制。

## 六、阶段边界

每条收益标签必须完整留在所属阶段：

\[
\boxed{label\_end<next\_stage\_start}
\]

按周期分别处理时，5D、10D、20D分别剔除阶段尾部可能跨界的标签。若为了实现简单而共用统一边界，则按最大20D处理：每个阶段最后20个交易日不参加该阶段收益评价。日期计算必须使用实际交易日，而不是自然日。

## 七、一次运行的限制

可以由一个命令完成数据准备和三个周期的计算，但阶段结果必须依次冻结和解封：

1. 只读取 Inner 结果并确定唯一周期；
2. 冻结周期和研究配置后读取 Out；
3. Out 冻结唯一套装后读取 Finalout；
4. 如果一次运行直接展示三个阶段的全部结果，Out 和 Finalout 均视为已污染，只能作为 Inner 开发结果。

因此，允许“一次命令运行”，不允许“一次性查看并据全部结果选择方案”。

## 八、参考依据

- [Investment Model Validation: A Guide for Practitioners](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf)：区分训练、验证和最终测试，并要求时间序列验证保持时间顺序。
- [A Backtesting Protocol in the Era of Machine Learning](https://people.duke.edu/~charvey/Research/Published_Papers/SSRN-id3275654.pdf)：强调预先确定样本、记录全部试验，以及反复使用样本外数据会使其失去样本外性质。
- [A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix](https://www.nber.org/papers/t0055)：Newey-West/HAC 稳健协方差估计依据。
- [A Taxonomy of Anomalies and Their Trading Costs](https://academic.oup.com/rfs/article-abstract/29/1/104/1844518)：交易成本、换手率与因子可实现性依据。

