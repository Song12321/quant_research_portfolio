# 🕵️ 数据侦探工具 (DataForensics)

一个专业的量化数据质量诊断工具，用于智能分析数据中的NaN值，区分"合理缺失"和"可疑缺失"。

## 🎯 核心功能

### 智能NaN归因分析
- **上市前缺失** ✅ - 股票尚未上市，NaN合理
- **退市后缺失** ✅ - 股票已经退市，NaN合理  
- **停牌期间缺失** ℹ️ - 交易期间停牌，NaN合理
- **可疑缺失** ❓ - 无法解释的NaN，可能存在数据质量问题

### 高效向量化计算
- 使用pandas向量化操作，避免低效的循环
- 支持大规模数据集处理
- 智能缓存机制，减少重复数据加载

### 多种使用模式
- **单字段诊断** - 深入分析特定字段
- **批量诊断** - 同时检查多个字段
- **质量报告** - 生成详细的数据质量评估
- **交互式模式** - 友好的命令行界面

## 📁 文件结构

```
quant_research_portfolio/
├── quant_lib/
│   └── data_forensics.py          # 核心数据侦探类
├── scripts/
│   ├── run_data_forensics.py      # 完整使用脚本
│   └── test_data_forensics.py     # 测试脚本
├── docs/
│   ├── data_forensics_guide.md    # 详细使用指南
│   └── installation_guide.md      # 安装指南
├── demo_data_forensics.py         # 演示版本（无需依赖）
└── test_forensics_basic.py        # 基础测试脚本
```

## 🚀 快速开始

### 1. 演示版本（推荐先试用）

无需安装任何依赖，直接运行演示：

```bash
python demo_data_forensics.py
```

### 2. 完整版本

安装依赖后使用完整功能：

```bash
# 安装依赖
pip install pyarrow pandas numpy

# 交互式模式
python scripts/run_data_forensics.py --mode interactive

# 单字段诊断
python scripts/run_data_forensics.py --mode single --field close --dataset daily_hfq

# 批量诊断
python scripts/run_data_forensics.py --mode batch

# 生成质量报告
python scripts/run_data_forensics.py --mode report
```

### 3. 编程接口

```python
from quant_lib.data_forensics import DataForensics

# 初始化
forensics = DataForensics()

# 诊断单个字段
forensics.diagnose_field_nan(
    field_name='close',
    dataset_name='daily_hfq',
    sample_stocks=8,
    detailed_analysis=True
)

# 批量诊断
batch_fields = [
    ('close', 'daily_hfq'),
    ('pe_ttm', 'daily_basic'),
    ('vol', 'daily_hfq')
]
forensics.batch_diagnose(batch_fields)

# 生成质量报告
report = forensics.generate_data_quality_report('quality_report.json')
```

## 📊 输出示例

### 归因分析结果
```
📋 NaN归因分析结果:
  ✅ 上市前缺失: 1,234,567 (45.2%) - 合理
  ✅ 退市后缺失: 234,567 (8.6%) - 合理  
  ℹ️  交易期间缺失: 1,123,456 (41.2%) - 大概率停牌
  ❓ 未知股票缺失: 156,789 (5.7%) - 需要检查

✅ 数据质量良好，大部分NaN都有合理解释。
```

### 详细个股分析
```
🔬 详细个股分析 (抽样 5 只股票):

[1] 股票: 000001.SZ (NaN数量: 245)
    📅 上市: 1991-04-03, 退市: 未退市
    ℹ️  交易期间NaN: 245个
       -> 形成 12 个连续缺失区间
       -> 最近缺失日期: ['2024-03-15', '2024-06-20', '2024-09-10']
```

## 🎮 交互式模式

启动交互式模式后，可以使用以下命令：

```bash
🔍 请输入命令: help                              # 显示帮助
🔍 请输入命令: diagnose close daily_hfq 8        # 诊断字段
🔍 请输入命令: batch                             # 批量诊断
🔍 请输入命令: report                            # 生成报告
🔍 请输入命令: quit                              # 退出
```

## 🔧 支持的数据格式

### 分区数据集
- `daily_hfq/` - 后复权日线数据
- `daily_basic/` - 每日基本面数据
- `daily/` - 原始日线数据
- 其他按年份分区的数据集

### 单文件数据集
- `stock_basic.parquet` - 股票基本信息
- `trade_cal.parquet` - 交易日历
- `namechange.parquet` - 股票名称变更

## 📈 诊断逻辑

数据侦探通过以下逻辑判断NaN的合理性：

1. **时间维度检查**
   - 上市前：`NaN日期 < 上市日期` → 合理 ✅
   - 退市后：`NaN日期 > 退市日期` → 合理 ✅

2. **交易状态检查**
   - 交易期间：大概率为停牌 → 基本合理 ℹ️
   - 连续缺失：分析停牌区间的连续性

3. **异常检测**
   - 无法归因的NaN → 可疑，需要检查 ❓

## 🛠️ 高级功能

### 自定义诊断
```python
# 自定义抽样和分析深度
forensics.diagnose_field_nan(
    field_name='turnover_rate',
    dataset_name='daily_basic',
    sample_stocks=15,
    detailed_analysis=True
)
```

### 质量评分
- **优秀** (95%+): 数据质量很好
- **良好** (80-95%): 基本可用，少量异常
- **需要关注** (<80%): 存在较多质量问题

### 批量处理
```python
# 定义检查清单
core_fields = [
    ('close', 'daily_hfq'),      # 收盘价
    ('vol', 'daily_hfq'),        # 成交量
    ('pe_ttm', 'daily_basic'),   # 市盈率
    ('pb', 'daily_basic'),       # 市净率
]

forensics.batch_diagnose(core_fields, sample_stocks=3)
```

## 📋 最佳实践

1. **定期检查**: 每次数据更新后运行诊断
2. **重点关注**: 优先检查核心字段（价格、成交量等）
3. **阈值设置**: 根据业务需求设置质量分数阈值
4. **结果存档**: 保存历史质量报告，跟踪趋势
5. **自动化**: 集成到数据管道中

## 🚨 故障排除

### 常见问题
- **依赖问题**: 参考 `docs/installation_guide.md`
- **路径配置**: 检查 `MARKET_DATA_ROOT` 设置
- **内存不足**: 减少 `sample_stocks` 参数
- **字段不存在**: 使用 `df.columns` 检查实际字段名

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 使用场景

### 场景1: 新数据验证
```python
# 验证新下载的数据
forensics.diagnose_field_nan('close', 'daily_hfq')
```

### 场景2: 定期质量检查
```python
# 生成月度质量报告
report = forensics.generate_data_quality_report(
    f'monthly_report_{datetime.now().strftime("%Y%m")}.json'
)
```

### 场景3: 问题排查
```python
# 深入分析特定字段的问题
forensics.diagnose_field_nan(
    field_name='suspicious_field',
    dataset_name='problematic_dataset',
    sample_stocks=20,
    detailed_analysis=True
)
```

## 📞 技术支持

- 📖 详细文档: `docs/data_forensics_guide.md`
- 🔧 安装指南: `docs/installation_guide.md`
- 🧪 测试脚本: `test_forensics_basic.py`
- 🎮 演示版本: `demo_data_forensics.py`

---

💡 **核心价值**: 数据侦探工具不仅能发现数据问题，更重要的是帮助你理解数据的特性和局限性，为量化分析提供可靠的数据基础。

🎉 **开始使用**: `python demo_data_forensics.py`
