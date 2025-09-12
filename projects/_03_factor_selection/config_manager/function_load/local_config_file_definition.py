from dataclasses import dataclass, field
from typing import List, Dict, Any
import copy
from dataclasses import dataclass, field, asdict

from projects._03_factor_selection.config_manager.base_config import INDEX_CODES


# 使用 @dataclass 装饰器，Python会自动为我们生成__init__, __repr__等方法
@dataclass
class IndexFilterConfig:
    """指数过滤配置"""
    enable: bool
    index_code: str
    def to_dict(self): return asdict(self)


@dataclass
class PoolFiltersConfig:
    """普适性的过滤器配置"""
    remove_st: bool = True
    remove_new_stocks: bool = True
    adapt_tradeable_matrix_by_suspend_resume: bool = True
    min_liquidity_percentile: float = 0.0
    min_market_cap_percentile: float = 0.0
    def to_dict(self): return asdict(self)


@dataclass
class StockPoolProfile:
    """单个股票池的完整配置"""
    index_filter: IndexFilterConfig
    filters: PoolFiltersConfig
    def to_dict(self): return asdict(self)


@dataclass
class BacktestConfig:
    """回测时间配置"""
    start_date: str
    end_date: str
    def to_dict(self): return asdict(self)


# 这是最顶层的完整配置对象
@dataclass
class FullQuantConfig:
    """最终生成的完整配置对象"""
    backtest: BacktestConfig
    stock_pool_profiles: Dict[str, StockPoolProfile]

    # 提供一个方法，方便地将自身转换为字典，以便系统其他部分使用
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest": self.backtest.to_dict(),
            "stock_pool_profiles": {
                name: profile.to_dict() for name, profile in self.stock_pool_profiles.items()
            }
        }

style_factor_list =  [
      # 1. 规模 (Size)
      'circ_mv',              # 基于流通市值，最核心的规模因子

      # 2. 价值 (Value)
      'bm_ratio',               # 账面市值比(B/P)，Fama-French三因子模型经典成员

      # 3. 质量 (Quality)
      'roe_ttm',                # 净资产收益率(TTM)，最核心的盈利质量指标

      # 4. 成长 (Growth)
      'net_profit_growth_ttm',  # TTM净利润同比增长，相比单季度YoY更平滑，代表稳定成长性

      # 5. 动量 (Momentum)
      'momentum_120d',          # 中期动量（约半年），代表价格趋势
      'reversal_21d',           # 短期反转，A股市场非常显著的独立效应

      # 6. 风险 (Risk / Volatility)
      'volatility_90d',         # 90日年化波动率，代表特质风险（低波异象）
      'beta',                   # Beta系数，代表市场系统性风险

      # 7. 流动性 (Liquidity)
      'ln_turnover_value_90d'   # 90日日均成交额对数，最核心的流动性/容量指标
  ]
# ==============================================================================
# 【新增】预设因子评价配置模板 (Evaluation Presets)
# ==============================================================================

# 模板1: 快速测试设置 (Fast/Debug Setting)
# 只跑一个核心周期，节省时间
EVAL_SETTING_FAST = {
    "forward_periods": [5,10], # 只跑一个周期
    "returns_calculator": ['o2o'], # 只用一种计算方式
    # style_factor_list 可以保持完整，因为它不影响计算速度
    "style_factor_list": [
    ]
}

# 模板2: 标准研究设置 (Standard Research Setting)
# 覆盖短、中、长周期，适用于大多数正式研究
EVAL_SETTING_STANDARD = {
    "forward_periods": [5, 21, 60], # 短、中、长
    "returns_calculator": [ 'o2o'], # 对比两种方式
    "style_factor_list":style_factor_list,
    "eva_data": ['raw','processed']  # 控制测试哪种数据状态：'raw'(原始)，'processed'(处理后)

}

# 模板3: 全面批量测试设置 (Massive Test Setting)
# 最全面的配置，用于最终的、彻底的因子有效性检验
EVAL_SETTING_FULL = {
    "quantiles": 5,
    "forward_periods": [1, 5, 10, 21, 40, 60, 120],
    "returns_calculator": [ 'o2o'],
    "style_factor_list": style_factor_list,
    "eva_data": ['raw', 'processed']  # 控制测试哪种数据状态：'raw'(原始)，'processed'(处理后)
}
# 东
dongbei_SETTING = {
    "quantiles": 5, # 兼容旧命名
    "forward_periods": [3],
    "returns_calculator": ['o2o'],
    "style_factor_list": [

    ]
}
fast_eva_SETTING = {
    "quantiles": 5, # 兼容旧命名
    "forward_periods": [3],
    "returns_calculator": ['o2o'],
    "style_factor_list": [

    ],
    'eva_data':['processed']
}
fast_eva_SETTING = {
    "quantiles": 5, # 兼容旧命名
    "forward_periods": [5],
    "returns_calculator": ['o2o'],
    "style_factor_list": [
    ],
    "eva_data": ['processed']  # 控制测试哪种数据状态：'raw'(原始)，'processed'(处理后)
}

really_eva_SETTING = {
    "quantiles": 5, # 兼容旧命名
    "forward_periods": [21],
    "returns_calculator": ['o2o'],
    "style_factor_list": [
    ],
    "eva_data": ['processed']  # 控制测试哪种数据状态：'raw'(原始)，'processed'(处理后)
}
# ==============================================================================
# 预设股票池模板 (Stock Pool Presets)
# 你可以在这里定义所有常用的股票池配置
# ==============================================================================

# 模板1: 机构标准池 (基于沪深300)

def make_pool_profile(pool_name, Index_filter, index_code,remove_st,remove_new_stocks,adapt_tradeable_matrix_by_suspend_resume, min_liquidity_percentile, min_market_cap_percentile):
    profile = StockPoolProfile(
        index_filter=IndexFilterConfig(enable=Index_filter, index_code=index_code),
        filters=PoolFiltersConfig(
            remove_st = remove_st,
            remove_new_stocks = remove_new_stocks,
            adapt_tradeable_matrix_by_suspend_resume = adapt_tradeable_matrix_by_suspend_resume,
            min_liquidity_percentile=min_liquidity_percentile,
            min_market_cap_percentile=min_market_cap_percentile
        )
    )
    return {
        pool_name:  # 没办法，写死吧，这个设计回旋镖打了自己
        profile
    }

HS300_fast_profile = make_pool_profile('HS300', True, INDEX_CODES['HS300'],False,False,False, 0, 0)
CSI300_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300',True,True,True, 0, 0)
CSI300_more_filter_profile = make_pool_profile('institutional_stock_pool', True, '000300.',True,True,True, 0.1, 0.05)
ZZ1000_more_filter_profile = make_pool_profile('ZZ1000', True, INDEX_CODES['ZZ1000'],True,True,True, 0.1, 0.05)
ZZ500_more_filter_profile = make_pool_profile('ZZ500', True, INDEX_CODES['ZZ500'],True,True,True, 0.1, 0.05)
ZZ1000_no_filter_profile = make_pool_profile('ZZ1000', True, INDEX_CODES['ZZ1000'],False,False,False, 0, 0)
HS300_no_filter_profile = make_pool_profile('HS300', True, INDEX_CODES['HS300'],False,False,False, 0, 0)
CSI300_none_TFF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300',True,False,False, 0, 0)
CSI300_none_FTF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300',False,True,False, 0, 0)
CSI300_none_FFT_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300',False,False,True, 0, 0)
CSI300_none_FFF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300',False,False,False, 0, 0)
ALL_none_FFF_most_basic_profile = make_pool_profile('ALL', False, INDEX_CODES['ZZ_ALL'],False,False,False, 0, 0)
fast_hs300_profile = make_pool_profile('fast', True, INDEX_CODES['HS300'], False, False, False, 0, 0)
东北_zz500_profile = make_pool_profile('ZZ500', True, INDEX_CODES['ZZ500'], True, True, True, 0, 0)
CSI500_none_FFF_most_basic_profile = make_pool_profile('microstructure_stock_pool', True, INDEX_CODES['ZZ500'],False,False,False, 0, 0)
CSI500_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000905', True,True,True,0, 0)
# 用于我需要在最真实的环境，交易，需要必须要过滤流动差劲的
_massive_test_ZZ800_profile = make_pool_profile('ZZ800', True, INDEX_CODES['ZZ800'], True, True, True, 0.1, 0.05)
temp_test_ZZ800_profile = make_pool_profile('ZZ800', True, INDEX_CODES['ZZ800'], True, False, True, 0, 0)
fast_ZZ800_profile = make_pool_profile('ZZ800', True, INDEX_CODES['ZZ800'], False, False, False, 0, 0)
fast_ZZ1000_profile = make_pool_profile('ZZ1000', True, INDEX_CODES['ZZ1000'], False, False, False, 0, 0)
pool_for_massive_test_MICROSTRUCTURE_profile = make_pool_profile('microstructure_stock_pool', False, 'None', True,True,True,0.2, 0.2)


def generate_dynamic_config(
        start_date: str,
        end_date: str,
        pool_profiles
) -> Dict[str, Any]:
    """
    【最终版】动态生成量化回测配置字典。

    Args:
        start_date (str): 回测开始日期, 'YYYY-MM-DD'
        end_date (str): 回测结束日期, 'YYYY-MM-DD'
        target_factors (List[str]): 要测试的因子名称列表, e.g., ['market_cap_log', 'beta']
        pool_custom_name (str): 生成的配置中，这个股票池的名字

    Returns:
        Dict[str, Any]: 一个完全合规的、可直接用于系统的配置字典。
    """
    print(f"🚀 正在动态生成配置...")
    print(f"   - 时间范围: {start_date} -> {end_date}")
    print(f"   - 股票池模板: {pool_profiles.keys()}")

    # 1. 检查预设是否存在

    # 2. 构建回测时间配置
    backtest_conf = BacktestConfig(start_date=start_date, end_date=end_date)


    # 4. 构建股票池配置 (使用深拷贝以防修改原始模板)
    #    这里只生成一个股票池，因为动态配置通常是针对单次实验的

    # 5. 组装成最终的完整配置对象
    full_config = FullQuantConfig(
        backtest=backtest_conf,
        stock_pool_profiles=pool_profiles
    )

    # 6. 返回字典格式
    return full_config.to_dict()
