"""
配置常量模块

存储项目中使用的各种常量配置。
"""

import os
from pathlib import Path

# 路径配置
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'
RESULT_DIR = ROOT_DIR / 'results'
LOG_DIR = ROOT_DIR / 'logs'
CONFIG_DIR = ROOT_DIR / 'config_manager'

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 市场数据路径：未来整体迁移时，只修改此处的唯一绝对路径。
MARKET_DATA_ROOT = Path(r'D:\lqs\quantity\market_data')

# 所有数据集均以 MARKET_DATA_ROOT 为根目录。键沿用项目内已有的逻辑数据集名。
_DATASET_RELATIVE_PATHS = {
    'trade_cal.parquet': Path('shared/trade_cal.parquet'),
    'stock_basic.parquet': Path('stock/reference/stock_basic.parquet'),
    'industry_record.parquet': Path('stock/reference/industry_record.parquet'),
    'daily': Path('stock/quotes/daily'),
    'daily_hfq': Path('stock/quotes/daily_hfq'),
    'adj_factor': Path('stock/quotes/adj_factor'),
    'daily_basic': Path('stock/market_metrics/daily_basic'),
    'margin_detail': Path('stock/market_metrics/margin_detail'),
    'stk_limit': Path('stock/trading_constraints/stk_limit'),
    'suspend_d.parquet': Path('stock/trading_constraints/suspend_d.parquet'),
    'balancesheet.parquet': Path('stock/fundamentals/balancesheet.parquet'),
    'cashflow.parquet': Path('stock/fundamentals/cashflow.parquet'),
    'income.parquet': Path('stock/fundamentals/income.parquet'),
    'fina_indicator.parquet': Path('stock/fundamentals/fina_indicator.parquet'),
    'dividend.parquet': Path('stock/corporate_actions/dividend.parquet'),
    'namechange.parquet': Path('stock/corporate_actions/namechange.parquet'),
    'index_daily.parquet': Path('index/broad_market/quotes/index_daily.parquet'),
    'index_weights': Path('index/broad_market/constituents/index_weights'),
    'sw_basic_info.parquet': Path('index/shenwan/reference/sw_basic_info.parquet'),
    'sw_daily.parquet': Path('index/shenwan/quotes/sw_daily.parquet'),
}


def get_market_data_path(dataset_name: str, root: Path | None = None) -> Path:
    """返回已登记数据集的路径；未登记名称必须由调用方修正。"""
    try:
        relative_path = _DATASET_RELATIVE_PATHS[dataset_name]
    except KeyError as exc:
        raise ValueError(f'未知市场数据集: {dataset_name}') from exc

    data_root = MARKET_DATA_ROOT if root is None else Path(root)
    return data_root / relative_path

# 回测配置
DEFAULT_BENCHMARK = '000300.SH'
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_SLIPPAGE_RATE = 0.0002
DEFAULT_CAPITAL = 10000000.0

# 因子配置
DEFAULT_FACTOR_WEIGHTS = {
    'value': 0.3,
    'momentum': 0.3,
    'quality': 0.2,
    'growth': 0.1,
    'volatility': 0.1
}

# 时间配置
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5


# 行业分类配置
INDUSTRY_CLASSIFICATION = {
    'sw_l1': '申万一级行业',
    'sw_l2': '申万二级行业',
    'sw_l3': '申万三级行业',
    'csi_l1': '中证一级行业',
    'csi_l2': '中证二级行业'
}

# 日志配置
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# API配置
TUSHARE_TOKEN_PATH = ROOT_DIR / 'quant_lib' / 'tushare' / 'tushare_token_manager' / 'token.txt'

# 绘图配置
PLOT_STYLE = 'seaborn'
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 100
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

parquet_file_names = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq', 'fina_indicator.parquet', 'index_weights',
                     'margin_detail', 'stk_limit']
every_day_parquet_file_names = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq', 'index_weights',
                     'margin_detail', 'stk_limit']
#'index_weights'  fina_indicator
need_fix = [ 'daily', 'daily_basic', 'daily_hfq',
                     'margin_detail', 'stk_limit']

permanent__day = '22000101'
