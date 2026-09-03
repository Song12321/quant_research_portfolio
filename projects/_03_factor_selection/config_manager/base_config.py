# 常用指数代码配置
from pathlib import Path

INDEX_CODES = {
    "HS300": "000300",     # 沪深300
    "ZZ500": "000905",     # 中证500
    "ZZ800": "000906",     # 中证800
    "ZZ1000": "000852",    # 中证1000
    'ZZ_ALL': "000985",    #中证全指
}
# 定义了每个因子派系默认应该被哪些风险因子中性化
# 这是你的“风险模型”的核心
FACTOR_STYLE_RISK_MODEL = {
    'value': ['market_cap', 'industry'],
    'quality': ['market_cap', 'industry'],
    'growth': ['market_cap', 'industry'],
    'momentum': ['market_cap', 'industry', 'pct_chg_beta'],
    'reversal': ['market_cap', 'industry', 'pct_chg_beta'],
    'risk': ['market_cap', 'industry'],  # 测试风险因子时，默认只对其他基础风险因子中性化
    'sentiment': ['market_cap', 'industry', 'pct_chg_beta'],
    'technical': ['market_cap', 'industry', 'pct_chg_beta'],
    # 默认配置，适用于未明确分类的因子
    'default': ['market_cap', 'industry']
}
# FACTOR_STYLE_RISK_MODEL = {
#     'value': ['market_cap', 'industry'],
#     'quality': ['market_cap', 'industry'],
#     'growth': ['market_cap', 'industry'],
#     'momentum': ['market_cap', 'industry', 'pct_chg_beta'],
#     'reversal': ['market_cap', 'industry', 'pct_chg_beta'],
#     'risk': ['market_cap', 'industry'],  # 测试风险因子时，默认只对其他基础风险因子中性化
#     'sentiment': ['market_cap', 'industry', 'pct_chg_beta'],
#     'technical': ['market_cap', 'industry', 'pct_chg_beta'],
#     # 默认配置，适用于未明确分类的因子
#     'default': ['market_cap', 'industry']
# }


workspaces_dir = Path(r'D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace')
workspaces_result_dir = workspaces_dir / 'result'
config_yaml_path=r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factory\config.yaml'
experiments_yaml_path=r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factory\experiments.yaml'
