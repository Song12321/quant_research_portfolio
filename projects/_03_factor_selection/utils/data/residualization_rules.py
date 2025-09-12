"""
时间序列残差化规则设计

核心思路：
- 只对高自相关、我们关心"变化"的因子进行残差化
- 基于因子的经济含义和统计特性进行分类判断
"""

from typing import Set
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


##
#  🎯 核心原理分析
#
#   残差化的目的：提取因子的"增量信息"而非"水平信息"，适用于：
#   - 高自相关性的因子（今天值≈昨天值）
#   - 我们关心"变化"而非"绝对水平"的因子#
##
# 到底哪些因子需要“时间序列残差化”？
# 现在回答你的核心问题。这个操作并非多多益善，它只适用于那些因子值本身有很强自相关性（今天的数值和昨天的很像），而我们认为**“变化”比“绝对水平”更重要**的场景。
#
# 你的代码里将其指向microstructure派因子，是教科书式的应用。以下是最需要考虑进行残差化的几类因子：
#
# 波动率/风险类因子 (Volatility / Risk)
#
# 例子： volatility_90d (90日波动率), beta。
#
# 逻辑： 一只高风险股票（如科技股）的波动率长期来看总是比一只低风险股票（如公用事业股）要高。如果我们只看波动率的绝对值，选出来的永远是那批“老面孔”。但真正的alpha信号往往隐藏在波动率的异常变化中。比如，一只一向平稳的公用事业股，其波动率突然从10%飙升到20%，这个“异常”信号（残差为正）可能预示着未来的下跌。而一只科技股波动率从50%降到40%，这个“异常平静”信号（残差为负）可能预示着企稳。
#
# 流动性因子 (Liquidity)
#
# 例子： amihud_liquidity (Amihud非流动性), turnover_rate (换手率)。
#
# 逻辑： 同理，大盘股的换手率通常系统性地低于小盘股。我们更关心的是换手率的突然放大或萎缩。一只股票的换手率突然从1%放大到5%（残差为正），这背后可能有利好/利空消息驱动，是重要的交易信号。
#
# 部分情绪类因子 (Sentiment)
#
# 例子： 分析师情绪、新闻舆情。
#
# 逻辑： 市场对某些明星股的关注度和情绪可能长期维持在高位。我们更关心的是情绪的边际变化，比如当分析师情绪首次出现下调时。#




##

#
# 作为风险因子 (Risk Factor)： 在做风险模型、组合优化、控制回撤时，我们关心的是它的绝对水平。此时，不应残差化。
#
# 作为Alpha因子 (Alpha Factor)： 在构建“低波异象”策略，预测未来收益时，我们更关心波动率的“异常”变化。此时，应该残差化，就像我们上一轮讨论的那样。
# #
def need_residualization_in_neutral_processing(factor_name: str, style_category: str = None) -> bool:
    """
    判断因子是否需要时间序列残差化
    
    Args:
        factor_name: 因子名称
        style_category: 因子风格类别（可选）
        
    Returns:
        bool: 是否需要残差化
    """
    #
    # # 方案一：基于因子名称的精确匹配（优先级最高）
    # if factor_name in get_residualization_factor_whitelist():
    #     logger.debug(f"✅ {factor_name}: 在残差化白名单中")
    #     return True

    #先注释调。 我自己强保证！
    # # 方案二：基于因子名称模式匹配
    # if _match_factor_name_patterns(factor_name):
    #     logger.debug(f"✅ {factor_name}: 匹配残差化模式")
    #     return True
    
    # 方案三：基于style_category的规则判断
    if style_category and _need_residualization_by_category(style_category, factor_name):
        logger.debug(f"✅ {factor_name}: 基于类别({style_category})需要残差化")
        return True
    
    logger.debug(f"❌ {factor_name}:基于类别({style_category}) 不需要残差化")
    return False


def get_residualization_factor_whitelist() -> Set[str]:
    """
    需要残差化的因子白名单（基于因子名称精确匹配）
    
    这些因子具有以下特征：
    1. 高度自相关（水平值变化缓慢）
    2. 我们更关心其变化趋势而非绝对水平
    3. 残差化能提取增量信息
    """
    return {
        # === 流动性类因子 ===
        'turnover_rate_90d_mean',      # 长期平均换手率变化更重要
        'turnover_rate_monthly_mean',   # 月度换手率变化
        'ln_turnover_value_90d',       # 成交额变化

        # === 情绪/技术指标类因子 ===  
        'rsi',                         # RSI指标的变化比绝对值更重要
        'cci',                         # CCI指标的突破更重要
        
        # === 部分波动率因子 ===
        # 注意：不是所有波动率因子都需要残差化
        'volatility_90d',         # 当作为Alpha因子（低波异象）时，可考虑残差化以捕捉“异常波动”。

        # 'volatility_40d',            # 波动率的绝对水平也很重要，暂不残差化（但作为风险因子时，应使用其绝对水平。此处默认不处理。

        # === 市场关注度/情绪类 ===
        # 'analyst_attention',         # 分析师关注度变化
        # 'news_sentiment_change',     # 新闻情绪变化
    }


def _match_factor_name_patterns(factor_name: str) -> bool:
    """
    基于因子名称模式匹配判断是否需要残差化
    """
    
    # 模式1：名称中包含"change"、"delta"等变化词汇的因子
    change_keywords = ['change', 'delta', 'diff', 'variation']
    if any(keyword in factor_name.lower() for keyword in change_keywords):
        return True
    
    # 模式2：技术指标类因子（通常高自相关）
    technical_indicators = ['rsi', 'cci', 'macd', 'kdj', 'stoch']
    if any(indicator in factor_name.lower() for indicator in technical_indicators):
        return True
    
    # 模式3：移动平均类因子（高自相关）
    moving_avg_patterns = ['_ma_', '_sma_', '_ema_', '_mean_']
    if any(pattern in factor_name.lower() for pattern in moving_avg_patterns):
        return True
    
    return False

def _need_residualization_by_category(style_category: str, factor_name: str) -> bool:
    """
    【修正版 V2】基于因子类别判断是否需要残差化（使用if/elif/else避免预执行问题）
    """
    category = style_category.lower()

    if category == 'liquidity':
        return _is_high_autocorr_liquidity_factor(factor_name)
    elif category == 'sentiment':
        return _is_high_autocorr_sentiment_factor(factor_name)
    elif category == 'technical':
        return _is_high_autocorr_technical_factor(factor_name)
    elif category == 'momentum':
        return _is_high_autocorr_momentum_factor(factor_name)
    elif category == 'risk':
        return _is_high_autocorr_risk_factor(factor_name)
    elif category in ['value', 'quality', 'growth', 'size','return','event','market_microstructure','money_flow','sw_industry','hodgepodge_combo','vol','volume_price']:
        # 对于明确不需要的类别，直接返回False
        return False
    else:
        # 对于未知的类别，可以返回一个安全的默认值，或者抛出异常
        raise ValueError(f"未知的因子类别: {style_category} factor_name:{factor_name}")
def _is_high_autocorr_liquidity_factor(factor_name: str) -> bool:
    """
    判断是否为高自相关的流动性因子
    """
    # 需要残差化的流动性因子
    high_autocorr_liquidity = {

    }
    
    # 不需要残差化的流动性因子
    low_autocorr_liquidity = {
        'amihud_liquidity',    # Amihud流动性的绝对水平很重要
        'turnover_rate',       # 单日换手率波动本身就大
        'turnover_t1_div_t20d_avg',       # 单日换手率波动本身就大
        #亲测下面残差后表现极差！
        'turnover_rate_90d_mean', # 如果残差：-0.05315123787885707 --->-0.01433680051466378  10d     如果不做残差化直接中性化：-0.04663723878250377
        'turnover_rate_monthly_mean',#如果残差：-0.031346797401695686  -0.006203743313116326
        'ln_turnover_value_90d',#如果残差：-0.028998347667171864   -0.018800373908799514
        'turnover_42d_std_252d_std_ratio',
    }
    
    if factor_name in high_autocorr_liquidity:
        return True
    elif factor_name in low_autocorr_liquidity:
        return False
    else:
        # 默认情况下，对于未明确识别的流动性因子，采取保守策略，不进行残差化。
        # 这样做是为了避免“假阳性”——即错误地处理了一个本不该处理的因子。
        # 如果发现新的流动性因子需要残差化，应将其显式地添加到 high_autocorr_liquidity 集合中。
        raise ValueError(f"请明确是否用于中性化前的残差化{factor_name}")


def _is_high_autocorr_sentiment_factor(factor_name: str) -> bool:
    """
    判断是否为高自相关的因子
    """
    # 需要残差化的因子
    high_autocorr = {

    }

    # 不需要残差化的因子
    low_autocorr = {
        'rsi',  # 原先经过残差化   -0.017846921724055536,-0.008593362640905006
        'cci',  # 原先经过残差化  -0.017931541610051054,-0.007463005120771512
    }

    if factor_name in high_autocorr:
        return True
    elif factor_name in low_autocorr:
        return False
    else:
        # 默认情况下，对于未明确识别的 因子，采取保守策略，不进行残差化。
        # 这样做是为了避免“假阳性”——即错误地处理了一个本不该处理的因子。
        # 如果发现新的因子需要残差化，应将其显式地添加到 high_autocorr_ 集合中。
        raise ValueError(f"请明确是否用于中性化前的残差化{factor_name}")


def _is_high_autocorr_technical_factor(factor_name: str) -> bool:
    """判断是否为高自相关的技术指标因子"""
    high_autocorr = {
      'macd_signal', # MACD的信号线通常较平滑
    }
    low_autocorr = {
        'macd_hist', # MACD的柱状图本身就是差异，波动大


    }
    if factor_name in high_autocorr:
        return True
    elif factor_name in low_autocorr:
        return False
    else:
        raise ValueError(f"技术指标因子'{factor_name}'需要被明确分类（是否需要残差化）。")

def _is_high_autocorr_risk_factor(factor_name: str) -> bool:
    """判断是否为高自相关的风险因子"""
    # 当波动率作为Alpha因子（低波异象）时，我们关心其“异常变化”，因此需要残差化。
    high_autocorr = {

    }
    # Beta的绝对水平是核心，代表系统风险敞口，不应残差化。
    low_autocorr = {
        'beta',
        'volatility_120d',# 原先经过残差化 -0.04437990893204699,-0.010864010877199395

        'volatility_90d',# 原先经过残差化 -0.06692222000796344,-0.010442950337841927
      

        'volatility_40d', # 原先经过残差化：-0.05480784268382542 -0.0025366013058551106
        'volatility_20d',
    }
    if factor_name in high_autocorr:
        return True
    elif factor_name in low_autocorr:
        return False
    else:
        raise ValueError(f"风险因子'{factor_name}'需要被明确分类（是否需要残差化）。")

def _is_high_autocorr_momentum_factor(factor_name: str) -> bool:
    """判断是否为高自相关的动量因子"""
    high_autocorr = {
        # 默认情况下，我们不对动量因子进行残差化。
        # 仅在进行“动量加速度”等高级研究时，才将对应动量因子移入此列表。
    }
    # 动量因子本身是价格变化率，反转因子是短期价格行为，默认不进行二次“变化”计算。
    low_autocorr = {
        'momentum_250d',
        'momentum_120d',
        'momentum_pct_60d',
        'momentum_20d',
        'reversal_5d',
        'reversal_21d',
        'momentum_12_1',
        'momentum_1d',
        'momentum_5d',
        'momentum_12_2',
        'sharpe_momentum_60d',
        'quality_momentum', # 组合因子，默认不处理
    }
    if factor_name in high_autocorr:
        return True
    elif factor_name in low_autocorr:
        return False
    else:
        raise ValueError(f"动量/反转因子'{factor_name}'需要被明确分类（是否需要残差化）。")
def get_residualization_config(factor_name: str) -> dict:
    """
    获取特定因子的残差化配置参数
    
    不同因子可能需要不同的残差化窗口
    """
    
    # 默认配置
    default_config = {
        'window': 20,
        'min_periods': 10
    }
    
    # 特殊因子的定制配置
    custom_configs = {
        # 流动性因子：稍长窗口捕捉流动性制度变化
        'turnover_rate_90d_mean': {'window': 30, 'min_periods': 15},
        'ln_turnover_value_90d': {'window': 30, 'min_periods': 15},
        
        # 技术指标：较短窗口保持敏感性
        'rsi': {'window': 10, 'min_periods': 5},
        'cci': {'window': 15, 'min_periods': 8},
    }
    
    return custom_configs.get(factor_name, default_config)


def print_residualization_summary():
    """
    打印残差化规则汇总，便于检查和调试
    """
    
    print("📊 时间序列残差化规则汇总")
    print("=" * 50)
    
    print("\n✅ 白名单因子（精确匹配）:")
    whitelist = get_residualization_factor_whitelist()
    for factor in sorted(whitelist):
        config = get_residualization_config(factor)
        print(f"  - {factor:30s} (窗口: {config['window']}天)")
    
    print(f"\n📋 按类别规则:")
    print(f"  - liquidity: 部分因子（高自相关的）")
    print(f"  - sentiment: 全部需要") 
    print(f"  - technical: 全部需要")
    print(f"  - 其他类别: 默认不需要")
    
    print(f"\n🔍 模式匹配规则:")
    print(f"  - 包含 'change', 'delta' 等变化词汇")
    print(f"  - 技术指标名称：rsi, cci, macd 等")
    print(f"  - 移动平均类：包含 '_mean_', '_ma_' 等")


if __name__ == "__main__":
    # 测试一些因子
    test_factors = [
        ('turnover_rate_90d_mean', 'liquidity'),
        ('rsi', 'sentiment'),
        ('bm_ratio', 'value'),
        ('volatility_90d', 'risk'),
        ('turnover_t1_div_t20d_avg', 'liquidity'),
        ('momentum_120d', 'momentum'),
        ('earnings_stability', 'quality')
    ]
    
    print("🧪 残差化规则测试:")
    print("-" * 40)
    
    for factor_name, style_cat in test_factors:
        need_resid = need_residualization_in_neutral_processing(factor_name, style_cat)
        status = "✅ 需要" if need_resid else "❌ 不需要" 
        print(f"{factor_name:25s} ({style_cat:10s}): {status}")
    
    print()
    print_residualization_summary()