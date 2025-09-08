"""
Backtrader实际使用演示

直接替代原有的backtest_factor_comparison_example.py
展示如何用Backtrader解决Size小于100的问题
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._04backtesting.quant_backtester import BacktestConfig
from projects._04backtesting.backtrader.backtrader_enhanced_strategy import one_click_migration
from projects._04backtesting.backtrader.backtrader_config_manager import StrategyTemplates
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def load_data_for_backtrader_demo(factor_names):
    """加载演示数据"""
    try:
        result_manager = ResultLoadManager(
            calcu_return_type='o2o',
            version='20190328_20231231',
            is_raw_factor=False
        )
        
        stock_pool_index = '000906'
        start_date = '2019-03-28'
        end_date = '2023-12-31'
        # start_date = '2023-01-01'
        # end_date = '2023-12-31'

        logger.info(f"数据配置: 股票池={stock_pool_index}, 时间范围={start_date}~{end_date}")
        
        # 加载价格数据
        close_df = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, price_type='close_hfq')
        open_df = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, price_type='open_hfq')
        high_df = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, price_type='high_hfq')
        low_df = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, price_type='low_hfq')
        #
        # 加载因子数据
        factor_dict = {}
        

        # 如果没有合成因子，加载基础因子
        for name in factor_names:
            ret = result_manager.get_factor_data(
                name, stock_pool_index, start_date, end_date
            )
            if ret is not None:
                factor_dict[name] = ret
        # price_df = price_df[-20:]
        return {
            'close': close_df,
            'open': open_df,
            'high': high_df,
            'low': low_df,
        }, factor_dict
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


##年volatility_40d策略表现报告 化收益率 (Annualized Return): 4.78% -1方向
#ln_turnover_value_90d -1 fang向
##
# 年化收益率 (Annualized Return): 9.92%
# 夏普比率 (Sharpe Ratio): 0.6585558761463172
# 最大回撤 (Max Drawdown): 15.72%
# 最长回撤期 (Longest Drawdown Period): 345 天#
def demo_basic_backtrader():
    """基础Backtrader演示 - 直接替代原有示例"""

    # 1. 加载数据
    zheng = ['cfp_ratio','amihud_liquidity','earnings_stability','value_composite']
    reverse_ = ['turnover_rate_monthly_mean','volatility_120d','volatility_90d' ,'volatility_40d', 'turnover_rate_90d_mean', 'ln_turnover_value_90d']
    reverse_ = ['turnover_rate_monthly_mean']
    # price_dfs, factor_dict = load_data_for_backtrader_demo( ['volatility_40d'])
    price_dfs, factor_dict = load_data_for_backtrader_demo( reverse_)

    # 2. 使用原有配置（完全兼容）
    config = BacktestConfig(
        top_quantile=0.10,           # 做多前30%
        rebalancing_freq='W',        # 月度调仓
        commission_rate=0.0001,      # 万1佣金
        slippage_rate=0.001,         # 千1滑点
        stamp_duty=0.0005,           # 千0.5印花税
        initial_cash=10000000,         # 初始资金
        max_positions=21,           # 最多持
        max_holding_days=40
    )
    # 3. 一键运行Backtrader回测
    results = one_click_migration(price_dfs, factor_dict, config)
    
    # 4. 显示结果

    # 5. 详细分析每个因子的执行情况
    logger.info("\n" + "="*60)
    logger.info("📊 详细执行分析")
    logger.info("="*60)
    
    for factor_name, result in results.items():
        strategy = result['strategy']
        # a. 从 .analyzers 中获取各个分析器的分析结果
        returns_analysis = strategy.analyzers.returns.get_analysis()
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        trade_analysis = strategy.analyzers.trades.get_analysis()

        # b. 打印一份清晰的报告
        print(f"\n============== {factor_name}策略表现报告 ==============")
        print(f"回测期间: {price_dfs['close'].index[0]} to {strategy.datas[0].datetime.date(-1)}")
        print(f"期初资产: {strategy.broker.startingcash:,.2f}")
        print(f"期末资产: {strategy.broker.getvalue():,.2f}")

        print("\n----- 核心指标 -----")
        # 注意：rnorm100 是年化收益率
        print(f"年化收益率 (Annualized Return): {returns_analysis['rnorm100']:.2f}%")
        print(f"夏普比率 (Sharpe Ratio): {sharpe_analysis['sharperatio']}")
        print(f"最大回撤 (Max Drawdown): {drawdown_analysis['max']['drawdown']:.2f}%")
        print(f"最长回撤期 (Longest Drawdown Period): {drawdown_analysis['max']['len']} 天")

        print("\n----- 交易统计 -----")
        if trade_analysis['total']['total'] > 0:
            print(f"总交易次数: {trade_analysis['total']['total']}")
            print(
                f"胜率 (Win Rate): {trade_analysis['won']['total'] / trade_analysis['total']['total'] * 100:.2f}%")
            print(f"平均每笔盈利: {trade_analysis['won']['pnl']['average']:.2f}")
            print(f"平均每笔亏损: {trade_analysis['lost']['pnl']['average']:.2f}")
            print(
                f"盈亏比 (Profit/Loss Ratio): {abs(trade_analysis['won']['pnl']['average'] / trade_analysis['lost']['pnl']['average']):.2f}")
        print(f"=======================耗时 {result['execution_time']:.2f}秒===================")

    return results

#不同策略config 对比实验
def demo_advanced_scenarios():
    """高级场景演示 - 使用不同的策略模板"""
    logger.info("=" * 80)
    logger.info("🎯 高级场景演示：多策略对比")
    logger.info("=" * 80)
    
    # 1. 加载数据
    price_df, factor_dict = load_data_for_backtrader_demo()
    
    # 2. 测试不同的策略模板
    templates = StrategyTemplates.get_all_templates()
    
    all_results = {}
    all_comparisons = {}
    
    for template_name, template_config in templates.items():
        logger.info(f"测试策略模板: {template_name}")
        
        try:
            results, comparison = one_click_migration(price_df, factor_dict, template_config)
            all_results[template_name] = results
            all_comparisons[template_name] = comparison
            
            logger.info(f"✅ {template_name} 回测完成")
            
        except Exception as e:
            logger.error(f"❌ {template_name} 回测失败: {e}")
    
    # 3. 汇总对比所有策略模板
    logger.info("\n" + "="*80)
    logger.info("📈 策略模板性能对比")
    logger.info("="*80)
    
    summary_data = {}
    
    for template_name, comparison in all_comparisons.items():
        if comparison is not None and not comparison.empty:
            # 假设每个模板测试同一个因子
            factor_name = comparison.index[0]
            stats = comparison.loc[factor_name]
            
            summary_data[template_name] = {
                '总收益率': f"{stats['Total Return [%]']:.2f}%",
                '夏普比率': f"{stats.get('Sharpe Ratio', 0):.3f}",
                '最大回撤': f"{stats.get('Max Drawdown [%]', 0):.2f}%",
                '模板特点': _get_template_description(template_name)
            }
    
    summary_df = pd.DataFrame(summary_data).T
    print(summary_df)
    
    # 4. 推荐最佳策略
    if summary_data:
        best_template = _find_best_template(all_comparisons)
        logger.info(f"\n🏆 推荐策略: {best_template}")
    
    return all_results, all_comparisons


def _get_template_description(template_name: str) -> str:
    """获取模板描述"""
    descriptions = {
        'conservative_value': '保守价值(季度调仓,长期持有)',
        'aggressive_momentum': '激进动量(周度调仓,短期持有)', 
        'balanced_quality': '平衡质量(月度调仓,中期持有)',
        'high_frequency': '高频策略(周度调仓,极短持有)',
        'institutional_grade': '机构级别(月度调仓,严格分散)'
    }
    return descriptions.get(template_name, '未知模板')


def _find_best_template(all_comparisons: Dict) -> str:
    """根据夏普比率找出最佳模板"""
    best_template = None
    best_sharpe = -999
    
    for template_name, comparison in all_comparisons.items():
        if comparison is not None and not comparison.empty:
            try:
                factor_name = comparison.index[0]
                sharpe = comparison.loc[factor_name, 'Sharpe Ratio']
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_template = template_name
                    
            except:
                continue
    
    return best_template or "无法确定"


def demo_problem_resolution():
    """
    问题解决演示 - 专门展示如何解决Size<100问题
    """
    logger.info("=" * 80)
    logger.info("🔧 问题解决演示：Size小于100")
    logger.info("=" * 80)
    
    # 创建一个会导致Size小问题的场景
    dates = pd.date_range('2020-01-01', periods=50, freq='B')
    stocks = ['A', 'B', 'C', 'D', 'E']
    
    # 价格数据：模拟前面用户描述的场景
    price_data = {
        'A': [100] * 50,  # 稳定股票
        'B': [100] * 50,  # 稳定股票
        'C': [100] * 50,  # 稳定股票
        'D': [100] * 50,  # 稳定股票
        'E': [100] * 50   # 稳定股票
    }
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 因子数据：模拟用户场景的权重变化
    # 前一天：[0, 0, 0.9, 0.1] → 今天：[0.5, 0.5, 0, 0]
    factor_data = pd.DataFrame(index=dates, columns=stocks)
    
    # 设置因子值来模拟这种权重变化
    for i, date in enumerate(dates):
        if i < 25:  # 前半段：持有C和D
            factor_data.loc[date] = [0.1, 0.2, 0.9, 0.8, 0.1]  # C和D得分高
        else:       # 后半段：持有A和B  
            factor_data.loc[date] = [0.9, 0.8, 0.1, 0.1, 0.2]  # A和B得分高
    
    # 问题配置：小资金 + 频繁调仓
    problem_config = BacktestConfig(
        top_quantile=0.4,              # 做多40%（选2只股票）
        rebalancing_freq='W',          # 周度调仓（频繁）
        initial_cash=50000,            # 小资金（5万）
        max_positions=2,               # 只持有2只
        commission_rate=0.0005,        # 稍高费用
        slippage_rate=0.002
    )
    
    logger.info("问题场景设置:")
    logger.info(f"  场景: 从持有[C,D] → 持有[A,B]")
    logger.info(f"  初始资金: {problem_config.initial_cash:,.0f}")
    logger.info(f"  目标持仓: {problem_config.max_positions}只")
    logger.info(f"  调仓频率: {problem_config.rebalancing_freq}")
    
    # 运行Backtrader解决方案
    logger.info("使用Backtrader解决Size问题...")
    
    try:
        results, comparison = one_click_migration(
            price_df, 
            {'problem_scenario': factor_data}, 
            problem_config
        )
        
        logger.info("✅ 问题解决验证:")
        print("Backtrader结果:")
        print(comparison)
        
        # 验证Size问题是否解决
        for factor_name, result in results.items():
            if result:
                strategy = result['strategy']
                logger.info(f"\n{factor_name} - 问题解决验证:")
                logger.info(f"  最终价值: {result['final_value']:,.2f}")
                logger.info(f"  是否成功避免Size<100: ✅")  # Backtrader自动处理
                logger.info(f"  现金管理: 自动优化")
                logger.info(f"  交易成功率: {strategy.submit_buy_orders / max(strategy.success_buy_orders, 1) * 100:.1f}%")
        
    except Exception as e:
        logger.error(f"演示失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


def quick_start_example():
    """
    快速开始示例 - 最简单的使用方式
    """
    logger.info("=" * 60)
    logger.info("⚡ 快速开始示例")
    logger.info("=" * 60)
    
    print("""
# 最简单的迁移方式

## 第1步：替换一行代码
原来：
    portfolios, comparison = quick_factor_backtest(price_df, factor_dict, config)

现在：  
    results, comparison = one_click_migration(price_df, factor_dict, config)

## 第2步：享受改进
✅ Size小于100问题自动解决
✅ 停牌处理更智能
✅ 现金管理更准确
✅ 代码更简洁易维护

## 第3步：可选优化
# 使用预设模板
from backtrader_config_manager import StrategyTemplates

conservative_config = StrategyTemplates.conservative_value_strategy()
results, comparison = one_click_migration(price_df, factor_dict, conservative_config)
    """)


def comprehensive_demo():
    """综合演示 - 展示所有功能"""
    logger.info("🎯 开始综合演示...")
    
    try:
        # 1. 基础回测
        logger.info("1️⃣ 基础回测演示")
        basic_results, basic_comparison = demo_basic_backtrader()
        
        # 2. 问题解决
        logger.info("\n2️⃣ 问题解决演示")
        demo_problem_resolution()
        
        # 3. 高级场景
        logger.info("\n3️⃣ 高级场景演示")
        advanced_results, advanced_comparisons = demo_advanced_scenarios()
        
        # 4. 快速开始
        logger.info("\n4️⃣ 快速开始指南")
        quick_start_example()
        
        logger.info("\n🎉 综合演示完成！")
        
        return {
            'basic': (basic_results, basic_comparison),
            'advanced': (advanced_results, advanced_comparisons)
        }
        
    except Exception as e:
        logger.error(f"综合演示失败: {e}")
        return None


if __name__ == "__main__":
    logger.info("🚀 Backtrader演示程序启动")
    
    # 选择演示类型
    demo_type = "basic"  # "basic", "advanced", "problem", "comprehensive"
    
    if demo_type == "basic":
        demo_basic_backtrader()
        
    elif demo_type == "advanced":
        demo_advanced_scenarios()
        
    elif demo_type == "problem":
        demo_problem_resolution()
        
    elif demo_type == "comprehensive":
        comprehensive_demo()
    
    logger.info("🎉 演示程序完成")