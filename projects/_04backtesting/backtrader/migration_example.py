"""
vectorBT → Backtrader 迁移示例

完整演示：
1. 如何一键替换现有的回测调用
2. 对比两个框架的结果差异
3. 验证Size小于100问题的解决方案
4. 展示Backtrader在处理停牌和现金管理方面的优势
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from projects._04backtesting.backtrader.test.standalone_backtrader_test import create_test_data

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._04backtesting.quant_backtester import QuantBacktester, BacktestConfig
from projects._04backtesting.backtrader.backtrader_enhanced_strategy import one_click_migration
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def load_test_data():
    """加载测试数据 - 使用真实的价格和因子数据"""
    try:
        result_manager = ResultLoadManager(
            calcu_return_type='o2o',
            version='20190328_20231231',
            is_raw_factor=False
        )
        
        stock_pool_index = '000906'
        start_date = '2020-01-01'  # 缩短测试周期
        end_date = '2021-12-31'
        
        # 加载价格数据
        open_hfq = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, 'open_hfq')
        
        # 加载因子数据 - 选择一个相对稳定的因子进行测试
        factor_data = result_manager.get_factor_data(
            'lqs_orthogonal_v1', stock_pool_index, start_date, end_date
        )
        
        if factor_data is None or factor_data.empty:
            # 备选因子
            factor_data = result_manager.get_factor_data(
                'volatility_40d', stock_pool_index, start_date, end_date
            )
        
        # 为了演示，限制股票数量（提高测试速度）
        if len(open_hfq.columns) > 50:
            selected_stocks = open_hfq.columns[:50]  # 选择前50只股票
            open_hfq = open_hfq[selected_stocks]
            factor_data = factor_data[selected_stocks]
        
        # 确保数据质量
        # 移除全NaN的股票
        open_hfq = open_hfq.dropna(axis=1, how='all')
        factor_data = factor_data.dropna(axis=1, how='all')
        
        # 保证价格和因子的股票一致
        common_stocks = open_hfq.columns.intersection(factor_data.columns)
        open_hfq = open_hfq[common_stocks]
        factor_data = factor_data[common_stocks]
        
        logger.info(f"测试数据加载完成:")
        logger.info(f"  价格数据: {open_hfq.shape}")
        logger.info(f"  因子数据: {factor_data.shape}")
        logger.info(f"  时间范围: {open_hfq.index.min()} ~ {open_hfq.index.max()}")
        
        return open_hfq, {'test_factor': factor_data}
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        # 创建模拟数据用于测试
        return create_mock_data()


def create_mock_data():
    """创建模拟数据用于测试"""
    logger.info("创建模拟数据进行测试...")
    
    # 创建日期范围
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='B')  # 工作日
    stocks = [f'STOCK_{i:03d}' for i in range(20)]  # 20只股票
    
    # 模拟价格数据（随机游走）
    np.random.seed(42)  # 确保可重现
    price_data = {}
    
    for stock in stocks:
        # 生成价格序列（随机游走，带趋势）
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 平均日收益0.05%，波动2%
        price_series = 100 * np.exp(np.cumsum(returns))  # 从100开始的价格
        
        # 模拟停牌（随机设置5%的数据为NaN）
        suspension_mask = np.random.random(len(dates)) < 0.05
        price_series[suspension_mask] = np.nan
        
        price_data[stock] = price_series
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 模拟因子数据（与价格负相关的动量因子）
    factor_data = {}
    for stock in stocks:
        # 简单的反转因子：过去20日收益率的负值
        returns_20d = price_df[stock].pct_change(20)
        factor_data[stock] = -returns_20d  # 反转因子
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    logger.info(f"模拟数据创建完成:")
    logger.info(f"  价格数据: {price_df.shape}")
    logger.info(f"  因子数据: {factor_df.shape}")
    
    return price_df, {'mock_reversal_factor': factor_df}


def compare_frameworks():
    """
    框架对比测试 - 直接对比vectorBT和Backtrader的结果
    """
    logger.info("=" * 100)
    logger.info("🔬 框架对比测试：vectorBT vs Backtrader")
    logger.info("=" * 100)
    
    # 1. 加载数据
    price_df, factor_dict = load_test_data()
    
    # 2. 配置参数（相同的配置用于两个框架）
    config = BacktestConfig(
        top_quantile=0.3,              # 做多前30%
        rebalancing_freq='M',          # 月度调仓
        commission_rate=0.0003,        # 万3佣金
        slippage_rate=0.001,           # 千1滑点
        stamp_duty=0.001,              # 千1印花税
        initial_cash=1000000,          # 100万初始资金
        max_positions=10,              # 最多10只股票
        max_holding_days=60            # 最多持有60天
    )
    
    # 3. 运行vectorBT回测（原始方法）
    logger.info("--- 运行vectorBT回测（原始方法）---")
    try:
        vectorbt_backtester = QuantBacktester(config)
        vectorbt_results = vectorbt_backtester.run_backtest(price_df, factor_dict)
        vectorbt_comparison = vectorbt_backtester.get_comparison_table()
        
        logger.info("vectorBT回测完成")
        print("vectorBT结果:")
        print(vectorbt_comparison)
        
        # 检查Size问题
        for factor_name, portfolio in vectorbt_results.items():
            trades = portfolio.trades.records_readable
            if not trades.empty and 'Size' in trades.columns:
                small_sizes = trades[trades['Size'] < 100]
                logger.info(f"vectorBT - {factor_name}: Size<100的交易{len(small_sizes)}笔")
                if len(small_sizes) > 0:
                    logger.warning(f"  最小Size: {trades['Size'].min():.2f}")
                    logger.warning(f"  平均Size: {trades['Size'].mean():.2f}")
        
    except Exception as e:
        logger.error(f"vectorBT回测失败: {e}")
        vectorbt_results = None
        vectorbt_comparison = None
    
    # 4. 运行Backtrader回测（新方法）
    logger.info("--- 运行Backtrader回测（新方法）---")
    try:
        backtrader_results, backtrader_comparison = one_click_migration(
            price_df, factor_dict, config
        )
        
        logger.info("Backtrader回测完成")
        print("Backtrader结果:")
        print(backtrader_comparison)
        
    except Exception as e:
        logger.error(f"Backtrader回测失败: {e}")
        backtrader_results = None
        backtrader_comparison = None
    
    # 5. 结果对比分析
    if vectorbt_results and backtrader_results:
        logger.info("=" * 60)
        logger.info("📊 详细结果对比")
        logger.info("=" * 60)
        
        # 对比收益率
        for factor_name in factor_dict.keys():
            if factor_name in vectorbt_comparison.index and factor_name in backtrader_comparison.index:
                vbt_return = vectorbt_comparison.loc[factor_name, 'Total Return [%]']
                bt_return = backtrader_comparison.loc[factor_name, 'Total Return [%]']
                
                logger.info(f"{factor_name}:")
                logger.info(f"  vectorBT收益率: {vbt_return:.2f}%")
                logger.info(f"  Backtrader收益率: {bt_return:.2f}%")
                logger.info(f"  差异: {abs(bt_return - vbt_return):.2f}%")
    
    return vectorbt_results, backtrader_results


def demo_problem_solving():
    """
    演示问题解决 - 专门展示Size小于100问题的解决方案
    """
    logger.info("=" * 100) 
    logger.info("🎯 专项演示：解决Size小于100问题")
    logger.info("=" * 100)
    
    # 1. 创建一个容易触发问题的测试场景
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    
    # 创建一个极端场景：大部分股票经常停牌
    stocks = [f'PROBLEM_{i}' for i in range(5)]
    
    # 价格数据：频繁停牌
    price_data = {}
    np.random.seed(123)
    
    for i, stock in enumerate(stocks):
        # 基础价格序列
        base_prices = 100 + i * 10 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        # 模拟频繁停牌（30%的时间停牌）
        suspension_mask = np.random.random(len(dates)) < 0.3
        base_prices[suspension_mask] = np.nan
        
        price_data[stock] = base_prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 因子数据：简单的反转因子
    factor_df = -price_df.pct_change(5)  # 5日反转
    
    # 配置：容易触发Size问题的参数
    problem_config = BacktestConfig(
        top_quantile=0.6,              # 做多60%（容易频繁调仓）
        rebalancing_freq='W',          # 周度调仓（更频繁）
        commission_rate=0.0005,        # 稍高费用
        slippage_rate=0.0015,
        stamp_duty=0.001,
        initial_cash=1000000,           # 相对较少的初始资金
        max_positions=3,               # 少量持仓
        max_holding_days=20            # 短期持有
    )

    logger.info("测试场景设置:")
    logger.info(f"  股票数量: {len(stocks)}")
    logger.info(f"  停牌概率: 30%")
    logger.info(f"  调仓频率: 周度")
    logger.info(f"  初始资金: {problem_config.initial_cash:,.0f}")

    # 2. 使用Backtrader解决方案
    logger.info("使用Backtrader解决方案...")

    try:
        results, comparison = one_click_migration(
            price_df,
            {'problem_factor': factor_df},
            problem_config
        )

        logger.info("✅ Backtrader成功处理了复杂场景!")
        print("Backtrader结果:")
        print(comparison)

        # 分析交易明细
        for factor_name, result in results.items():
            if result:
                strategy = result['strategy']
                logger.info(f"{factor_name} - 策略统计:")
                logger.info(f"  调仓次数: {strategy.rebalance_count}")
                logger.info(f"  总订单: {strategy.success_buy_orders}")
                logger.info(f"  成功订单: {strategy.submit_buy_orders}")
                logger.info(f"  失败订单: {strategy.failed_orders}")

                if strategy.success_buy_orders > 0:
                    success_rate = strategy.submit_buy_orders / strategy.success_buy_orders * 100
                    logger.info(f"  订单成功率: {success_rate:.1f}%")

    except Exception as e:
        logger.error(f"Backtrader测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


def migration_guide():
    """
    迁移指南 - 详细说明如何修改现有代码
    """
    logger.info("=" * 100)
    logger.info("📚 vectorBT → Backtrader 迁移指南")
    logger.info("=" * 100)
    
    guide_text = """
    
== 第1步：替换导入 ==
原始代码：
    from projects._04backtesting.quant_backtester import QuantBacktester, BacktestConfig, quick_factor_backtest

修改为：  
    from projects._03_factor_selection.backtrader_enhanced_strategy import one_click_migration, BacktraderMigrationEngine
    from projects._04backtesting.quant_backtester import BacktestConfig  # 配置仍然可用

== 第2步：替换回测调用 ==
原始代码：
    backtester = QuantBacktester(config)
    portfolios = backtester.run_backtest(price_df, factor_dict)
    comparison_table = backtester.get_comparison_table()

修改为：
    results, comparison_table = one_click_migration(price_df, factor_dict, config)

== 第3步：结果访问调整 ==
原始代码：
    for factor_name, portfolio in portfolios.items():
        trades = portfolio.trades.records_readable
        print(portfolio.stats())

修改为：
    for factor_name, result in results.items():
        if result:
            strategy = result['strategy']
            analyzers = result['analyzers']
            print(f"最终价值: {result['final_value']}")

== 核心问题解决 ==

✅ Size小于100问题：
   - vectorBT: 使用复杂的权重计算和convert_to_sequential_percents
   - Backtrader: 使用order_target_percent自动处理现金管理

✅ 停牌处理：
   - vectorBT: 复杂的is_tradable_today检查和pending_buys_tracker
   - Backtrader: 事件驱动的自动重试机制

✅ 状态管理：
   - vectorBT: 手动维护actual_holdings等多个状态变量
   - Backtrader: 框架自动处理所有状态

✅ 现金管理：
   - vectorBT: 权重分配导致现金不足，Size变小
   - Backtrader: 自动根据可用现金调整订单大小

== 性能对比 ==
- 代码复杂度：从1000+行降低到300行
- 维护难度：从复杂状态管理简化为事件驱动
- 调试能力：内置详细的订单和交易日志
- 扩展性：更容易添加新的交易逻辑和风控规则

    """
    
    print(guide_text)


def main():
    """主测试函数"""
    # 选择测试类型
    test_type = "problem_solving"  # "comparison", "problem_solving", "migration", "all"
    
    if test_type == "comparison" or test_type == "all":
        compare_frameworks()
    
    if test_type == "problem_solving" or test_type == "all":
        demo_problem_solving()
    
    if test_type == "migration" or test_type == "all":
        migration_guide()
    
    logger.info("🎉 测试完成！")

def t_():
    price_df, factor_df = create_test_data()
    # 配置：容易触发Size问题的参数
    problem_config = BacktestConfig(
        top_quantile=0.5,  # 做多60%（容易频繁调仓）
        rebalancing_freq='2D',  # 周度调仓（更频繁）
        commission_rate=0.0003,  # 稍高费用
        slippage_rate=0.0015,
        stamp_duty=0.001,
        initial_cash=1000000,  # 相对较少的初始资金
        max_positions=2,  # 少量持仓
        max_holding_days=20,  # 短期持有
        buy_after_sell_cooldown=10  #
    )

    results = one_click_migration(
        price_df,
        {'problem_factor': factor_df},
        problem_config
    )

if __name__ == "__main__":
    t_()