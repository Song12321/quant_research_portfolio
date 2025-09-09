"""
测试修复后的Backtrader代码

专门验证修复效果
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._04backtesting.quant_backtester import BacktestConfig
from projects._04backtesting.backtrader.test.backtrader_fixed import fixed_backtrader_test
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def create_simple_test_data():
    """创建简单的测试数据"""
    logger.info("创建简单测试数据...")
    
    # 创建30个交易日的数据
    dates = pd.date_range('2020-01-01', periods=60, freq='B')  # 工作日
    stocks = [f'TEST_{i:02d}' for i in range(10)]  # 10只股票
    
    np.random.seed(42)  # 保证结果可重现
    
    # 价格数据：随机游走
    price_data = {}
    for stock in stocks:
        returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
        prices = 100 * np.exp(np.cumsum(returns))  # 累积价格
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 因子数据：简单的动量因子
    factor_data = {}
    for stock in stocks:
        # 5日收益率作为因子
        returns_5d = price_df[stock].pct_change(5)
        factor_data[stock] = returns_5d
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    logger.info(f"测试数据创建完成: 价格{price_df.shape}, 因子{factor_df.shape}")
    return price_df, {'momentum_5d': factor_df}


def test_simple_data():
    """使用简单数据测试"""
    logger.info("=" * 60)
    logger.info("🧪 简单数据测试")
    logger.info("=" * 60)
    
    # 创建测试数据
    price_df, factor_dict = create_simple_test_data()
    
    # 配置参数
    config = BacktestConfig(
        top_quantile=0.3,              # 做多30%（3只股票）
        rebalancing_freq='M',          # 月度调仓
        initial_cash=100000,           # 10万资金
        max_positions=5,               # 最多5只
        max_holding_days=30
    )
    
    logger.info("开始简单数据回测...")
    
    # 运行修复版回测
    results, comparison = fixed_backtrader_test(price_df, factor_dict, config)
    
    # 显示结果
    logger.info("简单数据回测结果:")
    print(comparison)
    
    # 检查是否有交易
    for factor_name, result in results.items():
        if result:
            strategy = result['strategy']
            logger.info(f"{factor_name}: 调仓{strategy.rebalance_count}次, 最终价值{result['final_value']:,.2f}")
    
    return results, comparison


def test_real_data():
    """使用真实数据测试（小规模）"""
    logger.info("=" * 60)
    logger.info("🏭 真实数据测试（小规模）")
    logger.info("=" * 60)
    
    try:
        result_manager = ResultLoadManager(
            calcu_return_type='o2o',
            version='20190328_20231231',
            is_raw_factor=False
        )
        
        # 使用小范围数据
        start_date = '2021-01-01'
        end_date = '2021-12-31'
        stock_pool = '000906'
        
        # 加载数据
        open_hfq =  result_manager.get_price_data_by_type(stock_pool, start_date, end_date, 'open_hfq')
        factor_data = result_manager.get_factor_data(
            'lqs_orthogonal_v1', stock_pool, start_date, end_date
        )
        
        if factor_data is None:
            factor_data = result_manager.get_factor_data(
                'volatility_40d', stock_pool, start_date, end_date
            )
        
        # 限制股票数量（测试用）
        selected_stocks = open_hfq.columns[:30]  # 只选30只股票
        open_hfq = open_hfq[selected_stocks]
        factor_data = factor_data[selected_stocks]
        
        logger.info(f"真实数据: 价格{open_hfq.shape}, 因子{factor_data.shape}")
        
        # 配置
        config = BacktestConfig(
            top_quantile=0.2,
            rebalancing_freq='M',
            initial_cash=500000,
            max_positions=8,
            max_holding_days=45
        )
        
        # 运行回测
        results, comparison = fixed_backtrader_test(
            open_hfq,
            {'test_factor': factor_data}, 
            config
        )
        
        logger.info("真实数据回测结果:")
        print(comparison)
        
        return results, comparison
        
    except Exception as e:
        logger.error(f"真实数据测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def comprehensive_test():
    """综合测试"""
    logger.info("🚀 开始综合测试...")
    
    # 1. 简单数据测试
    logger.info("1️⃣ 简单数据测试")
    simple_results, simple_comparison = test_simple_data()
    
    # 验证简单测试结果
    if simple_results:
        success_count = sum(1 for r in simple_results.values() if r is not None)
        logger.info(f"✅ 简单测试: {success_count}/{len(simple_results)}成功")
    else:
        logger.error("❌ 简单测试失败")
        return
    
    # 2. 真实数据测试  
    logger.info("\n2️⃣ 真实数据测试")
    real_results, real_comparison = test_real_data()
    
    if real_results:
        success_count = sum(1 for r in real_results.values() if r is not None)
        logger.info(f"✅ 真实数据测试: {success_count}/{len(real_results)}成功")
    else:
        logger.error("❌ 真实数据测试失败")
    
    logger.info("\n🎉 综合测试完成!")
    
    return {
        'simple': (simple_results, simple_comparison),
        'real': (real_results, real_comparison)
    }


if __name__ == "__main__":
    logger.info("🧪 开始测试修复后的Backtrader")
    
    # 运行测试
    test_type = "comprehensive"  # "simple", "real", "comprehensive"
    
    if test_type == "simple":
        test_simple_data()
    elif test_type == "real":
        test_real_data()
    elif test_type == "comprehensive":
        comprehensive_test()
    
    logger.info("🎉 测试完成")