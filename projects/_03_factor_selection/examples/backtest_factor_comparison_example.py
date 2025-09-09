"""
因子对比回测示例

展示如何使用QuantBacktester进行"苹果vs苹果"的因子策略对比
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._04backtesting.quant_backtester import (
    QuantBacktester, 
    BacktestConfig,
    quick_factor_backtest
)
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def load_example_data():
    """
    加载示例数据 - 使用真实的价格和因子数据
    Returns:
        Tuple: (价格数据, 因子数据字典)
    """
    try:
        result_manager = ResultLoadManager(
            calcu_return_type='o2o',
            version='20190328_20231231',
            is_raw_factor=False
        )
        
        stock_pool_index = '000906'
        start_date = '2019-03-28'
        end_date = '2023-12-31'
        
        logger.info(f"数据配置: 股票池={stock_pool_index}, 时间范围={start_date}~{end_date}")
        
        # 1. 加载真实价格数据（后复权收盘价）
        open_hfq_df = result_manager.get_price_data_by_type(stock_pool_index, start_date, end_date, 'open_hfq')

        # 2. 加载日收益率数据（供参考，回测器内部会用价格数据计算收益）
        logger.info("正在加载收益率数据...")
        return_1d_df = result_manager.get_o2o_return_data(stock_pool_index, start_date, end_date, period_days=1)
        logger.info(f"日收益率数据加载成功: {return_1d_df.shape}")

        # 3. 加载因子数据
        logger.info("正在加载因子数据...")

        # 加载冠军因子
        champion_factor = result_manager.get_factor_data(
            'volatility_40d', stock_pool_index, start_date, end_date
        )
        
        # 加载合成因子
        composite_factor = result_manager.get_factor_data(
            'lqs_orthogonal_v1', stock_pool_index, start_date, end_date
        )
        
        # 4. 数据质量检查和汇总
        factor_dict = {}
        #
        # if champion_factor is not None and not champion_factor.empty:
        #     factor_dict['volatility_40d (冠军因子)'] = champion_factor
        #     logger.info(f"冠军因子加载成功: {champion_factor.shape}")
        # else:
        #     logger.warning("冠军因子 volatility_40d 加载失败或为空")
        
        if composite_factor is not None and not composite_factor.empty:
            factor_dict['lqs_orthogonal_v1 (合成因子)'] = composite_factor
            logger.info(f"合成因子加载成功: {composite_factor.shape}")
        else:
            logger.warning("合成因子 lqs_orthogonal_v1 加载失败或为空")
        
        # 5. 验证数据一致性
        if not factor_dict:
            raise ValueError("未能加载到有效的因子数据")
        
        # 检查时间对齐
        price_dates = set(open_hfq_df.index)
        for factor_name, factor_data in factor_dict.items():
            factor_dates = set(factor_data.index)
            common_dates = price_dates.intersection(factor_dates)
            logger.info(f"{factor_name} 与价格数据共同日期: {len(common_dates)}/{len(price_dates)}")
            
        # 检查股票对齐
        price_stocks = set(open_hfq_df.columns)
        for factor_name, factor_data in factor_dict.items():
            factor_stocks = set(factor_data.columns) 
            common_stocks = price_stocks.intersection(factor_stocks)
            logger.info(f"{factor_name} 与价格数据共同股票: {len(common_stocks)}/{len(price_stocks)}")
        
        # 6. 数据摘要
        logger.info("数据加载完成摘要:")
        logger.info(f"  📈 价格数据: {open_hfq_df.shape} (日期: {open_hfq_df.index.min()} ~ {open_hfq_df.index.max()})")
        logger.info(f"  🎯 有效因子数量: {len(factor_dict)}")
        
        for name, df in factor_dict.items():
            data_coverage = (1 - df.isnull().sum().sum() / df.size) * 100
            logger.info(f"    - {name}: {df.shape}, 数据覆盖率: {data_coverage:.1f}%")
        

        return open_hfq_df, factor_dict
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise


def example_basic_comparison():
    """基础对比回测示例"""
    logger.info("示例1: 基础因子对比回测")

    try:
        # 1. 加载数据
        price_df, factor_dict = load_example_data()
        # ##filter
        # price_df = price_df[['600373.SH','601318.SH','600062.SH','600667.SH','000895.SZ','600967.SH']]
        # price_df = price_df[price_df.index >=pd.to_datetime('2020-12-21')]
        # factor_dict = { name:item[['600373.SH','601318.SH','600062.SH','600667.SH','000895.SZ','600967.SH']] for name,item in factor_dict.items()}
        # factor_dict = { name:item[item.index>=pd.to_datetime('2020-12-21')] for name,item in factor_dict.items()}
        
        # 2. 配置回测参数
        config = BacktestConfig(
            top_quantile=0.30,           # 做多前20%
            rebalancing_freq='M',       # 月度调仓：M 周：W 季末：Q
            commission_rate=0.0001,     # 万3佣金
            slippage_rate=0.001,        # 千1滑点
            stamp_duty=0.0005,           # 千1印花税
            initial_cash=300000,       # 300万初始资金
            max_positions=30,            # 最多持30只股票
            max_holding_days=60

        )
        
        # 3. 使用便捷函数快速回测
        portfolios, comparison_table = quick_factor_backtest(
            price_df, factor_dict, config
        )
        
        # 4. 显示对比结果
        logger.info("因子对比结果:")
        print("\n" + "="*80)
        print("因子策略业绩对比表")
        print("="*80)
        print(comparison_table.round(4))
        
        # 5. 生成图表
        backtester = QuantBacktester(config)
        backtester.portfolios = portfolios
        backtester.plot_cumulative_returns(figsize=(15, 8))
        
        return portfolios, comparison_table
        
    except Exception as e:
        raise ValueError(f"基础对比示例运行失败: {e}")


def example_advanced_analysis():
    """高级分析示例"""
    logger.info("=" * 60)
    logger.info("示例2: 高级因子分析")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        price_df, factor_dict = load_example_data()
        
        # 使用更严格的配置
        config = BacktestConfig(
            top_quantile=0.15,          # 做多前15% (更精选)
            rebalancing_freq='M',       # 月度调仓
            commission_rate=0.0001,     # 稍高的交易成本
            slippage_rate=0.0015,
            stamp_duty=0.001,
            initial_cash=5000000,       # 500万资金
            max_positions=25,           # 更集中持仓
            max_weight_per_stock=0.08   # 单股最大8%
        )
        
        # 创建回测器
        backtester = QuantBacktester(config)
        
        # 运行回测
        portfolios = backtester.run_backtest(price_df, factor_dict)
        
        # 生成完整对比表
        detailed_metrics = [
            'Total Return [%]',
            'Sharpe Ratio',
            'Calmar Ratio',
            'Max Drawdown [%]',
            'Win Rate [%]',
            'Profit Factor',
            'Total Trades'
        ]
        comparison_table = backtester.get_comparison_table(detailed_metrics)
        
        logger.info("详细对比结果:")
        print("\n" + "="*100)
        print("详细策略业绩分析表")
        print("="*100)
        print(comparison_table.round(4))
        
        # 生成图表分析
        backtester.plot_cumulative_returns(figsize=(16, 10))
        backtester.plot_drawdown_analysis(figsize=(16, 12))
        
        # 生成完整报告
        report_path = backtester.generate_full_report("backtest_reports")
        logger.info(f"完整报告已生成: {report_path}")
        
        return portfolios, comparison_table
        
    except Exception as e:
        logger.error(f"高级分析示例运行失败: {e}")
        return None, None


def example_sensitivity_analysis():
    """敏感性分析示例"""
    logger.info("=" * 60)
    logger.info("示例3: 参数敏感性分析")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        price_df, factor_dict = load_example_data()
        
        # 如果因子太多，只选择前2个进行敏感性分析
        if len(factor_dict) > 2:
            factor_dict = dict(list(factor_dict.items())[:2])
        
        # 测试不同的分位数阈值
        quantile_tests = [0.1, 0.15, 0.2, 0.25, 0.3]
        results_summary = []
        
        for quantile in quantile_tests:
            logger.info(f"测试分位数阈值: {quantile:.1%}")
            
            config = BacktestConfig(
                top_quantile=quantile,
                rebalancing_freq='M',
                commission_rate=0.0003,
                slippage_rate=0.001,
                initial_cash=1000000
            )
            
            portfolios, comparison = quick_factor_backtest(price_df, factor_dict, config)
            
            # 记录每个因子的关键指标
            for factor_name in factor_dict.keys():
                if factor_name in comparison.index:
                    stats = comparison.loc[factor_name]
                    results_summary.append({
                        '分位数阈值': f"{quantile:.1%}",
                        '因子名称': factor_name,
                        '年化收益': f"{stats['Annual Return [%]']:.2f}%",
                        '夏普比率': f"{stats['Sharpe Ratio']:.3f}",
                        '最大回撤': f"{stats['Max Drawdown [%]']:.2f}%"
                    })
        
        # 汇总敏感性分析结果
        summary_df = pd.DataFrame(results_summary)
        
        logger.info("敏感性分析结果:")
        print("\n" + "="*80)
        print("分位数阈值敏感性分析")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        return summary_df
        
    except Exception as e:
        logger.error(f"敏感性分析示例运行失败: {e}")
        return None


def example_production_ready():
    """生产就绪示例 - 完整的实盘级回测"""
    logger.info("=" * 60)
    logger.info("示例4: 生产就绪回测 (实盘级配置)")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        price_df, factor_dict = load_example_data()
        
        # 实盘级配置
        production_config = BacktestConfig(
            # 策略参数
            top_quantile=0.2,
            rebalancing_freq='M',
            max_positions=50,
            max_weight_per_stock=0.05,  # 单股最大5%
            
            # 真实交易成本
            commission_rate=0.0003,     # 万3佣金
            slippage_rate=0.002,        # 千2滑点 (更保守)
            stamp_duty=0.001,           # 千1印花税
            min_commission=5.0,         # 5元最低佣金
            
            # 资金配置
            initial_cash=10000000,      # 1000万资金
            
            # 数据质量控制
            min_data_coverage=0.85,     # 85%数据覆盖率
            max_missing_consecutive_days=3
        )
        
        # 创建生产级回测器
        backtester = QuantBacktester(production_config)
        
        # 运行回测
        logger.info("开始生产级回测...")
        portfolios = backtester.run_backtest(price_df, factor_dict)
        
        # 生成完整分析
        comparison_table = backtester.get_comparison_table()
        
        logger.info("生产级回测结果:")
        print("\n" + "="*100)
        print("生产就绪策略回测结果")
        print("="*100)
        print(comparison_table.round(4))
        
        # 风险指标特别关注
        risk_focused_metrics = [
            'Max Drawdown [%]',
            'Avg Drawdown Duration', 
            'Annual Volatility [%]',
            'Sharpe Ratio',
            'Calmar Ratio'
        ]
        
        risk_table = backtester.get_comparison_table(risk_focused_metrics)
        
        print("\n" + "="*60)
        print("风险指标重点分析")
        print("="*60)
        print(risk_table.round(4))
        
        # 生成完整图表和报告
        backtester.plot_cumulative_returns(figsize=(18, 10))
        backtester.plot_drawdown_analysis(figsize=(18, 12))
        
        # 保存生产级报告
        report_path = backtester.generate_full_report("production_backtest_reports")
        logger.info(f"生产级报告已保存: {report_path}")
        
        # 结果评估
        logger.info("\n" + "="*60)
        logger.info("策略评估建议")
        logger.info("="*60)
        
        best_factor = comparison_table['Sharpe Ratio'].idxmax()
        best_sharpe = comparison_table.loc[best_factor, 'Sharpe Ratio']
        best_return = comparison_table.loc[best_factor, 'Annual Return [%]']
        best_drawdown = comparison_table.loc[best_factor, 'Max Drawdown [%]']
        
        logger.info(f"📊 最佳夏普比率策略: {best_factor}")
        logger.info(f"   夏普比率: {best_sharpe:.3f}")
        logger.info(f"   年化收益: {best_return:.2f}%")
        logger.info(f"   最大回撤: {best_drawdown:.2f}%")
        
        if best_sharpe > 1.0 and best_drawdown < 15.0:
            logger.info("✅ 策略表现优秀，建议进入实盘测试")
        elif best_sharpe > 0.5 and best_drawdown < 25.0:
            logger.info("⚠️ 策略表现一般，建议优化后再测试")
        else:
            logger.info("❌ 策略表现不佳，需要重新设计")
        
        return portfolios, comparison_table, report_path
        
    except Exception as e:
        logger.error(f"生产就绪示例运行失败: {e}")
        return None, None, None


if __name__ == "__main__":
    logger.info("🚀 因子对比回测示例程序开始")
    
    # 运行不同的示例
    example_choice = 1  # 1=基础对比, 2=高级分析, 3=敏感性分析, 4=生产级
    
    if example_choice == 1:
        portfolios, comparison = example_basic_comparison()
        
    elif example_choice == 2:
        portfolios, comparison = example_advanced_analysis()
        
    elif example_choice == 3:
        sensitivity_results = example_sensitivity_analysis()
        
    elif example_choice == 4:
        portfolios, comparison, report_path = example_production_ready()
    
    else:
        logger.info("运行所有示例...")
        example_basic_comparison()
        # example_advanced_analysis()
        # sensitivity_results = example_sensitivity_analysis()
        # example_production_ready()
    
    logger.info("🎉 因子对比回测示例程序完成")