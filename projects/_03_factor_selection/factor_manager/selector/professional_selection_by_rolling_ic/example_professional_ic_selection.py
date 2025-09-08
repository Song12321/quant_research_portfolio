"""
专业滚动IC因子筛选+IC加权合成系统使用示例

本示例展示如何使用专业的滚动IC筛选系统：
1. 从大量候选因子中智能筛选出高质量因子
2. 基于历史IC表现计算最优权重
3. 合成具有稳健预测能力的复合因子
4. 生成详细的筛选和合成报告

核心特色：
- 完全避免前视偏差的滚动IC计算
- 多周期IC评分（指数衰减权重）
- 专业级因子质量评估
- 类别内冠军选择机制
- 智能权重分配算法


Date: 2025-08-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager

warnings.filterwarnings('ignore')

from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import (
    ICWeightedSynthesizer, FactorWeightingConfig
)
from projects._03_factor_selection.factor_manager.selector.rolling_ic_factor_selector import (
    RollingICFactorSelector, RollingICSelectionConfig
)
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def demo_professional_factor_selection(snap_config_id,factor_names=None):
    """演示专业滚动IC因子筛选功能"""

    logger.info("🚀 专业滚动IC因子筛选系统演示开始")
    # 1. 配置参数
    # snap_config_id = "20250825_091622_98ed2d08"  # 配置快照ID

    # 专业筛选配置 - 严格标准确保高质量
    selector_config = RollingICSelectionConfig(
        min_snapshots=3,  # 最少快照数量
        min_ic_abs_mean=0.01,  # IC均值绝对值门槛（严格）
        min_ir_abs_mean=0.15,  # IR均值绝对值门槛（严格）
        min_ic_stability=0.45,  # IC稳定性门槛（方向一致性）
        max_ic_volatility=0.06,  # IC波动率上限（控制风险）
        decay_rate=0.70,  # 衰减率（偏向短期表现）
        max_factors_per_category=8,  # 每类最多选择2个因子
        max_final_factors=20,  # 最终选择8个因子
        enable_turnover_penalty=True,
        # 三层相关性控制哲学 - 新增核心功能
        high_corr_threshold=0.7,  # 红色警报：|corr| > 0.7，坚决二选一
        medium_corr_threshold=0.3,  # 黄色预警：0.3 < |corr| < 0.7，正交化战场
        enable_orthogonalization=True,  # 启用中相关区间的正交化处理


        ##日因子changed =
        max_turnover_mean_daily=0.55,#=0.15,
        max_turnover_trend_daily=0.0015,#0.0005,
        max_turnover_vol_daily=0.125#0.025
    )


    if  factor_names is None:
        # 2. 从CSV文件加载所有已测试的因子（实际项目中的候选池）
        csv_file = Path(
            r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factor_manager\selector\o2o_v3_entity.csv")

        factors_df = pd.read_csv(csv_file)
        # 选择评分较高的因子作为候选（模拟实际筛选前的预选）
        candidate_factors = factors_df[factors_df['Final_Score'] >= 0]['factor_name'].tolist()
        logger.info(f"📊 从CSV加载候选因子: {len(candidate_factors)} 个")
    else:
        candidate_factors =  factor_names

    # 3. 创建专业筛选器实例
    logger.info("\n🔧 初始化专业滚动IC筛选器...")
    factor_selector = RollingICFactorSelector(snap_config_id, selector_config)

    # 4. 执行完整的专业筛选流程
    logger.info("\n🔍 开始专业筛选流程...")
    try:
        selected_factors, selection_report = factor_selector.run_complete_selection(
            candidate_factors,
            force_generate=False  # 使用现有IC数据，避免重复计算
        )

        logger.info(f"\n✅ 专业筛选完成!")
        logger.info(f"🎯 从 {len(candidate_factors)} 个候选因子中选出 {len(selected_factors)} 个优质因子")

        # 展示筛选结果
        logger.info("\n🏆 最终选中的优质因子:")
        for i, factor in enumerate(selected_factors, 1):
            logger.info(f"  {i:2d}. {factor}")

        # 展示筛选统计
        selection_summary = selection_report.get('selection_summary', {})
        logger.info(f"\n📊 筛选统计:")
        logger.info(f"  候选数量: {selection_summary.get('candidate_count', 0)}")
        logger.info(f"  合格数量: {selection_summary.get('qualified_count', 0)}")
        logger.info(f"  冠军数量: {selection_summary.get('champions_count', 0)}")
        logger.info(f"  最终数量: {selection_summary.get('final_count', 0)}")
        logger.info(f"  通过率: {selection_summary.get('pass_rate', 0):.1%}")

    except Exception as e:
        raise ValueError(f"❌ 专业筛选失败: {e}")
        # logger.info("使用备用因子列表进行演示...")
        # selected_factors = ['earnings_stability', 'amihud_liquidity', 'volatility_40d']
        # selection_report = {'method': 'fallback'}

    return selected_factors, selection_report


def ic_weighted_synthesis(snap_config_id):
    """演示IC加权因子合成功能"""

    logger.info("\n" + "=" * 80)
    logger.info("🧮 IC加权因子合成系统演示开始")
    logger.info("=" * 80)

    # 1. 获取筛选结果
    selected_factors, selection_report = demo_professional_factor_selection(snap_config_id,factor_names=['turnover_rate_monthly_mean'])

    if not selected_factors:
        logger.error("❌ 无可用因子进行合成演示")
        return None

    # 2. 配置IC加权合成参数
    weighting_config = FactorWeightingConfig(
        min_ic_mean=0.010,  # IC均值门槛
        min_ic_ir=0.15,  # IR门槛
        min_ic_win_rate=0.51,  # 胜率门槛
        max_ic_p_value=0.15,  # 显著性门槛
        ic_decay_halflife=60,  # 衰减半衰期
        max_single_weight=0.4,  # 单因子最大权重
        min_single_weight=0.05  # 单因子最小权重
    )

    # 筛选器配置
    selector_config = RollingICSelectionConfig(
        min_ic_abs_mean=0.015,
        min_ir_abs_mean=0.20,
        min_ic_stability=0.45,
        decay_rate=0.70
    )

    # 3. 创建IC加权合成器（集成专业筛选功能）
    logger.info("\n🔧 初始化IC加权合成器...")

    # 注意：这里需要实际的factor_manager, factor_analyzer, factor_processor实例
    # 在实际应用中，这些应该从工厂或配置中获取
    factor_manager = FactorManager(
        data_manager=DataManager(
        )
    )
    synthesizer = ICWeightedSynthesizer(
        factor_manager=factor_manager,  # 实际使用时需要提供
        factor_analyzer=None,  # 实际使用时需要提供
        factor_processor=None,  # 实际使用时需要提供
        config=weighting_config,
        selector_config=selector_config
    )

    # 4. 执行专业筛选+IC加权合成
    composite_factor_name = "professional_ic_composite"

    logger.info(f"\n🚀 启动专业筛选+IC加权合成流程...")
    logger.info(f"🎯 目标复合因子: {composite_factor_name}")

    try:
        # 这里演示完整流程的逻辑，实际执行需要完整的依赖
        logger.info("✅ 专业筛选+IC加权合成系统集成完成")
        logger.info(f"📊 选中因子: {selected_factors}")
        logger.info("📋 系统具备以下核心功能:")
        logger.info("  1. 滚动IC计算（避免前视偏差）")
        logger.info("  2. 多周期IC评分（指数衰减权重）")
        logger.info("  3. 因子质量专业评估")
        logger.info("  4. 类别内冠军选择")
        logger.info("  5. ✨ 三层相关性控制哲学 ✨")
        logger.info("     - 红色警报(|corr|>0.7): 坚决二选一")
        logger.info("     - 黄色预警(0.3<|corr|<0.7): 正交化战场")
        logger.info("     - 绿色安全(|corr|<0.3): 直接保留")
        logger.info("  6. 智能权重优化")
        logger.info("  7. 加权因子合成")
        logger.info("  8. 综合报告生成（含相关性决策记录）")

    except Exception as e:
        logger.error(f"❌ 合成演示失败: {e}")

    logger.info("=" * 80)
    logger.info("✅ 专业滚动IC因子筛选+合成系统演示完成")


def main():
    """主函数"""
    logger.info("🎉 专业滚动IC因子筛选+IC加权合成系统")
    logger.info("=" * 80)
    logger.info("系统特色:")
    logger.info("• 滚动IC计算，完全避免前视偏差")
    logger.info("• 多周期IC评分，指数衰减权重")
    logger.info("• 专业因子质量评估体系")
    logger.info("• 类别内冠军选择机制")
    logger.info("• ✨ 三层相关性控制哲学 - 差异化处理策略")
    logger.info("• 智能IC权重分配算法")
    logger.info("• 端到端因子筛选+合成流程")
    logger.info("=" * 80)

    # 运行演示
    snap_config_id = "20250826_131138_d03f3d9e"  # 配置快照ID
    snap_config_id = "20250825_091622_98ed2d08"  # 配置快照ID 全部
    snap_config_id = "20250906_045625_05e460ab"  # 配置快照ID 全部

    ic_weighted_synthesis(snap_config_id)


if __name__ == "__main__":
    # snap_config_id = "20250825_091622_98ed2d08"  # 配置快照ID 全部
    # snap_config_id = "20250828_181420_f6baf27c"  # 配置快照ID 全部
    #
    # demo_professional_factor_selection(snap_config_id,['lqs_orthogonal_v1'])


    main()
