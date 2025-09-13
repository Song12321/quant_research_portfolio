"""
IC加权因子合成器 - 专业级因子合成引擎

核心功能：
1. 基于历史IC表现的智能权重分配
2. 多维度因子筛选机制
3. 动态权重调整
4. 风险控制和稳健性检验

设计理念：
- 以实盘盈利为终极目标
- 平衡因子预测能力与稳定性
- 防止过拟合，注重泛化能力
"""

import pandas as pd
from numpy.linalg import LinAlgError
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from projects._03_factor_selection.config_manager.factor_direction_config import get_new_factor_direction
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.selector.factor_selector import FactorSelector
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.factor_manager.ic_manager.rolling_ic_manager import (
    ICCalculationConfig, run_cal_and_save_rolling_ic_by_snapshot_config_id
)
from projects._03_factor_selection.factor_manager.selector.rolling_ic_factor_selector import (
    RollingICFactorSelector, RollingICSelectionConfig
)
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import ConfigSnapshotManager
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class FactorWeightingConfig:
    """因子权重配置"""
    # IC筛选标准
    min_ic_mean: float = 0.015  # 最小IC均值阈值
    min_ic_ir: float = 0.183  # 最小IC信息比率阈值
    min_ic_win_rate: float = 0.52  # 最小IC胜率阈值
    max_ic_p_value: float = 0.10  # 最大IC显著性p值

    # 权重计算参数
    ic_decay_halflife: int = 60  # IC权重衰减半衰期(天)
    max_single_weight: float = 0.5  # 单个因子最大权重
    min_single_weight: float = 0.05  # 单个因子最小权重

    # 风险控制
    max_factors_count: int = 8  # 最大因子数量
    correlation_threshold: float = 0.70  # 因子间相关性阈值

    # 回看期设置
    lookback_periods: List[str] = None  # IC计算周期
    # 正交化流程的稳健性控制参数
    min_orthogonalization_obs: int = 30  # 正交化回归时要求的最小样本量
    orthogonalization_x_std_eps: float = 1e-6  # 判断基准因子标准差是否过小的阈值
    #使用场景:_adjust_ic_stats_by_r_squared 用于调整正交化 ic
    main_evaluation_period = '5'

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = ['5d', '21d']


@dataclass
class FactorQualityReport:
    """因子质量报告"""
    factor_name: str
    ic_stats: Dict[str, float]
    weight: float
    selection_reason: str
    risk_flags: List[str]


class ICWeightCalculator:
    """IC权重计算器 - 核心算法引擎"""

    def __init__(self, config: FactorWeightingConfig):
        self.config = config

    def calculate_ic_based_weights(
            self,
            factor_ic_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        基于IC统计的智能权重分配
        
        Args:
            factor_ic_stats: {factor_name: {period: ic_stats_dict}}
            
        Returns:
            Dict[factor_name, weight]: 标准化后的权重分配
        """
        logger.info("🧮 开始计算IC加权权重...")

        # 1. 计算综合IC得分
        factor_scores = {}
        for factor_name, periods_stats in factor_ic_stats.items():
            normalize_stats = self._normalize_factor_stats_direction(periods_stats,factor_name)
            score = self._calculate_composite_ic_score(normalize_stats)
            factor_scores[factor_name] = score
            logger.debug(f"  {factor_name}: 综合IC得分 = {score:.4f}")

        # 2. 基于得分计算原始权重
        raw_weights = self._convert_scores_to_weights(factor_scores)

        # 3. 应用约束和标准化
        final_weights = self._apply_weight_constraints(raw_weights)

        logger.info(f"✅ IC权重计算完成，共{len(final_weights)}个因子被分配权重")
        return final_weights

    def _normalize_factor_stats_direction(self,
                                          raw_stats: Dict,
                                          factor_name: str) -> Dict:
        """
        根据预定义的因子方向，规范化IC统计数据。
        确保所有指标都处理成“正向”：值越大越好。
        """

        direction = get_new_factor_direction(factor_name)

        # 如果是正向因子，无需任何操作
        if direction == 1:
            return raw_stats
        logger.debug(f"检测到因子 {factor_name} 为负向因子(direction={direction})，开始进行方向规范化...")

        normalized_stats = {}
        for period, period_stats in raw_stats.items():
            # 复制以避免修改原始数据
            stats = period_stats.copy()

            # --- 核心扭转逻辑 ---
            # 1. 均值类指标：直接乘以方向 (-1)
            if 'ic_mean' in stats:
                stats['ic_mean'] *= direction
            if 'ic_ir' in stats:
                # IR的符号由IC均值决定，也直接扭转
                stats['ic_ir'] *= direction
            if 'ic_t_stat' in stats:
                stats['ic_t_stat'] *= direction

            # 3. 绝对值/方差类指标：保持不变
            #    ic_std, ic_p_value, ic_count, ic_mean_std, ic_ir_std 等都是衡量波动或统计量的，无需改变。

            normalized_stats[period] = stats

        return normalized_stats
    #ok
    def _calculate_composite_ic_score(self, periods_stats: Dict[str, Dict]) -> float:
        """计算因子的综合IC得分"""
        period_scores = []

        for period, stats in periods_stats.items():
            if not stats or 'ic_mean' not in stats:
                continue

            # 提取关键指标
            ic_mean = abs(stats.get('ic_mean', 0))  # 使用绝对值
            ic_ir = abs(stats.get('ic_ir', 0))
            ic_win_rate = stats.get('ic_win_rate', 0.5)
            ic_t_stat = abs(stats.get('ic_t_stat', 0))

            # 多维度评分模型
            # 1. IC均值得分 (40%权重)
            ic_mean_score = min(ic_mean / 0.05, 1.0) * 0.4

            # 2. IC稳定性得分 (30%权重) 
            ic_stability_score = min(ic_ir / 0.5, 1.0) * 0.3

            # 3. IC胜率得分 (20%权重)
            ic_win_score = max(0, (ic_win_rate - 0.5) * 2) * 0.2

            # 4. IC显著性得分 (10%权重)
            ic_sig_score = min(ic_t_stat / 2.0, 1.0) * 0.1

            period_score = ic_mean_score + ic_stability_score + ic_win_score + ic_sig_score
            period_scores.append(period_score)

        # 多周期平均，给短期稍高权重
        if not period_scores:
            return 0.0

        if len(period_scores) == 1:
            return period_scores[0]
        else:
            # --- 改进的加权方案 ---
            # 使用指数衰减权重，给短期更高权重，但依然考虑所有周期
            # decay_rate 越小，权重衰减越慢
            decay_rate = 0.75
            weights = np.array([decay_rate ** i for i in range(len(period_scores))])
            weights /= weights.sum()  # 权重归一化

            logger.debug(f"  多周期权重 (从1d到120d): {[f'{w:.2f}' for w in weights]}")
            return np.average(period_scores, weights=weights)

    def _convert_scores_to_weights(self, factor_scores: Dict[str, float]) -> Dict[str, float]:
        """将得分转换为权重"""
        if not factor_scores:
            return {}

        # 使用 softmax 函数将得分转换为权重，增强区分度
        scores = np.array(list(factor_scores.values()))

        # 过滤掉得分过低的因子
        valid_mask = scores > 0.1
        if not valid_mask.any():
            logger.warning("⚠️ 所有因子得分都过低，使用等权重")
            return {name: 1.0 / len(factor_scores) for name in factor_scores.keys()}

        # 对有效因子应用 softmax
        valid_scores = scores[valid_mask]
        valid_names = [name for i, name in enumerate(factor_scores.keys()) if valid_mask[i]]

        # 温度参数控制权重集中度，温度越高越平均
        temperature = 2.0
        exp_scores = np.exp(valid_scores / temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        return dict(zip(valid_names, softmax_weights))

    def _apply_weight_constraints(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        if not raw_weights:
            return {}

        weights = raw_weights.copy()

        # 1. 应用单因子权重上下限
        for name in weights:
            weights[name] = np.clip(weights[name],
                                    self.config.min_single_weight,
                                    self.config.max_single_weight)

        # 2. 重新标准化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}

        return weights


class FactorQualityFilter:
    """因子质量筛选器"""

    def __init__(self, config: FactorWeightingConfig):
        self.config = config

    def filter_factors_by_quality(
            self,
            factor_ic_stats: Dict[str, Dict[str, Dict]]
    ) -> Tuple[Dict[str, Dict[str, Dict]], List[FactorQualityReport]]:
        """
        基于多维度质量指标筛选因子
        
        Returns:
            (filtered_factor_stats, quality_reports)
        """
        logger.info("🔍 开始因子质量筛选...")

        filtered_stats = {}
        quality_reports = []

        for factor_name, periods_stats in factor_ic_stats.items():
            report = self._evaluate_single_factor(factor_name, periods_stats)
            quality_reports.append(report)

            if self._passes_quality_filter(report):
                filtered_stats[factor_name] = periods_stats
                logger.info(f"✅ {factor_name}: 通过筛选 (权重={report.weight:.3f}) - {report.selection_reason}")
            else:
                logger.info(f"❌ {factor_name}: 未通过筛选 - {report.selection_reason}")

        logger.info(f"📊 筛选结果: {len(filtered_stats)}/{len(factor_ic_stats)} 个因子通过质量检验")
        return filtered_stats, quality_reports

    def _evaluate_single_factor(self, factor_name: str, periods_stats: Dict) -> FactorQualityReport:
        """评估单个因子质量"""
        # 计算关键指标的综合表现
        ic_means = []
        ic_irs = []
        ic_win_rates = []
        ic_p_values = []

        for period, stats in periods_stats.items():
            if stats and 'ic_mean' in stats:
                ic_means.append(abs(stats.get('ic_mean', 0)))
                ic_irs.append(stats.get('ic_ir', 0))
                ic_win_rates.append(stats.get('ic_win_rate', 0.5))
                ic_p_values.append(stats.get('ic_p_value', 1.0))

        if not ic_means:
            return FactorQualityReport(
                factor_name=factor_name,
                ic_stats={},
                weight=0.0,
                selection_reason="缺少有效的IC统计数据",
                risk_flags=["数据不足"]
            )

        # 综合统计 之前对 每个period 用不同的时间进行aver，现在对不同的period进行aver
        avg_ic_mean = np.mean(ic_means)
        avg_ic_ir = np.mean(ic_irs)
        avg_win_rate = np.mean(ic_win_rates)
        min_p_value = min(ic_p_values)

        ic_stats = {
            'avg_ic_mean': avg_ic_mean,
            'avg_ic_ir': avg_ic_ir,
            'avg_win_rate': avg_win_rate,
            'min_p_value': min_p_value
        }

        # 质量评估
        risk_flags = []
        selection_reason = ""
        if avg_ic_mean < self.config.min_ic_mean:
            risk_flags.append(f"IC均值过低({avg_ic_mean:.3f})")
        if avg_ic_ir < self.config.min_ic_ir:
            risk_flags.append(f"IC信息比率过低({avg_ic_ir:.3f})")
        if avg_win_rate < self.config.min_ic_win_rate:
            risk_flags.append(f"IC胜率过低({avg_win_rate:.3f})")
        if min_p_value > self.config.max_ic_p_value:
            risk_flags.append(f"IC显著性不足(p={min_p_value:.3f})")

        if not risk_flags:
            selection_reason = f"高质量因子 (IC={avg_ic_mean:.3f}, IR={avg_ic_ir:.2f}, 胜率={avg_win_rate:.1%})"
            weight = self._calculate_factor_weight(ic_stats)
        else:
            selection_reason = "; ".join(risk_flags)
            weight = 0.0

        return FactorQualityReport(
            factor_name=factor_name,
            ic_stats=ic_stats,
            weight=weight,
            selection_reason=selection_reason,
            risk_flags=risk_flags
        )

    def _passes_quality_filter(self, report: FactorQualityReport) -> bool:
        """判断因子是否通过质量筛选"""
        return len(report.risk_flags) == 0 and report.weight > 0

    def _calculate_factor_weight(self, ic_stats: Dict[str, float]) -> float:
        """基于IC统计计算因子初步权重"""
        # 这里可以实现更复杂的权重逻辑
        # 目前使用简单的综合得分
        ic_mean = ic_stats.get('avg_ic_mean', 0)
        ic_ir = ic_stats.get('avg_ic_ir', 0)
        win_rate = ic_stats.get('avg_win_rate', 0.5)

        # 综合得分
        score = ic_mean * 0.5 + ic_ir * 0.3 + max(0, win_rate - 0.5) * 0.2
        return min(score, 1.0)

from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import FactorSynthesizer
class ICWeightedSynthesizer(FactorSynthesizer):
    """IC加权因子合成器 - 继承并扩展现有功能"""

    def __init__(self, factor_manager, factor_analyzer, factor_processor,
                 config: Optional[FactorWeightingConfig] = None, 
                 selector_config: Optional[RollingICSelectionConfig] = None):
        super().__init__(factor_manager, factor_analyzer, factor_processor)

        self.config = config or FactorWeightingConfig()
        self.weight_calculator = ICWeightCalculator(self.config)
        self.quality_filter = FactorQualityFilter(self.config)

        # 设置工作路径
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result")

        # 滚动IC管理器 - 核心改进
        rolling_ic_config = ICCalculationConfig(
            lookback_months=12,
            forward_periods=self.config.lookback_periods,
            calculation_frequency='M'
        )

        # 集成专业的滚动IC因子筛选器
        self.selector_config = selector_config or RollingICSelectionConfig()
        self.factor_selector = None  # 延迟初始化，需要snap_config_id

        # 缓存IC统计数据，避免重复计算
        self._ic_stats_cache = {}
    #这个函数没有正交化合成！
    def synthesize_ic_weighted_factor(
            self,
            composite_factor_name: str,
            stock_pool_index: str,
            candidate_factor_names: List[str],
            force_recalculate_ic: bool = False,
            snap_config_id: str = None

    ) -> Tuple[pd.DataFrame, Dict]:
        """
        IC加权因子合成主流程
        
        Args:
            composite_factor_name: 复合因子名称
            stock_pool_index: 股票池名称
            candidate_factor_names: 候选因子列表
            force_recalculate_ic: 是否强制重新计算IC
            
        Returns:
            (composite_factor_df, synthesis_report)
        """
        logger.info(f"\n🚀 开始IC加权因子合成: {composite_factor_name}")
        logger.info(f"📊 候选因子数量: {len(candidate_factor_names)}")
        logger.info(f"📈 目标股票池INDEX: {stock_pool_index}")

        # 第一步：收集候选因子的IC统计数据
        factor_ic_stats = self._collect_factor_ic_stats(
            candidate_factor_names,
            stock_pool_index,
            force_recalculate_ic,
            snap_config_id

        )

        # 第二步：质量筛选
        qualified_factor_stats, quality_reports = self.quality_filter.filter_factors_by_quality(
            factor_ic_stats
        )

        if not qualified_factor_stats:
            raise ValueError("❌ 没有因子通过质量筛选，无法进行合成")

        # 第三步：计算权重
        factor_weights = self.weight_calculator.calculate_ic_based_weights(
            qualified_factor_stats
        )

        # 第四步：执行加权合成
        composite_factor_df = self._execute_weighted_synthesis(
            composite_factor_name,
            stock_pool_index,
            factor_weights,
            snap_config_id
        )

        # 生成合成报告
        synthesis_report = self._generate_synthesis_report(
            composite_factor_name,
            candidate_factor_names,
            factor_weights,
            quality_reports
        )


        logger.info(f"✅ IC加权因子合成完成: {composite_factor_name}")
        return composite_factor_df, synthesis_report

    def calculate_rolling_weights(
            self,
            candidate_factor_names: List[str],
            stock_pool_index_name: str,
            calculation_date: str,
            resultLoadManager: ResultLoadManager
    ) -> Dict[str, float]:
        """
        滚动权重计算 - 核心改进：完全避免前视偏差
        
        Args:
            candidate_factor_names: 候选因子列表
            stock_pool_index_name: 股票池名称
            calculation_date: 权重计算时点（严格不使用此时点之后的数据）
            factor_data_source: 因子数据源
            return_data_source: 收益数据源
            
        Returns:
            Dict[factor_name, weight]: 基于历史IC的权重分配
        """
        logger.info(f"🔄 开始滚动权重计算 @ {calculation_date}")
        logger.info(f"📊 候选因子: {len(candidate_factor_names)} 个")

        # 第一步：获取截止到calculation_date的历史IC数据
        historical_ic_stats = {}

        for factor_name in candidate_factor_names:
            try:
                # 从滚动IC管理器获取历史IC快照
                latest_snapshot = self.rolling_ic_manager.get_ic_at_timepoint(
                    factor_name, stock_pool_index_name, calculation_date
                )

                if latest_snapshot and latest_snapshot.stats:
                    historical_ic_stats[factor_name] = latest_snapshot.stats
                    logger.debug(f"  ✅ {factor_name}: 获取历史IC @ {calculation_date}")
                else:
                    snapshot = self.rolling_ic_manager._calculate_ic_snapshot(
                        factor_name, stock_pool_index_name, calculation_date,
                        resultLoadManager
                    )
                    if snapshot and snapshot.stats:
                        historical_ic_stats[factor_name] = snapshot.stats
                        # 保存快照以供后续使用
                        self.rolling_ic_manager._save_snapshot(snapshot)
                        logger.debug(f"  🔄 {factor_name}: 实时计算IC @ {calculation_date}")
                    else:
                        logger.warning(f"  ❌ {factor_name}: 无法计算历史IC--正常：因为不满足120个观测点！")
            except Exception as e:
                logger.error(f"  ❌ {factor_name}: IC获取失败 - {e}")
                continue

        if not historical_ic_stats:
            logger.error("❌ 无任何因子的历史IC数据，无法计算权重")
            return {}

        logger.info(f"📊 成功获取 {len(historical_ic_stats)} 个因子的历史IC数据")

        # 第二步：基于历史IC进行质量筛选
        qualified_factor_stats, quality_reports = self.quality_filter.filter_factors_by_quality(
            historical_ic_stats
        )

        if not qualified_factor_stats:
            logger.warning("⚠️ 无因子通过质量筛选，返回等权重")
            equal_weight = 1.0 / len(candidate_factor_names)
            return {name: equal_weight for name in candidate_factor_names}

        # 第三步：计算权重（仅基于历史IC表现）
        factor_weights = self.weight_calculator.calculate_ic_based_weights(
            qualified_factor_stats
        )

        # 第四步：为未通过筛选的因子分配0权重
        final_weights = {}
        for factor_name in candidate_factor_names:
            final_weights[factor_name] = factor_weights.get(factor_name, 0.0)

        # 日志记录
        selected_factors = [name for name, weight in final_weights.items() if weight > 0]
        logger.info(f"✅ 滚动权重计算完成 @ {calculation_date}")
        logger.info(f"📊 选中因子: {len(selected_factors)}/{len(candidate_factor_names)}")

        for factor_name, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                logger.info(f"  🎯 {factor_name}: {weight:.1%}")

        return final_weights

    # 深入剖析
    def _collect_factor_ic_stats(
            self,
            factor_names: List[str],
            stock_pool_index: str,
            force_recalculate: bool = False,
            snap_config_id: str = None
    ) -> Dict[str, Dict[str, Dict]]:
        """收集因子IC统计数据"""
        logger.info("📊 正在收集因子IC统计数据...")

        factor_ic_stats = {}

        for factor_name in factor_names:
            cache_key = f"{factor_name}_{stock_pool_index}"

            if not force_recalculate and cache_key in self._ic_stats_cache:
                factor_ic_stats[factor_name] = self._ic_stats_cache[cache_key]
                logger.debug(f"  📥 {factor_name}: 使用缓存数据")
                continue

            try:
                # 从已保存的测试结果中读取IC统计
                ic_stats = self._load_factor_ic_stats(factor_name=factor_name,stock_pool_index= stock_pool_index, snap_config_id=snap_config_id)

                if ic_stats:
                    factor_ic_stats[factor_name] = ic_stats
                    self._ic_stats_cache[cache_key] = ic_stats
                    logger.debug(f"  ✅ {factor_name}: IC数据加载成功")
                else:
                    logger.warning(f"  ⚠️ {factor_name}: 未找到IC统计数据，跳过")

            except Exception as e:
                raise ValueError(f"  ❌ {factor_name}: 加载IC数据失败 - {e}")

        logger.info(f"📊 IC数据收集完成: {len(factor_ic_stats)}/{len(factor_names)} 个因子")
        return factor_ic_stats

    def _load_factor_ic_stats(self, factor_name: str, stock_pool_index: str, calcu_type='o2o', snap_config_id: str = None) -> Optional[Dict]:
        """
        从滚动IC存储中提取因子的IC统计数据
        Args:
            factor_name: 因子名称
            stock_pool_index: 股票池索引
            calcu_type: 收益计算类型，默认'o2o'
            snap_config_id: 配置快照ID，用于确定版本
            
        Returns:
            Dict[period, ic_stats]: 各周期的IC统计数据，格式与RollingICManager一致
        """
        try:
            if snap_config_id is None:
                logger.warning(f"未提供snap_config_id，无法确定数据版本")
                return None
                
            # 1. 从配置快照获取版本信息
            config_manager = ConfigSnapshotManager()
            pool_index, start_date, end_date, config_evaluation = config_manager.get_snapshot_config_content_details(snap_config_id)
            version = f"{start_date}_{end_date}"
            
            # 2. 构建滚动IC文件路径
            rolling_ic_dir = (self.main_work_path / stock_pool_index / factor_name / 
                             calcu_type / version / 'rolling_ic')
            
            if not rolling_ic_dir.exists():
                # 就地生成IC数据并保存到本地
                logger.info(f"滚动IC目录不存在，开始就地生成: {factor_name}")
                try:
                    # 调用生成函数为当前因子生成IC数据
                    run_cal_and_save_rolling_ic_by_snapshot_config_id(snap_config_id, [factor_name])
                    logger.info(f"✅ 成功生成滚动IC数据: {factor_name}")
                    
                    # 重新检查目录是否存在
                    if not rolling_ic_dir.exists():
                        raise ValueError(f"for-{factor_name} 生成IC数据后目录仍不存在: {rolling_ic_dir}")
                except Exception as e:
                    raise ValueError(f"生成滚动IC数据失败 {factor_name}: {e}")

            # 3. 查找所有IC快照文件
            ic_files = list(rolling_ic_dir.glob("ic_snapshot_*.json"))
            if not ic_files:
                # 如果目录存在但无文件，可能是IC生成不完整，尝试重新生成
                logger.warning(f"IC目录存在但无快照文件，尝试重新生成: {factor_name}")
                try:
                    run_cal_and_save_rolling_ic_by_snapshot_config_id(snap_config_id, [factor_name])
                    
                    # 重新查找文件
                    ic_files = list(rolling_ic_dir.glob("ic_snapshot_*.json"))
                    if not ic_files:
                        raise ValueError(f"重新生成后仍无IC快照文件: {rolling_ic_dir}")
                    logger.info(f"✅ 重新生成IC数据成功: {factor_name}")
                except Exception as e:
                    raise ValueError(f"重新生成IC数据失败 {factor_name}: {e}")

            logger.debug(f"找到 {len(ic_files)} 个IC快照文件 for {factor_name}")
            
            # 4. 加载并聚合IC统计数据
            all_periods_stats = {}
            
            for ic_file in ic_files:
                try:
                    with open(ic_file, 'r', encoding='utf-8') as f:
                        snapshot_data = json.load(f)
                    
                    # 提取ic_stats字段
                    ic_stats = snapshot_data.get('stats', {})
                    
                    # 聚合各周期的统计数据
                    for period, period_stats in ic_stats.items():
                        if period not in all_periods_stats:
                            all_periods_stats[period] = []
                        all_periods_stats[period].append(period_stats)
                        
                except Exception as e:
                    logger.warning(f"读取IC文件失败 {ic_file}: {e}")
                    continue
            
            if not all_periods_stats:
                logger.debug(f"未找到有效的IC统计数据")
                return None
            
            # 5. 计算聚合统计指标
            aggregated_stats = {}
            for period, stats_list in all_periods_stats.items():#stat_list：所有快照
                if not stats_list:
                    continue
                    
                # 计算时间序列的平均指标
                ic_means = [s.get('ic_mean', 0) for s in stats_list if s.get('ic_mean') is not None]
                ic_stds = [s.get('ic_std', 0) for s in stats_list if s.get('ic_std') is not None]
                ic_irs = [s.get('ic_ir', 0) for s in stats_list if s.get('ic_ir') is not None]
                ic_win_rates = [s.get('ic_win_rate', 0.5) for s in stats_list if s.get('ic_win_rate') is not None]
                ic_p_values = [s.get('ic_p_value', 1.0) for s in stats_list if s.get('ic_p_value') is not None]
                ic_t_stats = [s.get('ic_t_stat', 0) for s in stats_list if s.get('ic_t_stat') is not None]
                
                if not ic_means:
                    continue
                
                # 聚合统计
                aggregated_stats[period] = {
                    'ic_mean': np.mean(ic_means),
                    'ic_std': np.mean(ic_stds) if ic_stds else 0,
                    'ic_ir': np.mean(ic_irs) if ic_irs else 0,
                    'ic_win_rate': np.mean(ic_win_rates) if ic_win_rates else 0.5,
                    'ic_p_value': np.mean(ic_p_values) if ic_p_values else 1.0,
                    'ic_t_stat': np.mean(ic_t_stats) if ic_t_stats else 0,
                    'ic_count': len(ic_means),
                    'snapshot_count': len(stats_list),
                    'ic_mean_std': np.std(ic_means) if len(ic_means) > 1 else 0,  # IC均值的稳定性
                    'ic_ir_std': np.std(ic_irs) if len(ic_irs) > 1 else 0  # IR的稳定性
                }
            
            if not aggregated_stats:
                logger.debug(f"聚合后无有效统计数据")
                return None
                
            logger.debug(f"成功提取因子 {factor_name} 的IC统计: {list(aggregated_stats.keys())} 周期")
            return aggregated_stats
            
        except Exception as e:
            logger.error(f"加载因子IC统计失败 {factor_name}: {e}")
            return None

    def _execute_weighted_synthesis(
            self,
            composite_factor_name: str,
            stock_pool_index_name: str,
            factor_weights: Dict[str, float],
            snap_config_id:str
    ) -> pd.DataFrame:
        """执行加权因子合成"""
        logger.info(f"⚖️ 开始执行加权合成，使用{len(factor_weights)}个因子")

        processed_factors = []
        weights_list = []

        for factor_name, weight in factor_weights.items():
            logger.info(f"  🔄 处理因子: {factor_name} (权重: {weight:.3f})")

            # 处理单个因子
            processed_df = self.get_sub_factor_df_from_local(factor_name, stock_pool_index_name,snap_config_id)

            processed_factors.append(processed_df)
            weights_list.append(weight)

        if not processed_factors:
            raise ValueError("没有任何因子被成功处理")

        # 加权合成
        composite_factor_df = self._weighted_combine_factors(processed_factors, weights_list)

        # 最终标准化
        composite_factor_df = self.processor._standardize_robust(composite_factor_df)

        logger.info(f"✅ 加权合成完成: {composite_factor_name}")
        return composite_factor_df

    def _weighted_combine_factors(
            self,
            factor_dfs: List[pd.DataFrame],
            weights: List[float]
    ) -> pd.DataFrame:
        """加权合并因子数据框"""
        if len(factor_dfs) != len(weights):
            raise ValueError("因子数量与权重数量不匹配")

        # 确保权重归一化
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 加权合并
        result_df = None
        for i, (df, weight) in enumerate(zip(factor_dfs, weights)):
            weighted_df = df * weight

            if result_df is None:
                result_df = weighted_df
            else:
                result_df = result_df.add(weighted_df, fill_value=0)

        return result_df

    def _generate_synthesis_report(
            self,
            composite_factor_name: str,
            candidate_factors: List[str],
            final_weights: Dict[str, float],
            quality_reports: List[FactorQualityReport]
    ) -> Dict:
        """生成因子合成报告"""

        # 按权重排序
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

        report = {
            'composite_factor_name': composite_factor_name,
            'synthesis_timestamp': pd.Timestamp.now(),
            'candidate_factors_count': len(candidate_factors),
            'qualified_factors_count': len(final_weights),
            'final_weights': dict(sorted_weights),
            'top_3_factors': sorted_weights[:3],
            'config_used': {
                'min_ic_mean': self.config.min_ic_mean,
                'min_ic_ir': self.config.min_ic_ir,
                'min_ic_win_rate': self.config.min_ic_win_rate,
                'max_single_weight': self.config.max_single_weight
            },
            'quality_summary': {
                'passed': len([r for r in quality_reports if not r.risk_flags]),
                'failed': len([r for r in quality_reports if r.risk_flags]),
                'main_failure_reasons': self._summarize_failure_reasons(quality_reports)
            }
        }

        return report

    def _summarize_failure_reasons(self, quality_reports: List[FactorQualityReport]) -> Dict[str, int]:
        """汇总失败原因统计"""
        failure_counts = {}

        for report in quality_reports:
            for flag in report.risk_flags:
                reason = flag.split('(')[0]  # 提取原因主体
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        return failure_counts
    
    def _generate_comprehensive_report(
            self,
            composite_factor_name: str,
            candidate_factors: List[str],
            selected_factors: List[str],
            final_weights: Dict[str, float],
            selection_report: Dict
    ) -> Dict:
        """生成包含筛选和合成信息的综合报告"""
        
        # 基础合成报告
        base_report = self._generate_synthesis_report(
            composite_factor_name, candidate_factors, final_weights, []
        )
        
        # 添加专业筛选信息
        comprehensive_report = {
            **base_report,
            'professional_selection': {
                'selection_method': 'RollingIC-based Professional Selection',
                'candidate_count': len(candidate_factors),
                'selected_count': len(selected_factors),
                'selection_rate': len(selected_factors) / len(candidate_factors) if candidate_factors else 0,
                'selected_factors': selected_factors,
                'selection_report': selection_report
            }
        }
        
        return comprehensive_report

    def print_synthesis_report(self, report: Dict):
        """打印合成报告（支持专业筛选和传统筛选两种格式）"""
        print(f"\n{'=' * 80}")
        
        # 检查是否是专业筛选报告
        if 'professional_selection' in report:
            print(f"📊 专业滚动IC筛选+IC加权合成报告")
            print(f"{'=' * 80}")
            print(f"🎯 合成因子名称: {report['composite_factor_name']}")
            print(f"⏰ 合成时间: {report['synthesis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 专业筛选信息
            prof_sel = report['professional_selection']
            print(f"\n🔍 专业筛选结果:")
            print(f"  📈 候选因子数量: {prof_sel['candidate_count']}")
            print(f"  ✅ 筛选通过数量: {prof_sel['selected_count']}")
            print(f"  📊 筛选通过率: {prof_sel['selection_rate']:.1%}")
            print(f"  🏆 筛选方法: {prof_sel['selection_method']}")
            
            print(f"\n🎯 最终选中因子:")
            for i, factor in enumerate(prof_sel['selected_factors'], 1):
                weight = report['final_weights'].get(factor, 0)
                print(f"  {i:2d}. {factor:25s}: {weight:6.1%}")
                
        else:
            print(f"📊 IC加权因子合成报告")
            print(f"{'=' * 80}")
            print(f"🎯 合成因子名称: {report['composite_factor_name']}")
            print(f"⏰ 合成时间: {report['synthesis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📈 候选因子数量: {report['candidate_factors_count']}")
            print(f"✅ 通过筛选数量: {report['qualified_factors_count']}")

            print(f"\n🏆 最终权重分配:")
            for factor_name, weight in report['final_weights'].items():
                print(f"  {factor_name:25s}: {weight:6.1%}")

        print(f"\n🥇 权重前三名:")
        for i, (factor_name, weight) in enumerate(report.get('top_3_factors', []), 1):
            print(f"  {i}. {factor_name}: {weight:.1%}")

        # 质量筛选信息（如果存在）
        if 'quality_summary' in report:
            quality_summary = report['quality_summary']
            print(f"\n📋 质量筛选汇总:")
            print(f"  ✅ 通过: {quality_summary['passed']} 个")
            print(f"  ❌ 失败: {quality_summary['failed']} 个")

            if quality_summary['main_failure_reasons']:
                print(f"  主要失败原因:")
                for reason, count in quality_summary['main_failure_reasons'].items():
                    print(f"    - {reason}: {count} 个因子")

        print(f"{'=' * 80}")

    def execute_orthogonalization_plan(
            self,
            orthogonalization_plan: List[Dict],
            stock_pool_index: str,
            snap_config_id: str
    ) -> Dict[str, pd.DataFrame]:
        """
        执行正交化改造计划 - 核心功能：截面线性回归残差提取
        
        Args:
            orthogonalization_plan: 正交化计划列表，每个元素包含：
                - original_factor: 目标因子名称
                - base_factor: 基准因子名称
                - orthogonal_name: 正交化后的新因子名称
                - correlation: 原始相关性
                - base_score: 基准因子评分
                - target_score: 目标因子评分
            stock_pool_index: 股票池名称
            snap_config_id: 配置快照ID
            
        Returns:
            Dict[orthogonal_name, orthogonal_factor_df]: 正交化后的因子数据
        """
        if not orthogonalization_plan:
            logger.info("⚪ 无正交化计划，跳过执行")
            return {}
            
        logger.info(f"🔧 开始执行正交化计划，共 {len(orthogonalization_plan)} 项")
        
        orthogonal_factors = {}
        
        for plan_item in orthogonalization_plan:
            try:
                orthogonal_factor_df, avg_r_squared = self._execute_single_orthogonalization(
                    plan_item, stock_pool_index, snap_config_id
                )
                
                if orthogonal_factor_df is not None:
                    orthogonal_factors[plan_item['orthogonal_name']] = orthogonal_factor_df
                    logger.info(f"✅ 成功生成正交化因子: {plan_item['orthogonal_name']} (R²={avg_r_squared:.3f})")
                else:
                    logger.warning(f"⚠️ 正交化失败: {plan_item['orthogonal_name']}")
                    
            except Exception as e:
                logger.error(f"❌ 正交化执行异常 {plan_item['orthogonal_name']}: {e}")
                continue
        
        logger.info(f"🎯 正交化执行完成，成功生成 {len(orthogonal_factors)} 个正交化因子")
        return orthogonal_factors

    def _execute_single_orthogonalization(
            self,
            plan_item: Dict,
            stock_pool_index: str,
            snap_config_id: str
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """
        执行单个正交化改造 - 逐日截面OLS回归
        
        核心逻辑：
        1. 加载目标因子和基准因子数据
        2. 逐日进行截面线性回归：target_factor = α + β * base_factor + ε
        3. 提取残差ε作为正交化后的因子值
        
        Args:
            plan_item: 单个正交化计划
            stock_pool_index: 股票池名称
            snap_config_id: 配置快照ID
            
        Returns:
            (正交化后的因子DataFrame, 平均R²): 用于IC调整
        """
        target_factor = plan_item['original_factor']
        base_factor = plan_item['base_factor']
        orthogonal_name = plan_item['orthogonal_name']
        
        logger.debug(f"  🔄 执行正交化: {target_factor} vs {base_factor} -> {orthogonal_name}")
        
        try:
            # 1. 加载因子数据
            target_df = self.get_sub_factor_df_from_local(target_factor, stock_pool_index, snap_config_id)
            base_df = self.get_sub_factor_df_from_local(base_factor, stock_pool_index, snap_config_id)
            
            if target_df is None or base_df is None:
                raise ValueError(f"  ❌ 无法加载因子数据: target={target_df is not None}, base={base_df is not None}")

            # 2. 数据对齐和预处理
            aligned_target, aligned_base = self._align_factor_data(target_df, base_df)
            
            if aligned_target.empty or aligned_base.empty:
                logger.error("  ❌ 因子数据对齐后为空")
                return None, 0.0
            
            # 3. 逐日截面回归，获取R²用于IC调整
            orthogonal_df, avg_r_squared = self._daily_cross_sectional_orthogonalization(
                aligned_target, aligned_base, orthogonal_name
            )
            
            # 记录R²信息用于后续IC调整
            plan_item['avg_r_squared'] = avg_r_squared
            
            return orthogonal_df, avg_r_squared
            
        except Exception as e:
            logger.error(f"  ❌ 单项正交化失败 {orthogonal_name}: {e}")
            return None, 0.0

    def _align_factor_data(
            self,
            target_df: pd.DataFrame,
            base_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        因子数据对齐 - 确保时间和股票维度一致
        
        Args:
            target_df: 目标因子数据
            base_df: 基准因子数据
            
        Returns:
            (aligned_target, aligned_base): 对齐后的数据
        """
        # 找到共同的时间和股票
        common_dates = target_df.index.intersection(base_df.index)
        common_stocks = target_df.columns.intersection(base_df.columns)
        
        if len(common_dates) == 0 or len(common_stocks) == 0:
            logger.error(f"  ❌ 无共同时间点或股票：日期={len(common_dates)}, 股票={len(common_stocks)}")
            return pd.DataFrame(), pd.DataFrame()
        
        # 数据对齐
        aligned_target = target_df.loc[common_dates, common_stocks]
        aligned_base = base_df.loc[common_dates, common_stocks]
        
        logger.debug(f"  📊 数据对齐完成：{len(common_dates)}个交易日, {len(common_stocks)}只股票")
        
        return aligned_target, aligned_base

    def _daily_cross_sectional_orthogonalization(
            self,
            target_df: pd.DataFrame,
            base_df: pd.DataFrame,
            orthogonal_name: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        逐日截面正交化 (V2 - 战斗加固版)
        - 强化1: 使用 align 确保索引严格对齐
        - 强化2: 统一最小样本量参数
        - 强化3: 标准化时使用 ddof=0 并加 epsilon
        - 强化4: 返回包含成功率、失败日期等信息的元数据字典
        """
        logger.debug(f"  🧮 V2: 开始逐日截面回归，共{len(target_df)}个交易日")

        orthogonal_df = pd.DataFrame(index=target_df.index, columns=target_df.columns, dtype=np.float64)

        r_squared_list = []
        failed_dates = []
        epsilon = 1e-8

        for date in target_df.index:
            try:
                y_cross = target_df.loc[date]
                x_cross = base_df.loc[date]

                # 【强化】使用 align 严格对齐 y 和 x，并移除共同的NaN
                y_aligned, x_aligned = y_cross.align(x_cross, join='inner')
                valid_mask = (~y_aligned.isna()) & (~x_aligned.isna())
                y_valid = y_aligned[valid_mask]
                x_valid = x_aligned[valid_mask]

                # 【强化】统一最小样本量检查
                if len(y_valid) < self.config.min_orthogonalization_obs:
                    logger.debug(f"    ⚠️ {date}: 有效观测不足({len(y_valid)}个)，跳过")
                    failed_dates.append(date.strftime('%Y-%m-%d'))
                    continue

                residuals, r_squared = self._perform_cross_sectional_ols(y_valid, x_valid, date)

                if residuals is not None and r_squared is not None:
                    # 【强化】标准化时使用 ddof=0 (总体标准差)
                    std_val = residuals.std(ddof=0)

                    if std_val > epsilon:
                        mean_val = residuals.mean()  # OLS残差均值理论上为0
                        standardized_residuals = (residuals - mean_val) / std_val
                        orthogonal_df.loc[date, standardized_residuals.index] = standardized_residuals.values
                        r_squared_list.append(r_squared)
                    else:
                        # 如果标准差为0，说明残差为常数，填充为0
                        orthogonal_df.loc[date, residuals.index] = 0.0
                        r_squared_list.append(r_squared)
                else:
                    failed_dates.append(date.strftime('%Y-%m-%d'))

            except Exception as e:
                logger.debug(f"    ❌ {date}: 回归循环失败 - {e}")
                failed_dates.append(date.strftime('%Y-%m-%d'))
                continue

        successful_regressions = len(r_squared_list)
        total_days = len(target_df)

        # 【强化】构建包含丰富信息的元数据字典
        meta = {
            'successful_regressions': successful_regressions,
            'total_days': total_days,
            'success_rate': successful_regressions / total_days if total_days > 0 else 0.0,
            'avg_r_squared': np.mean(r_squared_list) if r_squared_list else 0.0,
            'failed_dates': failed_dates
        }

        if meta['success_rate'] < 0.9:  # 如果成功率低于90%，则提升日志级别为警告
            raise ValueError(f"  ⚠️ {orthogonal_name}: 截面回归成功率较低: {meta['success_rate']:.1%}")
        else:
            logger.debug(f"  ✅ {orthogonal_name}: 截面回归完成, 成功率 {meta['success_rate']:.1%}")

        return orthogonal_df, meta['avg_r_squared']

    def _perform_cross_sectional_ols(
            self,
            y: pd.Series,
            x: pd.Series,
            date: str = None
    ) -> Tuple[Optional[pd.Series], Optional[float]]:
        """
        执行单日截面OLS回归并提取残差 (V2 - 战斗加固版)
        - 强化1: 对高R²只告警不中断
        - 强化2: 检查自变量x的方差，避免退化回归
        - 强化3: 捕获 statsmodels 内部的拟合错误
        """
        try:
            # 【强化】检查自变量标准差，如果过小（接近常数），则回归无意义
            if x.std(ddof=0) < self.config.orthogonalization_x_std_eps:
                logger.debug(f"    ⚠️ {date}: 基准因子标准差过低({x.std(ddof=0):.2e})，跳过回归")
                return None, None

            # 使用 has_constant='add' 更稳健，避免重复添加常数项
            X_with_const = sm.add_constant(x, has_constant='add')

            # 【强化】单独捕获模型拟合时的线性代数错误（如奇异矩阵）
            try:
                model = sm.OLS(y, X_with_const).fit()
            except LinAlgError as e:
                raise ValueError(f"    ❌ {date}: OLS拟合失败 - 矩阵错误: {e}")

            residuals = model.resid
            r_squared = model.rsquared

            # 【强化】对于异常高的R²，只记录警告，不中断流程
            if r_squared > 0.95:
                logger.warning(f"    ⚠️ {date}: R²异常高({r_squared:.3f})，可能存在数据共线性问题")

            return residuals, r_squared

        except Exception as e:
            raise ValueError(f"    ❌ {date}: _perform_cross_sectional_ols_v2 发生意外错误: {e}")

    def _perform_ols_sklearn_fallback(
            self,
            y: pd.Series,
            x: pd.Series
    ) -> Tuple[Optional[pd.Series], Optional[float]]:
        """
        sklearn回归备用方案
        
        Args:
            y: 因变量
            x: 自变量
            
        Returns:
            (残差序列, R²值): 残差和拟合度
        """
        try:
            # sklearn会自动添加截距项（如果fit_intercept=True）
            reg = LinearRegression(fit_intercept=True)
            
            # reshape数据
            X = x.values.reshape(-1, 1)
            y_values = y.values
            
            # 拟合模型
            reg.fit(X, y_values)
            
            # 计算预测值和残差
            y_pred = reg.predict(X)
            residuals = y_values - y_pred
            
            # 计算R²
            r_squared = reg.score(X, y_values)
            
            # 返回pandas Series格式和R²
            residuals_series = pd.Series(residuals, index=y.index)
            return residuals_series, r_squared
            
        except Exception as e:
            logger.debug(f"    ❌ sklearn回归失败: {e}")
            return None, None

    def _standardize_orthogonal_factor(
            self,
            orthogonal_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        标准化正交化因子
        
        应用截面标准化：每个交易日内，因子值标准化为均值0、标准差1
        
        Args:
            orthogonal_df: 原始正交化因子
            
        Returns:
            标准化后的正交化因子
        """
        if orthogonal_df.empty:
            return orthogonal_df
        
        logger.debug("  📐 开始因子标准化")
        
        standardized_df = orthogonal_df.copy()
        
        # 逐日标准化
        for date in orthogonal_df.index:
            date_values = orthogonal_df.loc[date]
            valid_values = date_values.dropna()
            
            if len(valid_values) < 5:  # 至少需要5个有效值
                continue
            
            # Z-Score标准化
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            
            if std_val > 1e-8:  # 避免除零
                standardized_values = (valid_values - mean_val) / std_val
                standardized_df.loc[date, valid_values.index] = standardized_values
        
        logger.debug("  ✅ 因子标准化完成")
        return standardized_df

    def _adjust_ic_stats_by_r_squared(
            self,
            original_ic_stats: Dict[str, Dict],
            avg_r_squared: float,
            orthogonal_factor_name: str
    ) -> Dict[str, Dict]:
        """
        基于R²调整正交化因子的IC统计 (V2 - 最终生产版)

        核心理论：标准差/相关性分解 (Std Dev / Correlation Decomposition)
        此版本经过严格审查和加固，解决了IC IR不变性、p-value除零保护等问题。

        调整公式：
        - 核心调整系数: adjustment_factor = sqrt(1 - R²)
        - IC/Std/T-stat类指标: adjusted = original * adjustment_factor
        - IC IR: 理论上不变 (因子被抵消)
        - 胜率调整: 0.5 + (original_rate - 0.5) * adjustment_factor
        - p值调整: min(1.0, p_value / (adjustment_factor + eps))

        Args:
            original_ic_stats: 原始因子的IC统计数据
            avg_r_squared: 平均R²值（来自逐日回归）
            orthogonal_factor_name: 正交化因子名称（用于日志）

        Returns:
            调整后的IC统计数据
        """
        epsilon = 1e-8  # 用于防止除零错误

        # --- 1. 健全性检查 ---
        if not (0 < avg_r_squared < 1):
            raise ValueError(
                f"  ⚠️ {orthogonal_factor_name}: 异常或无效的R²值({avg_r_squared:.3f})，无法进行调整，将使用原始IC统计。")

        # --- 2. 计算核心调整系数 ---
        adjustment_factor = np.sqrt(1 - avg_r_squared)
        logger.debug(f"  📊 {orthogonal_factor_name}: R²={avg_r_squared:.3f}, 统一调整系数={adjustment_factor:.3f}")

        adjusted_ic_stats = {}

        # --- 3. 逐个预测周期进行调整 ---
        for period, period_stats in original_ic_stats.items():
            # 使用 .copy() 是一个好习惯，避免意外修改原始输入字典
            adjusted_period_stats = period_stats.copy()

            # 3.1: 直接乘以调整系数的指标 (线性相关类)
            keys_to_scale = [
                'ic_mean', 'ic_std', 'ic_volatility',
                'ic_t_stat', 'ic_nw_t_stat'
            ]
            # 自动包含所有以 _score 结尾的评分项
            score_keys = [k for k in adjusted_period_stats if k.endswith('_score')]
            keys_to_scale.extend(score_keys)

            for key in keys_to_scale:
                if key in adjusted_period_stats and isinstance(adjusted_period_stats[key], (int, float)):
                    adjusted_period_stats[key] *= adjustment_factor

            # 3.2: 具有特殊调整逻辑的指标
            # 胜率(Win Rate): 向50%基准进行收缩
            if 'ic_win_rate' in adjusted_period_stats:
                original_win_rate = adjusted_period_stats['ic_win_rate']
                adjusted_period_stats['ic_win_rate'] = 0.5 + (original_win_rate - 0.5) * adjustment_factor

            # P值(P-values): 反向调整，并用epsilon进行除零保护
            p_value_keys = ['ic_p_value', 'ic_nw_p_value']
            for key in p_value_keys:
                if key in adjusted_period_stats:
                    original_p_value = adjusted_period_stats[key]
                    adjusted_period_stats[key] = min(1.0, original_p_value / (adjustment_factor + epsilon))

            # 3.3: 理论上保持不变的指标
            # IC IR = IC均值 / IC标准差。调整因子在分子分母上被抵消，故IR理论上不变。
            # 其他如样本数、最大/最小值等统计量也保持不变。
            keys_to_keep = ['ic_ir', 'ic_count', 'ic_max', 'ic_min']
            # (我们因为使用了.copy()，所以无需额外代码，这些值已自动保留)

            adjusted_ic_stats[period] = adjusted_period_stats

        # --- 4. 日志记录调整效果 ---
        # 使用 .get() 链式调用来安全地获取值，避免KeyError
        original_main_ic = original_ic_stats.get(self.config.main_evaluation_period, {}).get('ic_mean', 0)
        adjusted_main_ic = adjusted_ic_stats.get(self.config.main_evaluation_period, {}).get('ic_mean', 0)

        adjustment_magnitude = (1 - adjustment_factor) * 100

        logger.info(f"  🔄 {orthogonal_factor_name}: 主周期IC调整 {original_main_ic:.4f} -> {adjusted_main_ic:.4f}")
        logger.info(
            f"      理论依据: sqrt(1-R²)={adjustment_factor:.3f}, IC预测力下降幅度: {adjustment_magnitude:.1f}%")

        return adjusted_ic_stats


    #todo 看这里
    def synthesize_with_orthogonalization(
            self,
            composite_factor_name: str,
            candidate_factor_names: List[str],
            snap_config_id: str,
            force_generate_ic: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        带正交化的专业因子合成流程
        
        完整流程：
        1. 专业筛选（红色区域淘汰 + 黄色区域正交化计划）
        2. 执行正交化改造计划
        3. 基于处理后的因子进行IC加权合成
        
        Args:
            composite_factor_name: 复合因子名称
            candidate_factor_names: 候选因子列表
            snap_config_id: 配置快照ID
            force_generate_ic: 是否强制重新生成IC数据
            
        Returns:
            (composite_factor_df, synthesis_report)
        """
        logger.info(f"\n🚀 启动带正交化的专业因子合成: {composite_factor_name}")
        logger.info(f"📊 候选因子数量: {len(candidate_factor_names)}")
        
        # 1. 初始化专业筛选器
        if self.factor_selector is None:
            self.factor_selector = FactorSelector(snap_config_id, self.selector_config)
            logger.info("✅ 滚动IC因子筛选器初始化完成")
        
        # 2. 执行完整的专业筛选流程（包含正交化计划生成）
        selected_factors, selection_report = self.factor_selector.run_complete_selection(
            factor_names = candidate_factor_names, force_generate = force_generate_ic
        )
        
        if not selected_factors:
            raise ValueError("❌ 专业筛选未选出任何因子，无法进行合成")
        
        # 3. 获取正交化计划
        orthogonalization_plan = selection_report.get('correlation_control').get('orthogonalized_factors',[])
        logger.info(f"📋 获取到 {len(orthogonalization_plan)} 项正交化计划")
        
        # 4. 执行正交化改造
        config_manager = ConfigSnapshotManager()
        pool_index, start_date, end_date, config_evaluation = config_manager.get_snapshot_config_content_details(snap_config_id)
        
        orthogonal_factors = {}
        if orthogonalization_plan:
            orthogonal_factors = self.execute_orthogonalization_plan(
                orthogonalization_plan, pool_index, snap_config_id
            )
        
        # 5. 构建最终因子列表（原始筛选因子 + 正交化因子）
        final_factor_list = selected_factors.copy()
        
        # 替换被正交化的因子
        for plan_item in orthogonalization_plan:
            original_factor = plan_item['original_factor']
            orthogonal_name = plan_item['orthogonal_name']
            
            if original_factor in final_factor_list and orthogonal_name in orthogonal_factors:
                final_factor_list.remove(original_factor)
                final_factor_list.append(orthogonal_name)
                logger.info(f"🔄 因子替换: {original_factor} -> {orthogonal_name}")
        
        logger.info(f"🎯 最终因子列表: {len(final_factor_list)} 个因子")
        
        # 6. 基于最终因子列表计算IC权重（修正后的逻辑）
        factor_ic_stats = {}
        for factor_name in final_factor_list:
            try:
                # 检查是否为正交化因子
                is_orthogonal_factor = False
                original_factor = factor_name
                avg_r_squared = 0.0
                
                # 查找对应的正交化计划项
                for plan_item in orthogonalization_plan:
                    if plan_item['orthogonal_name'] == factor_name:
                        is_orthogonal_factor = True
                        original_factor = plan_item['original_factor']
                        avg_r_squared = plan_item.get('avg_r_squared', 0.0)
                        break
                
                # 加载原始因子的IC统计
                ic_stats = self._load_factor_ic_stats(
                    original_factor, pool_index, snap_config_id=snap_config_id
                )
                
                if ic_stats:
                    if is_orthogonal_factor and avg_r_squared > 0:
                        # 🎯 核心修正：基于R²调整正交化因子的IC统计
                        logger.info(f"  🔧 正交化因子IC调整: {factor_name}")
                        adjusted_ic_stats = self._adjust_ic_stats_by_r_squared(
                            ic_stats, avg_r_squared, factor_name
                        )
                        factor_ic_stats[factor_name] = adjusted_ic_stats
                    else:
                        # 原始因子直接使用
                        factor_ic_stats[factor_name] = ic_stats
                        logger.debug(f"  📊 原始因子: {factor_name}")
                    
            except Exception as e:
                logger.error(f"  ❌ {factor_name}: IC统计处理异常 - {e}")
        
        # 7. 计算最终权重
        if factor_ic_stats:
            factor_weights = self.weight_calculator.calculate_ic_based_weights(factor_ic_stats)
        else:
            raise ValueError("⚠️ 无法获取IC统计，使用等权重合成")
            # equal_weight = 1.0 / len(final_factor_list)
            # factor_weights = {name: equal_weight for name in final_factor_list}
        
        # 8. 执行加权合成（支持正交化因子）
        composite_factor_df = self._execute_weighted_synthesis_with_orthogonal(
            composite_factor_name, pool_index, factor_weights, orthogonal_factors, snap_config_id
        )
        
        # 9. 生成综合报告
        synthesis_report = self._generate_orthogonalization_report(
            composite_factor_name,
            candidate_factor_names,
            selected_factors,
            final_factor_list,
            factor_weights,
            orthogonalization_plan,
            selection_report
        )
        
        logger.info(f"✅ 带正交化的专业因子合成完成: {composite_factor_name}")
        return composite_factor_df, synthesis_report

    def _execute_weighted_synthesis_with_orthogonal(
            self,
            composite_factor_name: str,
            stock_pool_index_name: str,
            factor_weights: Dict[str, float],
            orthogonal_factors: Dict[str, pd.DataFrame],
            snap_config_id: str
    ) -> pd.DataFrame:
        """
        支持正交化因子的加权合成
        
        Args:
            composite_factor_name: 复合因子名称
            stock_pool_index_name: 股票池名称
            factor_weights: 因子权重
            orthogonal_factors: 正交化因子数据 {name: df}
            snap_config_id: 配置快照ID
            
        Returns:
            合成后的因子DataFrame
        """
        logger.info(f"⚖️ 开始执行支持正交化的加权合成，使用{len(factor_weights)}个因子")
        
        processed_factors = []
        weights_list = []
        
        for factor_name, weight in factor_weights.items():
            logger.info(f"  🔄 处理因子: {factor_name} (权重: {weight:.3f})")
            
            # 检查是否为正交化因子
            if factor_name in orthogonal_factors:
                logger.debug(f"    📐 使用正交化因子数据: {factor_name}")
                processed_df = orthogonal_factors[factor_name]
            else:
                logger.debug(f"    📊 从本地加载原始因子: {factor_name}")
                processed_df = self.get_sub_factor_df_from_local(factor_name, stock_pool_index_name, snap_config_id)
            
            if processed_df is not None and not processed_df.empty:
                # --- 核心逻辑：因子方向统一 ---
                # 从IC统计数据中获取该因子的IC均值
                direction= get_new_factor_direction(factor_name)
                if direction < 0:
                    logger.info(f"    ⬇️  检测到负向因子 {factor_name}，进行方向翻转 * -1")
                    processed_df *= -1
                else:
                    logger.debug(f"    ⬆️  正向因子{factor_name}，无需处理")
                # --- 方向统一结束 ---
                processed_factors.append(processed_df)
                weights_list.append(weight)
            else:
                logger.warning(f"    ⚠️ 因子数据无效，跳过: {factor_name}")
        
        if not processed_factors:
            raise ValueError("没有任何因子被成功处理")
        
        # 加权合成
        composite_factor_df = self._weighted_combine_factors(processed_factors, weights_list)
        
        # 最终标准化
        composite_factor_df = self.processor._standardize_robust(composite_factor_df)
        
        logger.info(f"✅ 支持正交化的加权合成完成: {composite_factor_name}")
        return composite_factor_df

    def _generate_orthogonalization_report(
            self,
            composite_factor_name: str,
            candidate_factors: List[str],
            selected_factors: List[str],
            final_factor_list: List[str],
            factor_weights: Dict[str, float],
            orthogonalization_plan: List[Dict],
            selection_report: Dict
    ) -> Dict:
        """
        生成包含正交化信息的综合报告
        
        Args:
            composite_factor_name: 复合因子名称
            candidate_factors: 候选因子列表
            selected_factors: 初步筛选因子列表
            final_factor_list: 最终因子列表（经正交化处理后）
            factor_weights: 最终权重
            orthogonalization_plan: 正交化计划
            selection_report: 筛选报告
            
        Returns:
            综合报告
        """
        # 基础报告
        base_report = self._generate_comprehensive_report(
            composite_factor_name, candidate_factors, final_factor_list, factor_weights, selection_report
        )
        
        # 添加正交化信息
        orthogonalization_info = {
            'orthogonalization_enabled': True,
            'orthogonalization_plan_count': len(orthogonalization_plan),
            'orthogonalization_details': []
        }
        
        for plan_item in orthogonalization_plan:
            orthogonalization_info['orthogonalization_details'].append({
                'original_factor': plan_item['original_factor'],
                'base_factor': plan_item['base_factor'],
                'orthogonal_name': plan_item['orthogonal_name'],
                'original_correlation': plan_item.get('correlation', 0),
                'base_score': plan_item.get('base_score', 0),
                'target_score': plan_item.get('target_score', 0)
            })
        
        # 合并报告
        comprehensive_report = {
            **base_report,
            'orthogonalization': orthogonalization_info,
            'factor_transformation_summary': {
                'initial_selected_count': len(selected_factors),
                'orthogonalized_count': len(orthogonalization_plan),
                'final_factor_count': len(final_factor_list),
                'replacement_mapping': {
                    plan['original_factor']: plan['orthogonal_name']
                    for plan in orthogonalization_plan
                }
            }
        }
        
        return comprehensive_report

if __name__ == '__main__':
    data_manager = DataManager()
    factor_manager= FactorManager(data_manager)
    from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer

    factor_analyzer = FactorAnalyzer(factor_manager)
    factor_processor = FactorProcessor(factor_manager.data_manager.config)
    iCWeightedSynthesizer = (ICWeightedSynthesizer(factor_manager, factor_analyzer, factor_processor))
    ret  = {
        'log_circ_mv': iCWeightedSynthesizer._load_factor_ic_stats('log_circ_mv', '000906',snap_config_id = '20250906_045625_05e460ab')
    }
    iCWeightedSynthesizer.weight_calculator.calculate_ic_based_weights(ret)

    # (ICWeightedSynthesizer(factor_manager, factor_analyzer, factor_processor).synthesize_with_orthogonalization
    #  (composite_factor_name='composite_factor_name',candidate_factor_names=['volatility_40d','sp_ratio','earnings_stability','cfp_ratio','ep_ratio']
    #   ,snap_config_id= '20250825_091622_98ed2d08',force_generate_ic=False))

    ##todo 合成好的因子在进入 ic测试!! 直接用本地的close数据就行