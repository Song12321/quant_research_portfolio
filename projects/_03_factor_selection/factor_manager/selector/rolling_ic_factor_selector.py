"""
基于滚动IC的专业因子筛选器

核心原则：
1. 坚决使用滚动IC，彻底杜绝前视偏差
2. 多周期IC评分，指数衰减权重
3. 严格的时间序列稳定性验证
4. 为实盘稳定盈利策略服务


Date: 2025-08-25
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

from projects._03_factor_selection.factor_manager.ic_manager.rolling_ic_manager import run_cal_and_save_rolling_ic_by_snapshot_config_id
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import ConfigSnapshotManager
from quant_lib.config.logger_config import setup_logger

# 层次聚类相关导入
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

logger = setup_logger(__name__)

@dataclass 
class RollingICSelectionConfig:
    """滚动IC筛选配置"""
    # 基本筛选门槛
    min_snapshots: int = 3           # 最少快照数量
    min_ic_abs_mean: float = 0.01    # 滚动IC均值绝对值门槛
    min_ir_abs_mean: float = 0.15    # 滚动IR均值绝对值门槛
    min_ic_stability: float = 0.4    # IC稳定性门槛（方向一致性）
    max_ic_volatility: float = 0.05  # IC波动率上限
    
    # 多周期权重配置
    decay_rate: float = 0.75         # 衰减率，越小权重衰减越慢
    prefer_short_term: bool = True   # 偏向短期
    
    # 类别内选择
    max_factors_per_category: int = 3  # 每类最多因子数
    min_category_score: float = 10.0   # 类别最低评分
    
    # 最终筛选
    max_final_factors: int = 8         # 最多选择因子数
    
    # 相关性控制（三层决策哲学）
    high_corr_threshold: float = 0.7   # 高相关阈值（红色警报：二选一）
    medium_corr_threshold: float = 0.3 # 中低相关分界（黄色预警：正交化战场）
    enable_orthogonalization: bool = True  # 是否启用中相关区间正交化
    
    # 层次聚类配置
    clustering_method: str = 'graph'   # 聚类方法: 'graph'(图算法) 或 'hierarchical'(层次聚类)
    hierarchical_distance_threshold: float = 0.3  # 层次聚类距离阈值
    hierarchical_linkage_method: str = 'ward'  # 连接方法: 'ward', 'complete', 'average'
    max_clusters: int = None  # 最大簇数量限制 (None表示使用距离阈值)
    
    # 实盘交易成本控制（换手率一等公民）
    max_turnover_rate: float = 0.15    # 最大换手率阈值（月度）
    turnover_weight: float = 0.25      # 换手率在综合评分中的权重
    enable_turnover_penalty: bool = True  # 是否启用换手率惩罚

    # 1. 基础乘数相关配置
    reward_turnover_rate_daily: float = 0.0025
    max_turnover_rate_daily: float = 0.007
    penalty_slope_daily: float = 45.0
    heavy_penalty_slope_daily: float = 100.0
    base_turnover_multiplier_floor: float = 0.1  # 【新增】基础乘数的最低值，防止变为负数

    # 2. 波动率惩罚相关配置
    turnover_vol_threshold_ratio: float = 0.5
    turnover_vol_penalty_factor: float = 0.2

    # 3. 趋势惩罚相关配置
    turnover_trend_sensitivity: float = 50.0  # 【新增】趋势惩罚敏感度, 取代了旧的*100

    # 4. 最终乘数范围控制
    final_multiplier_min: float = 0.1  # 【新增】最终乘数下限
    final_multiplier_max: float = 1.2  # 【新增】最终乘数上限
    # 用于硬性淘汰的最终防线 (Final Gatekeeper Thresholds)
    max_turnover_mean_daily: float = 0.15    # 硬门槛：日均换手率不得超过1% (约等于月度21%)
    max_turnover_trend_daily: float = 0.00005 # 硬门槛：换手率每日恶化趋势不得超过0.002%
    max_turnover_vol_daily: float = 0.015     # 硬门槛：换手率波动率不得超过1.5%


@dataclass
class FactorRollingICStats:
    """因子滚动IC统计数据"""
    factor_name: str
    periods_data: Dict[str, Dict]  # 各周期数据
    avg_ic_with_sign: float #带符号
    avg_ir_ir_with_sign: float
    avg_ic_abs: float              # 平均IC绝对值
    avg_ir_abs: float              # 平均IR绝对值
    best_period_ic_ir:float  #ir所在 表现最佳的周期val
    nw_t_stat_series_mean:float
    avg_stability: float           # 平均稳定性
    avg_ic_volatility: float       # 平均IC波动率
    multi_period_score: float      # 多周期综合评分
    snapshot_count: int            # 快照数量
    time_range: Tuple[str, str]    # 时间范围
    
    # 实盘交易成本控制
    # avg_daily_rank_change: float = 0.0    # 平均月度换手率
    daily_rank_change_mean:float
    daily_turnover_trend:float
    daily_turnover_volatility:float
    turnover_adjusted_score: float = 0.0  # 换手率调整后评分
    

class RollingICFactorSelector:
    """基于滚动IC的专业因子筛选器"""
    
    def __init__(self, snap_config_id: str, config: Optional[RollingICSelectionConfig] = None):
        """
        初始化滚动IC因子筛选器
        
        Args:
            snap_config_id: 配置快照ID
            config: 筛选配置，如果为None使用默认配置
        """
        self.snap_config_id = snap_config_id
        self.config = config or RollingICSelectionConfig()
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result")
        
        # 从配置快照获取基础信息
        self._load_config_info()
        
        # 因子分类定义 - 完整版本
        self.factor_categories = {
            'Value': ['bm_ratio', 'ep_ratio', 'cfp_ratio', 'sp_ratio', 'value_composite', 'pb_ratio', 'pe_ttm', 'ps_ratio'],
            'Quality': ['roe_ttm', 'gross_margin_ttm', 'debt_to_assets', 'earnings_stability', 'quality_momentum', 
                       'operating_accruals', 'asset_turnover', 'roa_ttm', 'current_ratio'],
            'Momentum': ['momentum_20d', 'momentum_120d', 'momentum_12_1', 'momentum_pct_60d', 'sharpe_momentum_60d', 
                        'sw_l1_momentum_21d', 'momentum_6_1', 'momentum_3_1'],
            'Reversal': ['reversal_5d', 'reversal_21d', 'reversal_1d', 'reversal_10d'],
            'Size': ['log_circ_mv', 'log_total_mv', 'market_cap_weight'],
            'Volatility': ['volatility_40d', 'volatility_90d', 'volatility_120d', 'rsi', 'atr_20d',
                          'volatility_40d_经过残差化', 'volatility_90d_经过残差化', 'volatility_120d_经过残差化', 'rsi_经过残差化'],
            'Liquidity': ['amihud_liquidity', 'turnover_rate_90d_mean', 'turnover_rate_monthly_mean', 'ln_turnover_value_90d', 
                         'turnover_t1_div_t20d_avg', 'bid_ask_spread', 'turnover_rate_90d_mean-经过残差化', 
                         'turnover_rate_monthly_mean_经过残差化', 'ln_turnover_value_90d_经过残差化'],
            'Technical': ['cci', 'pead', 'macd', 'rsi_divergence', 'cci_经过残差化', 'bollinger_position'],
            'Growth': ['total_revenue_growth_yoy', 'net_profit_growth_yoy', 'eps_growth', 'operating_revenue_growth'],
            'Profitability': ['gross_profit_margin', 'operating_margin', 'net_margin', 'ebit_margin'],
            'Efficiency': ['inventory_turnover', 'receivables_turnover', 'working_capital_turnover']
        }
        
        # 缓存数据
        self._factor_stats_cache = {}
        
        logger.info(f"滚动IC因子筛选器初始化完成")
        logger.info(f"配置ID: {self.snap_config_id}")
        logger.info(f"股票池: {self.pool_index}")
        logger.info(f"时间范围: {self.start_date} - {self.end_date}")
        logger.info(f"数据版本: {self.version}")
    
    def _load_config_info(self):
        """加载配置信息"""
        config_manager = ConfigSnapshotManager()
        self.pool_index, self.start_date, self.end_date, self.config_evaluation = config_manager.get_snapshot_config_content_details(self.snap_config_id)
        self.version = f"{self.start_date}_{self.end_date}"
        self.forward_periods = self.config_evaluation.get('forward_periods', ['21'])
    
    def extract_factor_rolling_ic_stats(self, factor_name: str, force_generate: bool = False) -> Optional[FactorRollingICStats]:
        """
        提取单个因子的滚动IC统计数据
        
        Args:
            factor_name: 因子名称
            force_generate: 是否强制重新生成滚动IC数据
            
        Returns:
            FactorRollingICStats or None
        """
        # 检查缓存
        if not force_generate and factor_name in self._factor_stats_cache:
            return self._factor_stats_cache[factor_name]
        
        # 构建数据路径
        rolling_ic_dir = (self.main_work_path / self.pool_index / factor_name / 
                         'o2o' / self.version / 'rolling_ic')
        
        # 如果目录不存在，尝试生成
        if not rolling_ic_dir.exists() or force_generate:
            logger.info(f"为因子 {factor_name} 生成滚动IC数据...")
            try:
                run_cal_and_save_rolling_ic_by_snapshot_config_id(self.snap_config_id, [factor_name])
                logger.info(f"因子 {factor_name} 滚动IC数据生成成功")
            except Exception as e:
                raise ValueError(f"生成滚动IC数据失败 {factor_name}: {e}")

        # 检查文件
        ic_files = list(rolling_ic_dir.glob("ic_snapshot_*.json"))
        if not ic_files:
            raise ValueError(f"未找到因子 {factor_name} 的滚动IC文件")

        
        if len(ic_files) < self.config.min_snapshots:
            raise ValueError(f"因子 {factor_name} 滚动IC快照数量不足: {len(ic_files)} < {self.config.min_snapshots}")

        
        # 解析数据
        periods_data = {}
        dates_range = []
        
        for ic_file in ic_files:
            try:
                with open(ic_file, 'r', encoding='utf-8') as f:
                    snapshot = json.load(f)

                calc_date = snapshot['calculation_date']
                dates_range.append(calc_date)
                ic_stats_snap = snapshot.get('ic_stats', {})

                for period, stats in ic_stats_snap.items():
                    if period not in periods_data:
                        periods_data[period] = []
                    periods_data[period].append({
                        'date': calc_date,
                        'ic_mean': stats.get('ic_mean', 0),#底层是ewma来的
                        'ic_ir': stats.get('ic_ir', 0),#底层是ewma来的
                        'ic_win_rate': stats.get('ic_win_rate', 0.5),#底层是ewma来的
                        'avg_daily_rank_change_stats': stats.get('avg_daily_rank_change_stats'),
                        'ic_std': stats.get('ic_std', 0),
                        'ic_t_stat': stats.get('ic_t_stat', 0),
                        'ic_nw_t_stat': stats.get('ic_nw_t_stat', 0),
                        'ic_nw_p_value': stats.get('ic_nw_p_value', 1.0)#底层是Newey-West T-stat
                    })


            except Exception as e:
                raise ValueError(f"读取IC快照文件 {ic_file} 失败: {e}")

        
        if not periods_data:
            raise ValueError(f"因子 {factor_name} 无有效的滚动IC数据")

        
        # 计算统计指标 periods_data：存每个月底的ic数据 便于后续统aver
        ## periods_data 内容
        # 11:[1月31的快照ic数据，2月31的快照ic数据，3月31的数据..]
        # 5d:[1月31的快照ic数据，2月31的快照ic数据，3月31的数据..]
        # #
        factor_stats = self._calculate_factor_statistics(factor_name, periods_data, dates_range)
        
        # 缓存结果
        self._factor_stats_cache[factor_name] = factor_stats
        
        return factor_stats
    
    def _calculate_factor_statistics(self, factor_name: str, periods_data: Dict, dates_range: List[str]) -> FactorRollingICStats:
        """计算因子统计指标"""
        
        # 汇总各周期统计
        aggregated_periods = {}
        all_ic_means = []
        all_ic_irs = []
        all_stabilities = []
        all_ic_stds = []
        
        for period, time_series in periods_data.items():
            if len(time_series) < self.config.min_snapshots:
                continue
            
            # 提取时间序列数据
            ic_means = [d['ic_mean'] for d in time_series]
            ic_irs = [d['ic_ir'] for d in time_series]
            ic_win_rates = [d['ic_win_rate'] for d in time_series]
            ic_stds = [d['ic_std'] for d in time_series]
            nw_t_stat_series = [d['ic_nw_t_stat'] for d in time_series]

            # 计算统计指标 求（1月31快照ic数据...+n月快照数据）/n 平均
            avg_ic_period = np.mean(ic_means)
            avg_ir_period = np.mean(ic_irs)
            avg_win_rate_period = np.mean(ic_win_rates)
            ic_volatility_period = np.std(ic_means)
            nw_t_stat_series_mean = float(np.mean(nw_t_stat_series))

            # IC方向一致性（稳定性）
            if len(ic_means) > 1:
                # 核心思想：稳定性，是指“滚动的IC符号”与这段时期的“平均IC符号”是否一致
                # 1. 确定这个周期的“期望方向”，即IC均值的符号
                expected_sign = np.sign(avg_ic_period)

                # 2. 处理均值为0的罕见情况, 默认为正向
                if expected_sign == 0:
                    expected_sign = 1

                # 3. 计算有多少滚动IC的符号与“期望方向”一致
                #    这里我们用 np.sign 来处理，比 (ic > 0) 更严谨，可以正确处理ic为0的情况
                num_consistent = sum(1 for ic in ic_means if np.sign(ic) == expected_sign)
                stability = num_consistent / len(ic_means)
            else:
                stability = 1.0
            
            aggregated_periods[period] = {
                'ic_mean_avg': avg_ic_period,
                'ic_ir_avg': avg_ir_period,
                'ic_win_rate_avg': avg_win_rate_period,
                'ic_volatility_period': ic_volatility_period,
                'ic_stability': stability,#方向一致性
                'sample_count': len(time_series),
                'nw_t_stat_series_mean':nw_t_stat_series_mean,
                'time_series': time_series
            }
            
            # 收集全局统计
            all_ic_means.append(avg_ic_period)
            all_ic_irs.append(avg_ir_period)
            all_stabilities.append(stability)
            all_ic_stds.append(ic_volatility_period)

        # 周期加权 计算出平均ic
        ic_means_with_sign = [aggregated_periods[p]['ic_mean_avg'] for p in periods_data.keys()]
        ic_irs_with_sign = [aggregated_periods[p]['ic_ir_avg'] for p in periods_data.keys()]

        decay_rate = self.config.decay_rate
        weights = np.array([decay_rate ** i for i in range(len(periods_data.keys()))])
        weights /= weights.sum()

        # 得到最核心的两个综合指标
        avg_ic_with_sign = float(np.average(ic_means_with_sign, weights=weights))
        avg_ic_ir_with_sign = float(np.average(ic_irs_with_sign, weights=weights))
        best_period_ic_ir = ic_irs_with_sign.max()

        # 3. 从综合指标派生出用于筛选的绝对值指标
        avg_ic_abs = abs(avg_ic_with_sign)
        avg_ir_abs = abs(avg_ic_ir_with_sign)

        # 选择一个参考周期 (通常选择最短的，数据最完整) 截断少
        #      我们对periods_data的键（也就是周期）进行数字排序来找到最短的那个
        reference_period = sorted(periods_data.keys())[0]
        # 4.2. 从参考周期中提取完整的快照时间序列 (60个快照的列表)
        reference_time_series = [snap['avg_daily_rank_change_stats'] for snap in periods_data[reference_period]]

        # 4.3. 提取三个核心指标各自的时间序列
        #      使用 .get() 来安全地获取值，以防某个快照数据缺失
        avg_daily_rank_change_series  = [d.get('avg_daily_rank_change', 0) for d in reference_time_series]
        daily_turnover_volatility_series = [d.get('daily_turnover_volatility', 0) for d in reference_time_series]
        daily_turnover_trend_series = [d.get('daily_turnover_trend', 0) for d in reference_time_series]
        # 4.4. 计算整个五年期间的总平均统计值
        final_avg_change = float(np.mean(avg_daily_rank_change_series))
        final_avg_vol = float(np.mean(daily_turnover_volatility_series))  # 对波动率求均值，衡量平均不确定性
        final_avg_trend = float(np.mean(daily_turnover_trend_series))  # 对趋势求均值，衡量长期衰减倾向
        # if(IS_DEBUG_TEMP and (factor_name in ['turnover_rate_monthly_mean','volatility_40d'])):
        #     final_avg_change = 0.01
        #     final_avg_vol = 0.01

        # 4.5. 组装成最终的统计字典，用于评分函数
        final_turnover_stats = {
            'avg_daily_rank_change': final_avg_change,
            'daily_turnover_volatility': final_avg_vol,
            'daily_turnover_trend': final_avg_trend
        }

        # 计算多周期综合评分
        multi_period_score = self._calculate_multi_period_score(aggregated_periods)
        
        # 计算换手率调整后评分（实盘导向）
        turnover_adjusted_score = self._calculate_turnover_adjusted_score(
            multi_period_score, final_turnover_stats
        )
        
        factor_stats = FactorRollingICStats(
            factor_name=factor_name,
            periods_data=aggregated_periods,
            avg_ic_with_sign=avg_ic_with_sign,
            avg_ir_ir_with_sign=avg_ic_ir_with_sign,
            avg_ic_abs= avg_ic_abs,
            avg_ir_abs=avg_ir_abs,
            nw_t_stat_series_mean=nw_t_stat_series_mean,
            avg_stability=np.mean(all_stabilities) if all_stabilities else 0.0,
            avg_ic_volatility=np.mean(all_ic_stds) if all_ic_stds else 0.0,
            multi_period_score=multi_period_score,
            snapshot_count=len(dates_range),
            time_range=(min(dates_range), max(dates_range)) if dates_range else ('', ''),
            # 将三个核心换手率指标填入返回结构
            daily_rank_change_mean=final_turnover_stats['avg_daily_rank_change'],
            daily_turnover_trend=final_turnover_stats['daily_turnover_trend'],
            daily_turnover_volatility=final_turnover_stats['daily_turnover_volatility'],
            turnover_adjusted_score=turnover_adjusted_score
        )
        # 构建结果

        return factor_stats

    def _calculate_multi_period_score(self, periods_data: Dict) -> float:
        """
        计算多周期IC综合评分（带指数衰减权重）

        Args:
            periods_data: 多周期数据 {period: stats}

        Returns:
            float: 综合评分
        """
        if not periods_data:
            return 0.0

        # 按周期排序（短期到长期）
        try:
            periods = sorted(periods_data.keys(), key=lambda x: int(x.replace('d', '').replace('D', '')))
        except:
            periods = sorted(periods_data.keys())

        # 计算每个周期的得分
        period_scores = []

        # --- 核心改造：定义“满分标杆” ---
        IC_MEAN_BENCHMARK = 0.05  # IC均值达到0.05，我们认为表现优异
        IC_IR_BENCHMARK = 0.50  # IR达到0.5，我们认为稳定性优异
        IC_STABILITY_BENCHMARK = 1.0  # 稳定性是0-1，标杆就是1.0
        IC_WIN_RATE_BENCHMARK = 0.60  # 胜率达到60%，我们认为很不错
        IC_VOL_PENALTY_BASE = 0.02  # IC波动率超过2%开始惩罚

        # --- 核心改造：定义最终的权重配比 ---
        # 这些权重现在是真实的、可比的
        WEIGHTS = {
            'ic_mean': 0.40,  # 40% 权重给效果
            'ic_ir': 0.30,  # 30% 权重给稳定性
            'ic_stability': 0.10,  # 10% 权重给另一种稳定性
            'ic_win_rate': 0.20  # 20% 权重给胜率
        }

        for period in periods:
            stats = periods_data[period]

            # --- 核心改造：先归一化，得到0-1之间的分数 ---
            ic_norm_score = min(abs(stats.get('ic_mean_avg', 0)) / IC_MEAN_BENCHMARK, 1.0)
            ir_norm_score = min(abs(stats.get('ic_ir_avg', 0)) / IC_IR_BENCHMARK, 1.0)
            stability_norm_score = min(stats.get('ic_stability', 0) / IC_STABILITY_BENCHMARK, 1.0)

            # 胜率以50%为基准
            win_rate_norm_score = max(0, (stats.get('ic_win_rate_avg', 0.5) - 0.5) / (IC_WIN_RATE_BENCHMARK - 0.5))
            win_rate_norm_score = min(win_rate_norm_score, 1.0)

            # 惩罚项
            volatility_penalty = max(0, (stats.get('ic_volatility', 0) - IC_VOL_PENALTY_BASE) * 20)  # 惩罚力度可以调整

            # --- 核心改造：再加权 ---
            # 现在，所有分数都在0-1范围，权重可以公平地发挥作用
            weighted_score = (ic_norm_score * WEIGHTS['ic_mean'] +
                              ir_norm_score * WEIGHTS['ic_ir'] +
                              stability_norm_score * WEIGHTS['ic_stability'] +
                              win_rate_norm_score * WEIGHTS['ic_win_rate'])

            # 应用惩罚并确保分数在0-100之间 (乘以100方便阅读)
            total_score = (weighted_score - volatility_penalty) * 100
            period_scores.append(max(0, total_score))

        if not period_scores:
            return 0.0

        # 应用指数衰减权重（短期权重更高）
        decay_rate = self.config.decay_rate
        weights = np.array([decay_rate ** i for i in range(len(period_scores))])
        weights /= weights.sum()  # 权重归一化

        # 计算加权平均分数
        final_score = np.average(period_scores, weights=weights)

        return final_score

    def _estimate_factor_turnover(self, factor_name: str, periods_data: Dict) -> float:
        """
        估算因子换手率（实盘交易成本核心指标）
        
        Args:
            factor_name: 因子名称
            periods_data: 各周期数据
            
        Returns:
            float: 月度平均换手率估算
        """
        try:
            # 根据因子类型估算换手率（基于经验和研究）
            turnover_estimates = {
                # 高频类因子（技术面）
                'reversal_1d': 0.30, 'reversal_5d': 0.25, 'reversal_10d': 0.20,
                'momentum_20d': 0.18, 'rsi': 0.22, 'cci': 0.24,
                'macd': 0.20, 'bollinger_position': 0.28,
                
                # 中频类因子（价量结合）
                'momentum_60d': 0.15, 'momentum_120d': 0.12, 'momentum_12_1': 0.10,
                'volatility_40d': 0.16, 'volatility_90d': 0.14, 'volatility_120d': 0.12,
                'amihud_liquidity': 0.14, 'turnover_rate_90d_mean': 0.16,
                
                # 低频类因子（基本面）
                'ep_ratio': 0.08, 'bm_ratio': 0.07, 'sp_ratio': 0.08, 'cfp_ratio': 0.09,
                'roe_ttm': 0.06, 'gross_margin_ttm': 0.05, 'earnings_stability': 0.04,
                'total_revenue_growth_yoy': 0.07, 'net_profit_growth_yoy': 0.08,
                
                # 规模因子（极低频）
                'log_circ_mv': 0.03, 'log_total_mv': 0.03, 'market_cap_weight': 0.02,
                
                # 质量因子（低频）
                'debt_to_assets': 0.05, 'current_ratio': 0.04, 'asset_turnover': 0.06,
                'quality_momentum': 0.09
            }
            
            # 基础换手率估算
            base_turnover = turnover_estimates.get(factor_name, 0.12)  # 默认12%
            
            # 根据IC稳定性调整（稳定性低的因子通常换手率更高）
            if periods_data:
                avg_stability = np.mean([
                    stats.get('ic_stability', 0.5) 
                    for stats in periods_data.values()
                ])
                # 稳定性越低->avg_stability越低->stability_adjustment越大-》adjusted_turnover越大==换手率调整系数越高
                stability_adjustment = 1.0 + (0.5 - avg_stability) * 0.8
                adjusted_turnover = base_turnover * stability_adjustment
            else:
                adjusted_turnover = base_turnover
            
            # 换手率合理范围控制
            final_turnover = np.clip(adjusted_turnover, 0.02, 0.50)
            
            return final_turnover
            
        except Exception as e:
            logger.debug(f"换手率估算失败 {factor_name}: {e}")
            return 0.12  # 默认换手率12%
    #单侧通过
    def _calculate_turnover_adjusted_score(self, base_score: float, turnover_stats: Dict) -> float:
        """
        计算基于多维度换手率指标的调整后评分 (V3 - 最终生产版)

        此版本经过严格审查和加固，解决了中间值保护、趋势惩罚敏感度、
        分数符号保留和数值稳定性等问题，符合实盘生产要求。

        Args:
            base_score: 基础IC评分 (可能为负)
            turnover_stats: 来自 _calculate_daily_rank_change 的完整统计字典

        Returns:
            float: 换手率调整后评分，保留原始base_score的符号
        """
        if not self.config.enable_turnover_penalty:
            return base_score

        # 使用一个极小值来保证数值稳定性
        epsilon = 1e-8

        # --- 1. 基础乘数 (基于换手率均值) ---
        avg_daily_rank_change = turnover_stats.get('avg_daily_rank_change', 0.01)

        reward_rate_daily = self.config.reward_turnover_rate_daily
        max_rate_daily = self.config.max_turnover_rate_daily
        penalty_slope = self.config.penalty_slope_daily
        heavy_penalty_slope = self.config.heavy_penalty_slope_daily

        if avg_daily_rank_change <= reward_rate_daily:
            base_turnover_multiplier = 1.0 + (avg_daily_rank_change / (reward_rate_daily + epsilon)) * 0.1
        elif avg_daily_rank_change <= max_rate_daily:
            base_turnover_multiplier = 1.1 - (avg_daily_rank_change - reward_rate_daily) * penalty_slope
        else:
            boundary_multiplier = 1.1 - (max_rate_daily - reward_rate_daily) * penalty_slope
            excess_turnover = avg_daily_rank_change - max_rate_daily
            base_turnover_multiplier = boundary_multiplier - excess_turnover * heavy_penalty_slope

        # 【V3 核心改进】对基础乘数本身进行数值保护，防止其变为负或过小
        base_turnover_multiplier = max(base_turnover_multiplier, self.config.base_turnover_multiplier_floor)

        # --- 2. 波动率惩罚乘数 ---
        volatility = turnover_stats.get('daily_turnover_volatility', 0)
        volatility_threshold_ratio = self.config.turnover_vol_threshold_ratio
        volatility_penalty_factor = self.config.turnover_vol_penalty_factor

        volatility_penalty_multiplier = 1.0

        ratio = volatility / (avg_daily_rank_change + epsilon)
        if ratio > volatility_threshold_ratio:
            excess_ratio = ratio - volatility_threshold_ratio
            penalty = excess_ratio * volatility_penalty_factor
            volatility_penalty_multiplier = max(0.8, 1.0 - penalty)  # 惩罚下限0.8保持不变

        # --- 3. 趋势惩罚乘数 ---
        trend = turnover_stats.get('daily_turnover_trend', 0)
        trend_penalty_multiplier = 1.0

        if trend > 0:
            relative_trend = trend / (avg_daily_rank_change + epsilon)
            sensitivity = self.config.turnover_trend_sensitivity

            # 【V3 核心改进】移除了 *100 的硬编码，使用更灵活的敏感度参数
            trend_penalty_multiplier = np.exp(-relative_trend * sensitivity)
            trend_penalty_multiplier = max(0.7, trend_penalty_multiplier)  # 惩罚下限0.7保持不变

        # === 4. 最终计算 ===
        total_turnover_multiplier = base_turnover_multiplier * volatility_penalty_multiplier * trend_penalty_multiplier

        weight = self.config.turnover_weight
        final_multiplier = (1 - weight) + weight * total_turnover_multiplier

        # 使用可配置的上下限进行最终裁剪
        final_multiplier = np.clip(
            final_multiplier,
            self.config.final_multiplier_min,
            self.config.final_multiplier_max
        )

        adjusted_score = base_score * final_multiplier

        # 【V3 核心改进】移除 max(0.0, ...)，保留分数的原始符号
        logger.info(f"final_multiplier:{final_multiplier} total_turnover_multiplier:{total_turnover_multiplier}")
        return adjusted_score

    def screen_factors_by_rolling_ic(self, factor_names: List[str], force_generate: bool = False) -> Dict[str, FactorRollingICStats]:
        """
        基于滚动IC筛选因子
        
        Args:
            factor_names: 候选因子列表
            force_generate: 是否强制重新生成滚动IC
            
        Returns:
            Dict[factor_name, FactorRollingICStats]: 通过筛选的因子
        """
        logger.info(f"开始基于滚动IC筛选 {len(factor_names)} 个因子...")
        
        qualified_factors = {}
        
        for i, factor_name in enumerate(factor_names, 1):
            logger.info(f"处理因子 {i}/{len(factor_names)}: {factor_name}")
            
            try:
                # 提取因子统计
                factor_stats = self.extract_factor_rolling_ic_stats(factor_name, force_generate)
                
                if factor_stats is None:
                    raise ValueError(f"因子 {factor_name}: 无法获取滚动IC统计")

                # 应用筛选条件
                passes_screening = self._evaluate_factor_quality(factor_stats)
                
                if passes_screening:
                    qualified_factors[factor_name] = factor_stats
                    direction = "+" if  np.sign(factor_stats.avg_ic_with_sign) > 0 else "-"
                    logger.info(f"  {direction} {factor_name}: 通过筛选")
                    logger.info(f"    IC={factor_stats.avg_ic_abs:.3f}, IR={factor_stats.avg_ir_abs:.2f}")
                    logger.info(f"    稳定性={factor_stats.avg_stability:.2f}, 日换手率={factor_stats.daily_rank_change_mean:.1%}")
                    logger.info(f"    基础评分={factor_stats.multi_period_score:.1f}, 调整评分={factor_stats.turnover_adjusted_score:.1f}")
                else:
                    logger.info(f"  - {factor_name}: 未通过筛选")
                    
            except Exception as e:
                raise ValueError(f"处理因子 {factor_name} 时出错: {e}")
                # continue
        
        logger.info(f"by滚动IC_筛选(ic（稳定、胜率）、周期、换手率)完成: {len(qualified_factors)}/{len(factor_names)} 个因子通过")
        return qualified_factors
    
    def _evaluate_factor_quality(self, factor_stats: FactorRollingICStats) -> bool:
        """
        评估因子质量是否通过筛选（实盘导向，换手率一等公民）
        """
        
        # 基本门槛检查
        basic_conditions = [
            factor_stats.avg_ic_abs >= self.config.min_ic_abs_mean,
            factor_stats.avg_ir_abs >= self.config.min_ir_abs_mean,
            factor_stats.avg_stability >= self.config.min_ic_stability,
            factor_stats.avg_ic_volatility <= self.config.max_ic_volatility,
            factor_stats.multi_period_score >= self.config.min_category_score,
            factor_stats.snapshot_count >= self.config.min_snapshots
        ]

        # 换手率门槛检查（实盘交易成本控制）
        turnover_condition = (
                not self.config.enable_turnover_penalty  or (
                # 硬门槛1: 平均换手率不能过高 ("简历关")
                factor_stats.daily_rank_change_mean <= self.config.max_turnover_mean_daily and

                # 硬门槛2: 换手率恶化趋势不能为正 ("面试关 - 重大风险项")
                factor_stats.daily_turnover_trend <= self.config.max_turnover_trend_daily and

                # 硬门槛3: 换手率波动率不能过高 ("背景调查关")
                factor_stats.daily_turnover_volatility <= self.config.max_turnover_vol_daily
        )
        )
        
        all_conditions = basic_conditions + [turnover_condition]
        
        # 记录详细的未通过原因（便于调试）
        if not all(all_conditions):
            failed_checks = []
            if factor_stats.avg_ic_abs < self.config.min_ic_abs_mean:
                failed_checks.append(f"IC均值过低({factor_stats.avg_ic_abs:.3f}<{self.config.min_ic_abs_mean})")
            if factor_stats.avg_ir_abs < self.config.min_ir_abs_mean:
                failed_checks.append(f"IR过低({factor_stats.avg_ir_abs:.2f}<{self.config.min_ir_abs_mean})")
            if factor_stats.avg_stability < self.config.min_ic_stability:
                failed_checks.append(f"稳定性不足({factor_stats.avg_stability:.2%}<{self.config.min_ic_stability:.0%})")
            if factor_stats.avg_ic_volatility > self.config.max_ic_volatility:
                failed_checks.append(f"IC波动过高({factor_stats.avg_ic_volatility:.3f}>{self.config.max_ic_volatility})")
            if factor_stats.multi_period_score < self.config.min_category_score:
                failed_checks.append(f"综合评分过低({factor_stats.multi_period_score:.1f}<{self.config.min_category_score})")
            if factor_stats.snapshot_count < self.config.min_snapshots:
                failed_checks.append(f"快照不足({factor_stats.snapshot_count}<{self.config.min_snapshots})")
            if (self.config.enable_turnover_penalty and 
                factor_stats.daily_rank_change_mean > self.config.max_turnover_mean_daily):
                failed_checks.append(f"日换手率过高({factor_stats.daily_rank_change_mean:.1%}>{self.config.max_turnover_mean_daily:.0%})")

            if (self.config.enable_turnover_penalty and
                    factor_stats.daily_turnover_trend > self.config.max_turnover_trend_daily):
                failed_checks.append(
                    f"换手率每日恶化趋势不得超过2%({factor_stats.daily_turnover_trend:.1%}>{self.config.max_turnover_trend_daily:.0%})")

            if (self.config.enable_turnover_penalty and
                    factor_stats.daily_turnover_volatility > self.config.max_turnover_vol_daily):
                failed_checks.append(
                    f"换手率波动率不能过高({factor_stats.daily_turnover_volatility:.3%}>{self.config.max_turnover_vol_daily:.1%})")

            logger.debug(f"因子 {factor_stats.factor_name} 未通过筛选: {'; '.join(failed_checks)}")

        return all(all_conditions)
    
    def select_category_champions(self, qualified_factors: Dict[str, FactorRollingICStats]) -> Dict[str, List[str]]:
        """
        类别内冠军选择
        
        Args:
            qualified_factors: 通过基本筛选的因子
             for :n个类别"
                类内排名逻辑：按换手率加权的周期衰减总ic分数
                每个类别只要2个
            
        Returns:
            Dict[category, List[factor_names]]: 各类别的冠军因子
        """
        logger.info("开始类别内冠军选择...")
        
        category_champions = {}
        #注意 遍历的是类别！，而不是因子，所以务必需要保证类别在config配置文件！
        for category, factor_list in self.factor_categories.items():
            # 找到该类别中的合格因子
            category_factors = {
                name: stats for name, stats in qualified_factors.items() 
                if name in factor_list
            }
            
            if not category_factors:
                continue
            
            # 按换手率调整后评分排序（实盘导向优化）
            sorted_factors = sorted(
                category_factors.items(), 
                key=lambda x: x[1].turnover_adjusted_score if self.config.enable_turnover_penalty else x[1].multi_period_score, 
                reverse=True
            )
            
            # 选择前N名
            max_count = min(len(sorted_factors), self.config.max_factors_per_category)
            champions = [name for name, _ in sorted_factors[:max_count]]
            
            if champions:
                category_champions[category] = champions
                logger.info(f"{category}: {len(champions)} 个冠军")
                for name in champions:
                    stats = qualified_factors[name]
                    direction = "+" if  np.sign(stats.avg_ic_with_sign) > 0 else "-"
                    score_used = stats.turnover_adjusted_score if self.config.enable_turnover_penalty else stats.multi_period_score
                    logger.info(f"  {direction} {name}: 调整评分={score_used:.1f} (日换手率={stats.daily_rank_change_mean:.1%})")
        
        return category_champions
    
    def apply_correlation_control(
            self, 
            candidate_factors: List[str],
            qualified_factors: Dict[str, FactorRollingICStats]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        应用三层相关性控制哲学（两阶段无顺序依赖版本）
        
        🎯 核心改进：消除顺序依赖性，确保结果唯一确定
        
        📊 两阶段架构：
        阶段1: 🚨 红色区域集群消杀 (|corr|>0.7) - 每个高相关集群只保留最强者
        阶段2: ⚠️ 黄色区域正交化处理 (0.3<|corr|<0.7) - 基于幸存者生成正交化计划
        
        Args:
            candidate_factors: 候选因子列表
            qualified_factors: 合格因子统计
            
        Returns:
            (final_factors, correlation_report)
        """
        logger.info("🔍 开始执行三层相关性控制（无顺序依赖版本）...")
        logger.info(f"📊 输入因子数量: {len(candidate_factors)}")
        
        # 计算因子相关性矩阵
        correlation_matrix = self._calculate_factor_correlations(candidate_factors)
        if correlation_matrix is None:
            logger.warning("⚠️ 无法计算相关性矩阵，跳过相关性控制")
            return candidate_factors, {}
        
        # === 阶段1：根据配置选择聚类方法 ===
        if self.config.clustering_method == 'hierarchical':
            logger.info("🔬 阶段1：层次聚类数据驱动分析...")
            red_zone_survivors, red_zone_decisions = self._process_clusters_hierarchical(
                candidate_factors, correlation_matrix, qualified_factors
            )
        else:#todo 对比看看 新方法结果一致不
            logger.info("🚨 阶段1：红色区域集群消杀...")
            red_zone_survivors, red_zone_decisions = self._process_red_zone_clusters(
                candidate_factors, correlation_matrix, qualified_factors
            )
        
        logger.info(f"  📈 集群消杀结果: {len(candidate_factors)} → {len(red_zone_survivors)}")
        
        # === 阶段2：黄色区域正交化处理 ===
        logger.info("⚠️ 阶段2：黄色区域正交化处理...")
        final_factors, orthogonalization_plan, yellow_zone_decisions = self._process_yellow_zone_orthogonalization(
            red_zone_survivors, qualified_factors
        )
        
        logger.info(f"  📊 正交化处理结果: {len(red_zone_survivors)} → {len(final_factors)} + {len(orthogonalization_plan)} 个正交化计划")
        
        # === 合并决策记录 ===
        all_decisions = red_zone_decisions + yellow_zone_decisions
        
        # 生成详细报告
        correlation_report = {
            'algorithm_version': '两阶段无顺序依赖版本',
            'input_count': len(candidate_factors),
            'red_zone_survivors_count': len(red_zone_survivors), 
            'final_count': len(final_factors),
            'orthogonalized_count': len(orthogonalization_plan),
            'decisions': all_decisions,
            'orthogonalized_factors': orthogonalization_plan,
            'correlation_matrix': correlation_matrix.to_dict(),
            'thresholds': {
                'high_corr': self.config.high_corr_threshold,
                'medium_corr': self.config.medium_corr_threshold
            },
            'processing_stages': {
                'stage1_red_zone': {
                    'input_count': len(candidate_factors),
                    'output_count': len(red_zone_survivors),
                    'decisions_count': len(red_zone_decisions)
                },
                'stage2_yellow_zone': {
                    'input_count': len(red_zone_survivors),
                    'output_count': len(final_factors),
                    'orthogonalization_count': len(orthogonalization_plan),
                    'decisions_count': len(yellow_zone_decisions)
                }
            }
        }
        
        logger.info("🎯 三层相关性控制完成:")
        logger.info(f"  📈 输入因子: {len(candidate_factors)}")
        logger.info(f"  🔥 红色区域幸存者: {len(red_zone_survivors)}")
        logger.info(f"  🏆 最终因子: {len(final_factors)}")
        logger.info(f"  🔄 正交化因子: {len(orthogonalization_plan)}")
        logger.info(f"  📊 总决策记录: {len(all_decisions)}")
        
        return final_factors, correlation_report

    def _calculate_factor_correlations(self, factor_names: List[str]) -> Optional[pd.DataFrame]:
        """计算因子间相关性矩阵（向量化高效版）"""
        """计算因子间相关性矩阵（内置配对对齐的最终版）"""
        try:
            # Step 1: 仅加载所有因子数据
            factor_data_dict = self._load_all_factor_data(factor_names)

            final_factor_names = list(factor_data_dict.keys())
            if len(final_factor_names) < 2:
                logger.warning("有效因子不足，跳过相关性计算")
                return None

            correlation_matrix = pd.DataFrame(index=final_factor_names, columns=final_factor_names, dtype=float)

            # Step 2: 计算相关性 (在循环内部进行配对对齐)
            for i in range(len(final_factor_names)):
                for j in range(i, len(final_factor_names)):
                    factor1_name = final_factor_names[i]
                    factor2_name = final_factor_names[j]

                    if i == j:
                        correlation_matrix.loc[factor1_name, factor1_name] = 1.0
                        continue

                    data1 = factor_data_dict[factor1_name]
                    data2 = factor_data_dict[factor2_name]

                    # --- 核心改进：在这里进行配对对齐 ---
                    common_index = data1.index.intersection(data2.index)
                    common_columns = data1.columns.intersection(data2.columns)

                    aligned_data1 = data1.loc[common_index, common_columns]
                    aligned_data2 = data2.loc[common_index, common_columns]
                    # --- 对齐结束 ---

                    # 使用向量化计算截面相关性时间序列
                    time_corrs = aligned_data1.corrwith(aligned_data2, axis=1, method='spearman')

                    # 检查每日有效样本数 (这一步依然非常专业且必要)
                    valid_counts = aligned_data1.notna() & aligned_data2.notna()
                    valid_daily_counts = valid_counts.sum(axis=1)

                    valid_time_corrs = time_corrs[valid_daily_counts > 10]

                    if not valid_time_corrs.empty:
                        avg_corr = valid_time_corrs.mean()
                        correlation_matrix.loc[factor1_name, factor2_name] = avg_corr
                        correlation_matrix.loc[factor2_name, factor1_name] = avg_corr
                    else:
                        # 如果没有任何一天满足计算条件，则认为无相关性
                        correlation_matrix.loc[factor1_name, factor2_name] = 0.0
                        correlation_matrix.loc[factor2_name, factor1_name] = 0.0

            return correlation_matrix.astype(float)

        except Exception as e:
            # 在顶层函数捕获异常，而不是在加载函数中抛出
            raise ValueError(f"❌ 相关性矩阵计算失败: {e}")

    def _load_factor_data(self, factor_name: str) -> Optional[pd.DataFrame]:
        """加载单个因子数据用于相关性计算"""
        try:
            # 构建数据路径
            factor_dir = (self.main_work_path / self.pool_index / factor_name / 
                         'o2o' / self.version)
            
            # 寻找处理后的因子文件
            processed_file = factor_dir / 'processed_factor.parquet'
            if processed_file.exists():
                return pd.read_parquet(processed_file)
            
            # 备用：寻找其他可能的数据文件
            parquet_files = list(factor_dir.glob("*.parquet"))
            if parquet_files:
                return pd.read_parquet(parquet_files[0])
            
            raise ValueError(f"  未找到 {factor_name} 的数据文件")

        except Exception as e:
            raise ValueError(f"  加载 {factor_name} 数据失败: {e}")

    def _select_best_factor(
            self, 
            competitors: List[str], 
            qualified_factors: Dict[str, FactorRollingICStats]
    ) -> str:
        """从竞争因子中选择最佳因子（用于红色警报区域）"""
        
        # 按多周期综合评分排序
        scored_competitors = []
        for factor in competitors:
            if factor in qualified_factors:
                score = qualified_factors[factor].multi_period_score
                scored_competitors.append((factor, score))
            else:
                # 如果没有统计数据，给予最低评分
                scored_competitors.append((factor, 0.0))
        
        # 选择评分最高的因子
        scored_competitors.sort(key=lambda x: x[1], reverse=True)
        winner = scored_competitors[0][0]
        
        return winner
    
    def generate_final_selection(self, category_champions: Dict[str, List[str]], 
                                qualified_factors: Dict[str, FactorRollingICStats]) -> List[str]:
        """
        生成最终因子选择
        （只是过滤数量的过滤而已），限制最多八个
        Args:
            category_champions: 各类别冠军
            qualified_factors: 合格因子统计
            
        Returns:
            List[str]: 最终选择的因子名单
        """
        logger.info("生成最终因子选择...")
        
        # 收集所有冠军
        all_champions = []
        for category, champions in category_champions.items():
            for champion in champions:
                if champion in qualified_factors:
                    all_champions.append((champion, qualified_factors[champion]))
        
        # 按多周期评分排序
        all_champions.sort(key=lambda x: x[1].multi_period_score, reverse=True)
        
        # 选择前N名
        max_selection = min(len(all_champions), self.config.max_final_factors)
        final_selection = [name for name, _ in all_champions[:max_selection]]
        
        logger.info(f"最终选择 {len(final_selection)} 个因子:")
        for i, (name, stats) in enumerate(all_champions[:max_selection], 1):
            direction = "+" if list(stats.periods_data.values())[0]['ic_mean_avg'] > 0 else "-"
            logger.info(f"{i}. {direction} {name}")
            logger.info(f"   评分: {stats.multi_period_score:.1f}")
            logger.info(f"   IC: {stats.avg_ic_abs:.3f}, IR: {stats.avg_ir_abs:.2f}")
            logger.info(f"   稳定性: {stats.avg_stability:.1%}")
            logger.info(f"   时间跨度: {stats.time_range[0]} ~ {stats.time_range[1]}")
        
        return final_selection
    
    def run_complete_selection(self, factor_names: List[str], force_generate: bool = False) -> Tuple[List[str], Dict[str, Any]]:
        """
        运行完整的因子筛选流程
        
        Args:
            factor_names: 候选因子列表
            force_generate: 是否强制重新生成滚动IC
            
        Returns:
            Tuple[List[str], Dict]: (选中因子列表, 详细报告)
        """
        logger.info("=" * 60)
        logger.info("开始基于滚动IC的完整因子筛选")
        logger.info("=" * 60)
        
        # 第一步：基于滚动IC筛选
        qualified_factors = self.screen_factors_by_rolling_ic(factor_names, force_generate)
        
        if not qualified_factors:
            logger.warning("警告：没有因子通过滚动IC筛选")
            return [], {}
        
        # 第二步：类别内选择
        category_champions = self.select_category_champions(qualified_factors)
        
        if not category_champions:
            logger.warning("警告：没有类别冠军")
            return [], {}
        
        # 第三步：初步最终选择 （只是过滤数量的过滤而已），限制最多八个
        preliminary_selection = self.generate_final_selection(category_champions, qualified_factors)
        
        # 第四步：三层相关性控制哲学
        final_selection, correlation_report = self.apply_correlation_control( #debug here
            preliminary_selection, qualified_factors
        )
        
        # 生成详细报告
        report = self._generate_selection_report(
            factor_names, qualified_factors, category_champions, final_selection, correlation_report
        )
        
        logger.info("=" * 60)
        logger.info("滚动IC因子筛选完成！")
        logger.info(f"推荐用于IC加权合成: {final_selection}")
        logger.info("=" * 60)
        
        return final_selection, report
    
    def _generate_selection_report(self, candidate_factors: List[str], 
                                  qualified_factors: Dict[str, FactorRollingICStats],
                                  category_champions: Dict[str, List[str]], 
                                  final_selection: List[str],
                                  correlation_report: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成选择报告"""
        
        # 统计信息
        qualified_count = len(qualified_factors)
        champions_count = sum(len(champions) for champions in category_champions.values())
        final_count = len(final_selection)
        
        # 评分统计
        if qualified_factors:
            scores = [stats.multi_period_score for stats in qualified_factors.values()]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        # 类别分布
        category_distribution = {}
        for factor in final_selection:
            for category, factor_list in self.factor_categories.items():
                if factor in factor_list:
                    category_distribution[category] = category_distribution.get(category, 0) + 1
                    break
        
        # 构建报告
        report = {
            'selection_config': {
                'snap_config_id': self.snap_config_id,
                'pool_index': self.pool_index,
                'time_range': f"{self.start_date} - {self.end_date}",
                'selection_criteria': {
                    'min_ic_abs_mean': self.config.min_ic_abs_mean,
                    'min_ir_abs_mean': self.config.min_ir_abs_mean,
                    'min_ic_stability': self.config.min_ic_stability,
                    'decay_rate': self.config.decay_rate
                }
            },
            'selection_summary': {
                'candidate_count': len(candidate_factors),
                'qualified_count': qualified_count,
                'champions_count': champions_count,
                'final_count': final_count,
                'pass_rate': qualified_count / len(candidate_factors) if candidate_factors else 0.0
            },
            'score_statistics': {
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score
            },
            'category_distribution': category_distribution,
            'final_selection': final_selection,
            'factor_details': {
                factor: {
                    'multi_period_score': qualified_factors[factor].multi_period_score,
                    'avg_ic_abs': qualified_factors[factor].avg_ic_abs,
                    'avg_ir_abs': qualified_factors[factor].avg_ir_abs,
                    'avg_stability': qualified_factors[factor].avg_stability,
                    'snapshot_count': qualified_factors[factor].snapshot_count,
                    'time_range': qualified_factors[factor].time_range
                }
                for factor in final_selection if factor in qualified_factors
            }
        }
        
        # 添加相关性控制报告
        if correlation_report:
            report['correlation_control'] = {
                'enabled': True,
                'philosophy': '三层相关性控制哲学',
                'thresholds': correlation_report.get('thresholds', {}),
                'processing_summary': {
                    'input_factors': correlation_report.get('input_count', 0),
                    'final_factors': correlation_report.get('final_count', 0),
                    'orthogonalized_factors': correlation_report.get('orthogonalized_count', 0),
                    'total_decisions': len(correlation_report.get('decisions', []))
                },
                'decisions_breakdown': self._summarize_correlation_decisions(correlation_report.get('decisions', [])),
                'orthogonalized_factors': correlation_report.get('orthogonalized_factors', []),
                'detailed_decisions': correlation_report.get('decisions', [])
            }
        else:
            report['correlation_control'] = {
                'enabled': False,
                'reason': '相关性控制跳过或失败'
            }
        
        return report
    
    def _summarize_correlation_decisions(self, decisions: List[Dict]) -> Dict[str, int]:
        """汇总相关性决策统计"""
        summary = {
            '红色警报-二选一': 0,
            '黄色预警-正交化': 0,
            '绿色安全-直接保留': 0
        }
        
        for decision in decisions:
            decision_type = decision.get('decision', '')
            if decision_type in summary:
                summary[decision_type] += 1
        
        return summary

        # 函数1: 只负责加载，不再负责对齐

    def _load_all_factor_data(self, factor_names: List[str]) -> Dict[str, pd.DataFrame]:
        """仅加载所有因子数据到字典中，不进行对齐"""
        factor_data_dict = {}
        for factor_name in factor_names:
            try:
                factor_data = self._load_factor_data(factor_name)
                if factor_data is not None and not factor_data.empty:
                    factor_data_dict[factor_name] = factor_data
                else:
                    logger.warning(f"  ⚠️ {factor_name}: 数据加载失败或为空")
            except Exception as e:
                logger.warning(f"  ❌ {factor_name}: 数据加载异常 - {e}")
                continue

        if len(factor_data_dict) < 2:
            raise ValueError("⚠️ 有效因子数量不足，无法计算相关性")

        return factor_data_dict
    def _process_red_zone_clusters(
            self, 
            candidate_factors: List[str], 
            correlation_matrix: pd.DataFrame,
            qualified_factors: Dict[str, FactorRollingICStats]
    ) -> Tuple[List[str], List[Dict]]:
        """
        阶段1：红色区域集群消杀 - 处理高相关性集群
        
        🎯 核心算法：
        1. 构建高相关图（|corr| > threshold）
        2. 使用图算法找出连通分量（集群）
        3. 每个集群内选择评分最高的因子作为代表
        4. 产出：幸存者列表 + 决策记录
        
        Args:
            candidate_factors: 候选因子列表
            correlation_matrix: 相关性矩阵
            qualified_factors: 因子评分统计
            
        Returns:
            (survivors, decisions): 幸存者列表和决策记录
        """
        from collections import defaultdict
        
        # Step 1: 构建高相关图
        high_corr_graph = defaultdict(set)
        high_corr_pairs = []
        
        for i in range(len(candidate_factors)):
            for j in range(i + 1, len(candidate_factors)):
                factor1 = candidate_factors[i]
                factor2 = candidate_factors[j]
                corr = abs(correlation_matrix.loc[factor1, factor2])
                
                if corr >= self.config.high_corr_threshold:
                    high_corr_graph[factor1].add(factor2)
                    high_corr_graph[factor2].add(factor1)
                    high_corr_pairs.append((factor1, factor2, corr))
        
        # Step 2: 使用DFS找出连通分量（高相关集群）
        def find_clusters():
            visited = set()
            clusters = []
            
            def dfs(node, current_cluster):#node:需给这个node找帮凶， 都放在这个cluster中
                if node in visited:
                    return
                visited.add(node)#染黑，下次进来发现！已经被处理
                current_cluster.add(node)
                for neighbor in high_corr_graph[node]:#找出与之相关的，B C ，B又去找与B相关的xx ，（简直就是连根拔起，然后放入一个集合，最后可能多个集合，我们只要每个集合的高分选手！
                    dfs(neighbor, current_cluster)
            
            for factor in candidate_factors:
                if factor not in visited:
                    cluster = set()
                    dfs(factor, cluster)
                    if len(cluster) > 1:  # 只关心有相关性的集群
                        clusters.append(cluster)
                    elif len(cluster) == 1:  # 单 （没有帮手） 那么可以直接加入幸存者
                        pass
            
            return clusters
        
        clusters = find_clusters()
        
        # Step 3: 每个集群选择代表（评分最高者）
        survivors = []
        decisions = []
        processed_factors = set()
        
        # 处理高相关集群
        for i, cluster in enumerate(clusters):
            cluster_list = list(cluster)
            
            # 选择集群内评分最高的因子
            cluster_scores = []
            for factor in cluster_list:
                if factor in qualified_factors:
                    score = qualified_factors[factor].multi_period_score
                    cluster_scores.append((factor, score))
                else:
                    cluster_scores.append((factor, 0.0))
            
            # 按评分排序，选择最高者
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            champion = cluster_scores[0][0] #高相关里 最厉害的
            losers = [name for name, _ in cluster_scores[1:]]
            
            survivors.append(champion)
            processed_factors.update(cluster)
            
            # 记录决策
            for loser in losers:
                # 找出champion和loser的具体相关系数
                loser_corr = abs(correlation_matrix.loc[champion, loser])
                decisions.append({
                    'stage': 'red_zone_cluster',
                    'cluster_id': i,
                    'cluster_size': len(cluster),
                    'champion': champion,
                    'loser': loser,
                    'correlation': loser_corr,
                    'decision': '红色警报-集群消杀',
                    'reason': f'高相关集群内竞争(|corr|={loser_corr:.3f}>{self.config.high_corr_threshold})'
                })
            
            logger.info(f"  🔥 集群{i+1}: {len(cluster)}个因子 → 选择 {champion}，淘汰 {losers}")
        
        # Step 4: 处理无高相关的独立因子（直接幸存）
        independent_factors = [f for f in candidate_factors if f not in processed_factors]
        survivors.extend(independent_factors)
        
        for factor in independent_factors:
            logger.info(f"  ✅ 独立因子: {factor} 直接幸存")
        
        logger.info(f"🚨 红色区域处理完成: 发现 {len(clusters)} 个高相关集群，{len(independent_factors)} 个独立因子")
        logger.info(f"   最终幸存者: {len(survivors)} 个")
        
        return survivors, decisions
    
    def _process_clusters_hierarchical(
        self,
        candidate_factors: List[str],
        correlation_matrix: pd.DataFrame,
        qualified_factors: Dict[str, FactorRollingICStats]
    ) -> Tuple[List[str], List[Dict]]:
        """
        阶段1：使用层次聚类进行数据驱动的集群划分和代表选举
        
        🎯 核心优势:
        1. 全局视角：同时考虑所有因子间的相关性结构
        2. 数据驱动：无需人工设定阈值，自动发现最优簇结构
        3. 层次信息：保留因子间的层次相似关系
        4. 稳健性：Ward连接方法最小化簇内方差，结果更稳定
        
        Args:
            candidate_factors: 候选因子列表
            correlation_matrix: 相关性矩阵
            qualified_factors: 因子评分统计
            
        Returns:
            (survivors, decisions): 幸存者列表和决策记录
        """
        if len(candidate_factors) < 2:
            logger.info("  ⚠️ 候选因子不足2个，跳过层次聚类")
            return candidate_factors, []

        try:
            # Step 1: 将相关性矩阵转化为距离矩阵
            # 距离 = 1 - |相关系数|，这样强相关（corr=1）的因子距离为0
            abs_corr_matrix = abs(correlation_matrix)
            distance_matrix = 1 - abs_corr_matrix
            
            # 确保距离矩阵对角线为0（自己与自己的距离）
            np.fill_diagonal(distance_matrix.values, 0)
            
            # 转换为scipy层次聚类所需的压缩距离向量
            condensed_distance = squareform(distance_matrix.values, force='tovector')
            
            # Step 2: 执行层次聚类
            linkage_method = self.config.hierarchical_linkage_method
            logger.info(f"  🔬 执行层次聚类 (method={linkage_method})...")
            
            linkage_matrix = linkage(condensed_distance, method=linkage_method)
            
            # Step 3: 根据配置决定簇划分策略
            if self.config.max_clusters is not None:
                # 策略A: 固定簇数量
                cluster_labels = fcluster(linkage_matrix, self.config.max_clusters, criterion='maxclust')
                logger.info(f"  📊 固定簇数量策略: {self.config.max_clusters} 个簇")
            else:
                # 策略B: 距离阈值自适应
                distance_threshold = self.config.hierarchical_distance_threshold
                cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
                logger.info(f"  📊 距离阈值策略: threshold={distance_threshold}")
            
            # Step 4: 构建簇信息
            clusters = {}
            for i, factor in enumerate(candidate_factors):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(factor)
            
            n_clusters = len(clusters)
            logger.info(f"  🎯 发现 {n_clusters} 个层次簇")
            
            # Step 5: 每个簇选择最佳代表因子
            survivors = []
            decisions = []
            
            for cluster_id, cluster_factors in clusters.items():
                cluster_size = len(cluster_factors)
                
                if cluster_size == 1:
                    # 单因子簇：直接保留
                    survivor = cluster_factors[0]
                    survivors.append(survivor)
                    logger.info(f"  🏆 簇{cluster_id}: 单因子 {survivor} 直接保留")
                    
                else:
                    # 多因子簇：选择最佳代表
                    champion = self._elect_best_factor_in_cluster(cluster_factors, qualified_factors)
                    losers = [f for f in cluster_factors if f != champion]
                    survivors.append(champion)
                    
                    # 计算簇内平均相关性（用于记录）
                    cluster_correlations = []
                    for i in range(len(cluster_factors)):
                        for j in range(i+1, len(cluster_factors)):
                            factor1, factor2 = cluster_factors[i], cluster_factors[j]
                            corr = abs_corr_matrix.loc[factor1, factor2]
                            cluster_correlations.append(corr)
                    
                    avg_intra_cluster_corr = np.mean(cluster_correlations) if cluster_correlations else 0.0
                    
                    logger.info(f"  🏆 簇{cluster_id}: {cluster_size}个因子 → 选择 {champion}")
                    logger.info(f"      淘汰: {losers}")
                    logger.info(f"      簇内平均相关性: {avg_intra_cluster_corr:.3f}")
                    
                    # 记录决策
                    for loser in losers:
                        loser_corr = abs_corr_matrix.loc[champion, loser]
                        decisions.append({
                            'stage': 'hierarchical_clustering',
                            'cluster_id': cluster_id,
                            'cluster_size': cluster_size,
                            'champion': champion,
                            'loser': loser,
                            'correlation': loser_corr,
                            'avg_intra_cluster_corr': avg_intra_cluster_corr,
                            'decision': '层次聚类-簇内竞选',
                            'reason': f'层次聚类簇内竞争(簇{cluster_id},平均|corr|={avg_intra_cluster_corr:.3f})',
                            'clustering_method': linkage_method,
                            'distance_threshold': self.config.hierarchical_distance_threshold
                        })
            
            # Step 6: 生成聚类洞察报告
            self._generate_clustering_insights(
                linkage_matrix, cluster_labels, candidate_factors, survivors, correlation_matrix
            )
            
            logger.info(f"🔬 层次聚类完成:")
            logger.info(f"   输入因子: {len(candidate_factors)}")
            logger.info(f"   发现簇数: {n_clusters}")
            logger.info(f"   选出代表: {len(survivors)}")
            logger.info(f"   淘汰因子: {len(candidate_factors) - len(survivors)}")
            
            return survivors, decisions
            
        except Exception as e:
            logger.error(f"❌ 层次聚类失败: {e}")
            logger.info("   回退到图算法方法...")
            # 回退到原始图算法方法
            return self._process_red_zone_clusters(candidate_factors, correlation_matrix, qualified_factors)
    
    def _elect_best_factor_in_cluster(
        self, 
        cluster_factors: List[str], 
        qualified_factors: Dict[str, FactorRollingICStats]
    ) -> str:
        """
        在簇内选举最佳代表因子
        
        综合评分标准:
        1. 多周期IC评分 (60%权重)
        2. Newey-West显著性 (25%权重) 
        3. 因子稳定性 (15%权重)
        """
        if len(cluster_factors) == 1:
            return cluster_factors[0]
        
        # 计算每个因子的综合竞选分数
        candidates_scores = []
        
        for factor in cluster_factors:
            if factor in qualified_factors:
                stats = qualified_factors[factor]
                
                # 1. IC评分 (归一化到0-1)
                ic_score = min(stats.multi_period_score / 100.0, 1.0)
                
                # 2. 显著性评分 (基于Newey-West t统计量)
                nw_significance_score = min(abs(stats.nw_t_stat_series_mean) / 3.0, 1.0)
                
                # 3. 稳定性评分
                stability_score = stats.avg_stability
                
                # 综合评分
                comprehensive_score = (
                    ic_score * 0.60 + 
                    nw_significance_score * 0.25 + 
                    stability_score * 0.15
                )
                
                candidates_scores.append((factor, comprehensive_score, {
                    'ic_score': ic_score,
                    'nw_significance': nw_significance_score, 
                    'stability': stability_score
                }))
            else:
                # 没有统计数据的因子给予最低分
                candidates_scores.append((factor, 0.0, {}))
        
        # 按综合分数排序，选择最高分
        candidates_scores.sort(key=lambda x: x[1], reverse=True)
        
        champion = candidates_scores[0][0]
        champion_score = candidates_scores[0][1]
        
        logger.debug(f"      簇内竞选结果: {champion} (综合分数: {champion_score:.3f})")
        
        return champion
    
    def _generate_clustering_insights(
        self, 
        linkage_matrix: np.ndarray,
        cluster_labels: np.ndarray, 
        factor_names: List[str],
        survivors: List[str],
        correlation_matrix: pd.DataFrame
    ) -> None:
        """
        生成层次聚类洞察报告 (可选可视化)
        """
        try:
            # 1. 簇间距离分析
            n_clusters = len(set(cluster_labels))
            
            # 2. 因子保留率分析
            retention_rate = len(survivors) / len(factor_names) if factor_names else 0
            
            # 3. 平均簇内相关性
            clusters = {}
            for i, factor in enumerate(factor_names):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(factor)
            
            cluster_internal_correlations = []
            for cluster_factors in clusters.values():
                if len(cluster_factors) > 1:
                    cluster_corrs = []
                    for i in range(len(cluster_factors)):
                        for j in range(i+1, len(cluster_factors)):
                            corr = abs(correlation_matrix.loc[cluster_factors[i], cluster_factors[j]])
                            cluster_corrs.append(corr)
                    if cluster_corrs:
                        cluster_internal_correlations.append(np.mean(cluster_corrs))
            
            avg_intra_cluster_corr = np.mean(cluster_internal_correlations) if cluster_internal_correlations else 0
            
            logger.info(f"  📈 聚类洞察:")
            logger.info(f"     因子保留率: {retention_rate:.1%}")
            logger.info(f"     平均簇内相关性: {avg_intra_cluster_corr:.3f}")
            logger.info(f"     多因子簇数量: {len(cluster_internal_correlations)}")
            
            # 可选：保存树状图 (在研究环境中很有用)
            # self._save_dendrogram(linkage_matrix, factor_names)
            
        except Exception as e:
            logger.debug(f"聚类洞察生成失败: {e}")
    
    def _save_dendrogram(self, linkage_matrix: np.ndarray, factor_names: List[str]) -> None:
        """保存层次聚类树状图 (可选功能)"""
        try:
            plt.figure(figsize=(15, 8))
            dendrogram(
                linkage_matrix,
                labels=factor_names,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True
            )
            plt.title('Factor Hierarchical Clustering Dendrogram')
            plt.xlabel('Factors')
            plt.ylabel('Distance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存到工作目录
            output_path = self.main_work_path / f"dendrogram_{self.snap_config_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  📊 树状图已保存: {output_path}")
            
        except Exception as e:
            logger.debug(f"树状图保存失败: {e}")
            plt.close()

    def _process_yellow_zone_orthogonalization(
            self, 
            red_zone_survivors: List[str],
            qualified_factors: Dict[str, FactorRollingICStats]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        阶段2：黄色区域正交化处理 - 基于幸存者处理中度相关性
        
        🎯 核心逻辑：
        1. 基于红色区域幸存者重新计算相关性
        2. 找出所有中度相关对 (0.3 < |corr| < 0.7)
        3. 生成正交化改造计划（不直接修改因子列表）
        4. 产出：最终因子列表 + 正交化计划 + 决策记录
        
        Args:
            red_zone_survivors: 红色区域幸存者
            qualified_factors: 因子评分统计
            
        Returns:
            (final_factors, orthogonalization_plan, decisions)
        """
        # Step 1: 基于幸存者重新计算相关性
        if len(red_zone_survivors) < 2:
            logger.info("  ⚠️ 幸存者不足2个，跳过黄色区域处理")
            return red_zone_survivors, [], []
        
        try:
            survivors_correlation_matrix = self._calculate_factor_correlations(red_zone_survivors)
            if survivors_correlation_matrix is None:
                raise ValueError("  ⚠️ 无法计算幸存者相关性矩阵，跳过正交化处理")
                # return red_zone_survivors, [], []
        except Exception as e:
            raise ValueError(f"  ⚠️ 幸存者相关性计算失败: {e}，跳过正交化处理")
            # return red_zone_survivors, [], []
        
        # Step 2: 找出中度相关对
        medium_corr_pairs = []
        for i in range(len(red_zone_survivors)):
            for j in range(i + 1, len(red_zone_survivors)):
                factor1 = red_zone_survivors[i]
                factor2 = red_zone_survivors[j]
                corr = abs(survivors_correlation_matrix.loc[factor1, factor2])
                
                if self.config.medium_corr_threshold <= corr < self.config.high_corr_threshold:
                    medium_corr_pairs.append((factor1, factor2, corr))
        
        logger.info(f"  📊 发现 {len(medium_corr_pairs)} 对中度相关因子")
        
        # Step 3: 生成正交化计划
        orthogonalization_plan = []
        decisions = []
        final_factors = red_zone_survivors.copy()  # 先保留所有幸存者
        
        if not self.config.enable_orthogonalization:
            logger.info("  ⚠️ 正交化功能已禁用，所有幸存者直接保留")
            return final_factors, [], []
        
        # 按相关性从高到低处理
        medium_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for factor1, factor2, corr in medium_corr_pairs:
            # 选择评分更高的作为基准
            score1 = qualified_factors[factor1].multi_period_score if factor1 in qualified_factors else 0.0
            score2 = qualified_factors[factor2].multi_period_score if factor2 in qualified_factors else 0.0
            
            if score1 >= score2:
                base_factor, target_factor = factor1, factor2
            else:
                base_factor, target_factor = factor2, factor1
            
            # 生成正交化计划
            orthogonal_name = f"{target_factor}_orth_vs_{base_factor}"#base 高分！
            
            orthogonalization_plan.append({
                'original_factor': target_factor,
                'base_factor': base_factor,
                'orthogonal_name': orthogonal_name,
                'correlation': corr,
                'base_score': qualified_factors[base_factor].multi_period_score if base_factor in qualified_factors else 0.0,
                'target_score': qualified_factors[target_factor].multi_period_score if target_factor in qualified_factors else 0.0
            })
            
            # 记录决策
            decisions.append({
                'stage': 'yellow_zone_orthogonalization',
                'base_factor': base_factor,
                'target_factor': target_factor,
                'orthogonal_name': orthogonal_name,
                'correlation': corr,
                'decision': '黄色预警-正交化',
                'reason': f'中度相关({self.config.medium_corr_threshold}<=|corr|={corr:.3f}<{self.config.high_corr_threshold})'
            })
            
            logger.info(f"  🔄 正交化计划: {target_factor} → {orthogonal_name} (基于 {base_factor}，相关性={corr:.3f})")
        
        # Step 4: 最终检查 - 确保没有高相关遗漏
        remaining_high_corr = []
        for i in range(len(final_factors)):
            for j in range(i + 1, len(final_factors)):
                factor1 = final_factors[i]
                factor2 = final_factors[j]
                corr = abs(survivors_correlation_matrix.loc[factor1, factor2])
                if corr >= self.config.high_corr_threshold:
                    remaining_high_corr.append((factor1, factor2, corr))
        
        if remaining_high_corr:
            raise ValueError(f"  ❌ 严重问题：最终因子中仍存在高相关因子 {factor1} vs {factor2}: {corr:.3f}")
        logger.info(f"⚠️ 黄色区域处理完成:")
        logger.info(f"   最终因子数: {len(final_factors)}")
        logger.info(f"   正交化计划: {len(orthogonalization_plan)} 个")
        logger.info(f"   决策记录: {len(decisions)} 条")
        
        return final_factors, orthogonalization_plan, decisions