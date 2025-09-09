"""
滚动IC管理器 - 解决前视偏差的关键组件

核心功能：
1. 时点化IC计算：严格按时间点滚动计算IC，避免未来信息泄露
2. 增量存储：支持增量计算和存储，提升效率
3. 窗口管理：灵活的回看窗口配置
4. 数据完整性：确保IC计算的时间一致性

设计理念：
- 完全杜绝前视偏差
- 支持实盘级别的严格时间控制
- 高效的增量计算和存储
"""
import math
from scipy import stats

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json

from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import ConfigSnapshotManager
from projects._03_factor_selection.utils.date.trade_date_utils import get_end_day_pre_n_day
from quant_lib.config.logger_config import setup_logger
from quant_lib.evaluation.evaluation import _calculate_newey_west_tstat

logger = setup_logger(__name__)


@dataclass
class ICCalculationConfig:
    """IC计算配置"""
    lookback_months: int = 12  # 回看窗口(月) 目前写死-注意调整 0.1
    forward_periods: List[str] = None  # 前向收益周期
    min_require_observations: int = 120  # 最小观测数量  目前写死-注意调整 0.1
    calculation_frequency: str = 'M'  # 计算频率 ('M'=月末, 'Q'=季末)
    significance_threshold: float = 1.96  # 显著性阈值 (95%置信度)
    ewma_span: int = 126  # EWMA窗口 (约半年)
    max_monthly_turnover: float = 0.40  # 最大月度换手率上限
    turnover_mode: str = 'calculate'  # 换手率计算模式: 'estimate'(经验估算) 或 'calculate'(动态计算)

    def __init__(self,lookback_months=12, forward_periods: list=None , min_require_observations: int = 120, calculation_frequency: str = 'M',calcu_type='o2o', version='20190328_20231231', turnover_mode='estimate'):
        self.lookback_months = lookback_months
        self.forward_periods = forward_periods
        self.min_require_observations = min_require_observations
        self.calculation_frequency = calculation_frequency
        self.turnover_mode = turnover_mode

        self.calcu_type=calcu_type
        self.version=version


@dataclass
class ICSnapshot:
    """IC快照数据结构"""
    calculation_date: str  # 计算时点
    factor_name: str  # 因子名称
    stock_pool_index: str  # 股票池
    window_start: str  # 回看窗口起点
    window_end: str  # 回看窗口终点
    ic_stats: Dict[str, Dict]  # 各周期IC统计
    metadata: Dict  # 元数据信息


class RollingICManager:
    """滚动IC管理器 - 无前视偏差的IC计算与存储"""

    def __init__(self,calcu_return_type, config: Optional[ICCalculationConfig] = None,version=None):
        self.main_work_path = Path(
            r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result")
        self.config = config or ICCalculationConfig()
        self.calcu_return_type=calcu_return_type
        self.version = version

        # 时点IC索引
        self._ic_index = {}
        self._load_ic_index()
    #ok 肉眼逐行debug 数据完美
    def calculate_and_store_rolling_ic(
            self,
            factor_names: List[str],
            stock_pool_index: str,
            start_date: str,
            end_date: str,
            resultLoadManager:ResultLoadManager,  # 数据源
            force_recalculate: bool = False
    ) -> Dict[str, List[ICSnapshot]]:
        """
        计算并存储滚动IC
        
        Args:
            factor_names: 因子名称列表
            stock_pool_index: 股票池名称
            start_date: 开始计算时点
            end_date: 结束计算时点
            factor_data_source: 因子数据源
            return_data_source: 收益数据源
            force_recalculate: 是否强制重新计算
            
        Returns:
            Dict[factor_name, List[ICSnapshot]]: 所有因子的IC快照序列
        """
        logger.info(f"🔄 开始滚动IC计算: {start_date} -> {end_date}")
        logger.info(f"📊 因子数量: {len(factor_names)}, 股票池: {stock_pool_index}")

        # 1. 生成计算时点序列
        calculation_dates = self._generate_calculation_dates(start_date, end_date)
        logger.info(f"⏰ 计算时点数量: {len(calculation_dates)}")

        # 2. 逐时点计算IC
        all_factor_snapshots = {name: [] for name in factor_names}

        for calc_date in calculation_dates:
            logger.info(f"📅 计算时点: {calc_date}")

            for factor_name in factor_names:
                try:
                    # 检查是否已存在计算结果
                    if not force_recalculate and self._snapshot_exists(
                            factor_name, stock_pool_index, calc_date
                    ):
                        snapshot = self._load_snapshot(factor_name, stock_pool_index, calc_date)
                        logger.debug(f"  📥 {factor_name}: 使用已有快照")
                    else:
                        # 计算新的IC快照
                        snapshot = self._calculate_ic_snapshot(
                            factor_name, stock_pool_index, calc_date,
                            resultLoadManager
                        )

                        if snapshot:
                            self._save_snapshot(snapshot)
                            # logger.debug(f"  ✅ {factor_name}:{calc_date} IC快照计算完成")
                        else:#很正常啊，比如不满足观测点个数的时候
                            continue

                    all_factor_snapshots[factor_name].append(snapshot)

                except Exception as e:
                    logger.error(f"  ❌ {factor_name}: IC计算异常 - {e}")
                    continue

        logger.info(f"✅ 滚动IC计算完成")
        return all_factor_snapshots

    def get_ic_at_timepoint(
            self,
            factor_name: str,
            stock_pool: str,
            calculation_date: str
    ) -> Optional[ICSnapshot]:
        """获取指定时点的IC快照"""
        return self._load_snapshot(factor_name, stock_pool, calculation_date)

    def get_ic_series(
            self,
            factor_name: str,
            stock_pool: str,
            start_date: str,
            end_date: str
    ) -> List[ICSnapshot]:
        """获取时间序列的IC快照"""
        snapshots = []

        # 从索引中查找符合条件的快照
        key_pattern = f"{factor_name}_{stock_pool}"

        for key, metadata in self._ic_index.items():
            if key.startswith(key_pattern):
                calc_date = metadata['calculation_date']
                if start_date <= calc_date <= end_date:
                    snapshot = self._load_snapshot(factor_name, stock_pool, calc_date)
                    if snapshot:
                        snapshots.append(snapshot)

        # 按计算时点排序
        snapshots.sort(key=lambda x: x.calculation_date)
        return snapshots

    def _calculate_ic_snapshot(
            self,
            factor_name: str,
            stock_pool_index: str,
            calculation_date: str,#月度快照 1231 0131 0229 0331.。
            resultLoadManager:ResultLoadManager
    ) -> Optional[ICSnapshot]:
        """计算单个时点的IC快照"""
        try:
            # 1. 确定回看窗口（严格避免前视偏差）
            calc_date = pd.Timestamp(calculation_date)
            window_end = calc_date #todo 应该是以据period cal_date 减去 period
            window_start = calc_date - relativedelta(months=self.config.lookback_months) #回看12个月

            # 2. 获取窗口内的因子数据
            factor_data = resultLoadManager.get_factor_data(
                factor_name, stock_pool_index,
                window_start.strftime('%Y-%m-%d'),
                window_end.strftime('%Y-%m-%d')
            )

            if factor_data is None or factor_data.empty:
                raise ValueError(f"因子 {factor_name} 在窗口 {window_start}-{window_end} 内无数据")

            # 3. 计算各周期IC统计
            ic_stats = {}

            for period in self.config.forward_periods:
                # 获取前向收益数据
                period_days = period
                return_end = window_end + timedelta(days=period_days + 10)  # 留充足余量

                return_data = resultLoadManager.get_o2o_return_data(
                    stock_pool_index,
                    window_start.strftime('%Y-%m-%d'),
                    return_end.strftime('%Y-%m-%d'),
                    period_days
                )

                if return_data is None or return_data.empty:
                    raise ValueError('收益率数据不可能为空！，严重错误！')

                # 根据预测周期，确定因子数据的有效截止日期
                # 站在 calc_date，要评价一个预测期为 period 的因子，
                # 最晚的因子日期 T 必须满足 T + period <= calc_date。
                # c. 对原始的、完整的因子数据进行【截断】，得到本次计算所需的安全子集
                # 1. 精确计算出因子数据在此周期下的有效截止日期
                effective_end_date = get_end_day_pre_n_day(calculation_date,period)
                # 2. 使用布尔索引，基于【日期】进行过滤
                factor_data = factor_data[factor_data.index <= effective_end_date]


                # 计算IC
                period_ic_stats = self.generate_ic_snapshot_stats(
                    factor_data, return_data, period_days,
                    factor_name=factor_name,
                    stock_pool_index=stock_pool_index,
                    resultLoadManager=resultLoadManager
                )

                if period_ic_stats:
                    ic_stats[period] = period_ic_stats

            if not ic_stats:
                logger.warning(f"因子 {factor_name} 在时点 {calculation_date} 无有效IC统计--正常：因为不满足120个观测点！（人话：回头看的天数没有达到120天")
                return None

            # 4. 构建IC快照
            snapshot = ICSnapshot(
                calculation_date=calculation_date,
                factor_name=factor_name,
                stock_pool_index=stock_pool_index,
                window_start=window_start.strftime('%Y-%m-%d'),
                window_end=window_end.strftime('%Y-%m-%d'),
                ic_stats=ic_stats,
                metadata={
                    'config_manager': {
                        'lookback_months': self.config.lookback_months,
                        'min_require_observations': self.config.min_require_observations
                    },
                    'data_points': len(factor_data),
                    'created_timestamp': datetime.now().isoformat()
                }
            )

            return snapshot

        except Exception as e:
            logger.error(f"计算IC快照失败 {factor_name}@{calculation_date}: {e}")
            return None

    def generate_ic_snapshot_stats(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        period_days: int,
        factor_name: str = None,
        stock_pool_index: str = None,
        resultLoadManager = None
    ) -> Optional[Dict]:
        """计算特定周期的IC统计 - 修复重叠窗口偏差"""
        try:
            # 对齐因子和收益数据
            aligned_factor, aligned_return = self._align_data(factor_data, return_data)

            if len(aligned_factor) < self.config.min_require_observations:
                return None
            ic_series= resultLoadManager.get_ic_series_by_period(stock_pool_index, factor_name, period_days)
            if len(ic_series) == 0:
                logger.info(f"因子 {factor_name} 非重叠采样后无有效IC统计 ic个数为0")
                return None

            # IC统计指标 - 使用EWMA动态计算 (可配置span，默认126约等于半年)
            ewma_span = getattr(self.config, 'ewma_span', 126)

            # 注意：由于非重叠采样后样本数减少，需要调整EWMA参数
            adjusted_ewma_span = min(ewma_span // period_days, len(ic_series) // 2)#月度：结果6
            if adjusted_ewma_span < 6:
                # 样本太少，使用传统统计方法
                ic_mean = ic_series.mean()
                ic_std_ewma = ic_series.std()
                ic_std_rolling = ic_series.std()
            else:
                ic_mean = ic_series.ewm(span=adjusted_ewma_span).mean().iloc[-1]
                ic_std_ewma = ic_series.ewm(span=adjusted_ewma_span).std().iloc[-1]
                ic_std_rolling = ic_series.std()

            ic_ir = ic_mean / ic_std_ewma if ic_std_ewma > 0 else 0
            # 确定长期方向，增加阈值保护
            threshold_mean = 0.001
            long_term_ic_mean = ic_series.mean()
            if abs(long_term_ic_mean) < threshold_mean:
                # 方向不明显，直接按正向处理或略过
                factor_direction = 1
            else:
                factor_direction = np.sign(long_term_ic_mean)

            # 胜负序列（保留方向性）
            win_loss_series = ((ic_series * factor_direction) > 0).astype(int)#同号

            # EWMA 胜率
            ic_win_rate_ewma = win_loss_series.ewm(span=ewma_span).mean().iloc[-1]
            # t检验 (传统)
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(ic_series, 0)

            # 动态Newey-West T-stat (稳健版异方差调整) 不需要ewma处理 （因为他本质是全量统计
            ic_nw_t_stat, ic_nw_p_value = _calculate_newey_west_tstat(ic_series) #ok

            #动态计算 (基于实际排名变化)
            avg_daily_rank_change_turnover_stats  = self._calculate_daily_rank_change(
                factor_name,aligned_factor, stock_pool_index, ic_series.index, resultLoadManager
            )
            # 计算显著性标记和质量评估
            significance_threshold = getattr(self.config, 'significance_threshold', 1.96)
            max_turnover = getattr(self.config, 'max_monthly_turnover', 0.40)

            is_significant_nw = abs(ic_nw_t_stat) > significance_threshold
            is_significant_traditional = abs(t_stat) > significance_threshold

            # 综合质量评级
            quality_score = self._calculate_quality_score(
                ic_mean, ic_ir, ic_win_rate_ewma, ic_nw_t_stat, avg_daily_rank_change_turnover_stats.get('avg_daily_rank_change', 0)
            )

            return {
                'ic_mean': float(ic_mean),
                'ic_std_ewma': float(ic_std_ewma),  # 新增EWMA标准差
                'ic_std_rolling': float(ic_std_rolling),  # 保留全样本标准差
                'ic_ir': float(ic_ir),  # 基于EWMA标准差的IR
                'ic_win_rate': float(ic_win_rate_ewma) ,
                'ic_t_stat': float(t_stat),  # 传统t统计量
                'ic_p_value': float(p_value),
                'ic_nw_t_stat': float(ic_nw_t_stat),  # Newey-West调整T统计量
                'ic_nw_p_value': float(ic_nw_p_value),  # 对应p值
                'ic_count': len(ic_series),
                'ic_max': float(ic_series.max()),
                'ic_min': float(ic_series.min()),
                # 新增质量指标
                'is_significant_nw': bool(is_significant_nw),  # Newey-West显著性
                'is_significant_traditional': bool(is_significant_traditional),  # 传统显著性
                'avg_daily_rank_change_stats': avg_daily_rank_change_turnover_stats,  # 换手率约束
                # 'quality_score': float(quality_score),  # 综合质量评分
                **avg_daily_rank_change_turnover_stats  # 动态换手率统计
            }

        except Exception as e:
            raise  ValueError(f"计算周期IC失败: {e}")
    #a give


    def _calculate_quality_score(
        self, 
        ic_mean: float, 
        ic_ir: float, 
        ic_win_rate: float, 
        nw_t_stat: float, 
        avg_daily_rank_change: float
    ) -> float:
        """
        计算因子综合质量评分 (0-1范围)
        
        考虑因素:
        1. IC绝对值 (40%权重)
        2. IR指标 (25%权重) 
        3. 显著性 (20%权重)
        4. 胜率 (10%权重)
        5. 换手率惩罚 (5%权重)
        """
        try:
            # 1. IC强度评分 (0-1)
            ic_score = min(abs(ic_mean) / 0.05, 1.0) * 0.4
            
            # 2. IR指标评分 (0-1) 
            ir_score = min(abs(ic_ir) / 2.0, 1.0) * 0.25
            
            # 3. 显著性评分 (0-1)
            significance_score = min(abs(nw_t_stat) / 3.0, 1.0) * 0.2
            
            # 4. 胜率评分 (0-1)
            win_rate_score = max(0, (ic_win_rate - 0.5) * 2) * 0.1
            
            # 5. 换手率惩罚 (0-1)
            turnover_penalty = max(0, 1 - avg_daily_rank_change / 0.5) * 0.05
            
            total_score = ic_score + ir_score + significance_score + win_rate_score + turnover_penalty
            return min(total_score, 1.0)
            
        except:
            return 0.0
    
    def _calculate_daily_rank_change(
        self, 
        factor_name: str,
        factor_data: pd.DataFrame,
        stock_pool_index: str,
        date_index: pd.DatetimeIndex,
        resultLoadManager = None
    ) -> Dict[str, float]:
        """
       基于实际因子排名变化)
        
        Args:
            factor_name: 因子名称
            stock_pool_index: 股票池索引
            date_index: IC计算日期索引
            resultLoadManager: 数据加载管理器
            衡量了在整个股票池中，股票的“座次”平均发生了多大的变化。
        Returns:
            Dict: 换手率相关统计指标
        """
        try:
            if resultLoadManager is None or len(date_index) < 2:
                return self._get_empty_turnover_stats()

            if factor_data is None or factor_data.empty:
                logger.debug(f"因子 {factor_name} 数据不足，无法计算动态换手率")
                return self._get_empty_turnover_stats()
            
            # 2. 计算每个日期的因子排名百分位 (使用分位数排名，更稳定)
            monthly_rankings = {}
            monthly_dates = sorted(date_index)
            
            for calc_date in monthly_dates:
                calc_timestamp = pd.Timestamp(calc_date)
                
                # 获取该月末前后几天的数据来增强稳定性
                #防止在进行1231计算月度快照计算，发现1230 1229 都没有数据！（可能是假期
                window_start = calc_timestamp - pd.Timedelta(days=5)
                window_end = calc_timestamp + pd.Timedelta(days=1)
                
                # 在因子数据中找到最接近的交易日
                available_dates = factor_data.index
                valid_dates = available_dates[
                    (available_dates >= window_start) & (available_dates <= window_end)
                ]
                
                if len(valid_dates) == 0:
                    continue
                
                # 使用最接近目标日期的数据
                target_date = valid_dates[np.argmin(np.abs((valid_dates - calc_timestamp).days))]
                daily_factor = factor_data.loc[target_date].dropna()
                
                if len(daily_factor) < 10:  # 至少需要10只股票
                    continue
                
                # 计算分位数排名 (0-1之间)
                rankings = daily_factor.rank(pct=True, method='average')
                monthly_rankings[calc_date] = rankings
            
            if len(monthly_rankings) < 2:
                logger.debug(f"因子 {factor_name} 有效排名数据不足，无法计算换手率")
                return self._get_empty_turnover_stats()
            
            # 3. 计算相邻期间的排名变化 (换手率计算)
            turnover_rates = []
            ranking_dates = sorted(monthly_rankings.keys())
            
            for i in range(1, len(ranking_dates)):
                prev_date = ranking_dates[i-1]
                curr_date = ranking_dates[i]
                
                prev_rankings = monthly_rankings[prev_date]
                curr_rankings = monthly_rankings[curr_date]
                
                # 找到两期共同的股票
                common_stocks = prev_rankings.index.intersection(curr_rankings.index)
                
                if len(common_stocks) < 10:
                    continue
                
                prev_common = prev_rankings.loc[common_stocks]
                curr_common = curr_rankings.loc[common_stocks]
                
                # 计算排名变化的绝对值平均 (这是换手率的核心指标)
                ranking_changes = np.abs(curr_common - prev_common)
                mean_absolute_rank_change = ranking_changes.mean()
                
                turnover_rates.append(mean_absolute_rank_change)
            
            if len(turnover_rates) == 0:
                return self._get_empty_turnover_stats()
            
            # 4. 统计换手率时间序列
            turnover_array = np.array(turnover_rates)
            
            avg_daily_rank_change = np.mean(turnover_array)
            daily_turnover_volatility = float(np.std(turnover_array))
            
            # 计算换手率趋势 (线性回归斜率)
            if len(turnover_rates) >= 3:
                x = np.arange(len(turnover_rates))
                z = np.polyfit(x, turnover_array, 1)
                daily_turnover_trend = z[0]    #斜率
            else:
                daily_turnover_trend = 0.0
            
            return {
                'avg_daily_rank_change': float(avg_daily_rank_change),
                'daily_turnover_volatility': float(daily_turnover_volatility),
                'daily_turnover_trend': float(daily_turnover_trend),
                'sample_periods': len(turnover_rates),
                'calculation_method': 'rank_change_dynamic'
            }
            
        except Exception as e:
            raise ValueError(f"动态换手率计算失败 {factor_name}: {e}")

    #可删除
    def _estimate_factor_turnover(self, factor_name: str) -> Dict[str, float]:
        """
        估算因子换手率 (方案1: 基于经验和因子类型)
        
        Args:
            factor_name: 因子名称
            
        Returns:
            Dict: 换手率相关统计指标
        """
        try:
            # 根据因子类型估算换手率（基于经验和研究）
            turnover_estimates = {
                # 高频类因子（技术面）
                'reversal_1d': 0.30, 'reversal_5d': 0.25, 'reversal_10d': 0.20, 'reversal_21d': 0.18,
                'momentum_20d': 0.18, 'rsi': 0.22, 'cci': 0.24, 'rsi_经过残差化': 0.22, 'cci_经过残差化': 0.24,
                'macd': 0.20, 'bollinger_position': 0.28, 'rsi_divergence': 0.26, 'pead': 0.23,
                
                # 中频类因子（价量结合）
                'momentum_60d': 0.15, 'momentum_120d': 0.12, 'momentum_12_1': 0.10, 'momentum_6_1': 0.13, 'momentum_3_1': 0.16,
                'momentum_pct_60d': 0.15, 'sharpe_momentum_60d': 0.14, 'sw_l1_momentum_21d': 0.17,
                'volatility_40d': 0.16, 'volatility_90d': 0.14, 'volatility_120d': 0.12,
                'volatility_40d_经过残差化': 0.16, 'volatility_90d_经过残差化': 0.14, 'volatility_120d_经过残差化': 0.12,
                'atr_20d': 0.18,
                'amihud_liquidity': 0.14, 'turnover_rate_90d_mean': 0.16, 'turnover_rate_monthly_mean': 0.15,
                'turnover_rate_90d_mean-经过残差化': 0.16, 'turnover_rate_monthly_mean_经过残差化': 0.15,
                'ln_turnover_value_90d': 0.17, 'ln_turnover_value_90d_经过残差化': 0.17,
                'turnover_t1_div_t20d_avg': 0.19, 'bid_ask_spread': 0.21,
                
                # 低频类因子（基本面）
                'ep_ratio': 0.08, 'bm_ratio': 0.07, 'sp_ratio': 0.08, 'cfp_ratio': 0.09, 'pb_ratio': 0.08,
                'pe_ttm': 0.08, 'ps_ratio': 0.08, 'value_composite': 0.07,
                'roe_ttm': 0.06, 'gross_margin_ttm': 0.05, 'earnings_stability': 0.04, 'roa_ttm': 0.06,
                'total_revenue_growth_yoy': 0.07, 'net_profit_growth_yoy': 0.08, 'eps_growth': 0.08,
                'operating_revenue_growth': 0.07, 'gross_profit_margin': 0.05, 'operating_margin': 0.05,
                'net_margin': 0.05, 'ebit_margin': 0.05,
                
                # 规模因子（极低频）
                'log_circ_mv': 0.03, 'log_total_mv': 0.03, 'market_cap_weight': 0.02,
                
                # 质量因子（低频）
                'debt_to_assets': 0.05, 'current_ratio': 0.04, 'asset_turnover': 0.06,
                'quality_momentum': 0.09, 'operating_accruals': 0.07, 'inventory_turnover': 0.06,
                'receivables_turnover': 0.06, 'working_capital_turnover': 0.07
            }
            
            # 基础换手率估算
            base_turnover = turnover_estimates.get(factor_name, 0.12)  # 默认12%
            
            return {
                'avg_daily_rank_change': float(base_turnover),
                'daily_turnover_volatility': float(base_turnover * 0.3),  # 波动率约为均值的30%
                'daily_turnover_trend': 0.0,  # 经验估算无法提供趋势信息
                'calculation_method': 'factor_type_estimate'
            }
            
        except Exception as e:
            logger.debug(f"换手率估算失败 {factor_name}: {e}")
            return self._get_empty_turnover_stats()
    
    def _get_empty_turnover_stats(self) -> Dict[str, float]:
        """返回空的换手率统计"""
        return {
            'avg_daily_rank_change': 0.0,
            'daily_turnover_volatility': 0.0,
            'daily_turnover_trend': 0.0,
        }

    def _align_data(self, factor_data: pd.DataFrame, return_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """对齐因子和收益数据"""
        # 找到共同的时间和股票
        common_dates = factor_data.index.intersection(return_data.index)
        common_stocks = factor_data.columns.intersection(return_data.columns)

        aligned_factor = factor_data.loc[common_dates, common_stocks]
        aligned_return = return_data.loc[common_dates, common_stocks]

        return aligned_factor, aligned_return
    #返回每个月最后一天!
    def _generate_calculation_dates(self, start_date: str, end_date: str) -> List[str]:
        """生成计算时点序列"""
        dates = []
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 根据频率生成时点
        if self.config.calculation_frequency == 'M':
            # 月末
            while current <= end:
                # 找到当月最后一个工作日
                month_end = current + pd.offsets.MonthEnd(0)
                if month_end <= end:
                    dates.append(month_end.strftime('%Y-%m-%d'))
                current = month_end + pd.offsets.MonthEnd(1)
        elif self.config.calculation_frequency == 'Q':
            # 季末
            while current <= end:
                quarter_end = current + pd.offsets.QuarterEnd(0)
                if quarter_end <= end:
                    dates.append(quarter_end.strftime('%Y-%m-%d'))
                current = current + pd.offsets.QuarterEnd(1)

        return dates

    def _snapshot_exists(self, factor_name: str, stock_pool: str, calculation_date: str) -> bool:
        """检查IC快照是否已存在"""
        snapshot_key = f"{factor_name}_{stock_pool}_{calculation_date}"
        return snapshot_key in self._ic_index

    def _save_snapshot(self, snapshot: ICSnapshot):
        """保存IC快照"""
        # 构建文件路径
        snapshot_dir = self.main_work_path / snapshot.stock_pool_index / snapshot.factor_name / self.calcu_return_type / self.version / 'rolling_ic'
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ic_snapshot_{snapshot.calculation_date}.json"
        filepath = snapshot_dir / filename

        # 序列化快照
        snapshot_dict = {
            'calculation_date': snapshot.calculation_date,
            'factor_name': snapshot.factor_name,
            'stock_pool_index': snapshot.stock_pool_index,
            'window_start': snapshot.window_start,
            'window_end': snapshot.window_end,
            'ic_stats': snapshot.ic_stats,
            'metadata': snapshot.metadata
        }

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot_dict, f, ensure_ascii=False, indent=2)

        # 更新索引
        snapshot_key = f"{snapshot.factor_name}_{snapshot.stock_pool_index}_version_{self.version}_calculation_date_{snapshot.calculation_date}"
        self._ic_index[snapshot_key] = {
            'calculation_date': snapshot.calculation_date,
            'filepath': str(filepath),
            'created_at': datetime.now().isoformat()
        }

        self._save_ic_index()
        logger.debug(f"IC快照已保存: {filepath}")

    def _load_snapshot(self, factor_name: str, stock_pool_index: str, calculation_date: str) -> Optional[ICSnapshot]:
        """加载IC快照"""
        snapshot_key =f'{factor_name}_{stock_pool_index}_version_{self.version}_calculation_date_{calculation_date}'

        if snapshot_key not in self._ic_index:
            return None

        try:
            filepath = self._ic_index[snapshot_key]['filepath']

            with open(filepath, 'r', encoding='utf-8') as f:
                snapshot_dict = json.load(f)

            return ICSnapshot(**snapshot_dict)

        except Exception as e:
            logger.error(f"加载IC快照失败 {snapshot_key}: {e}")
            return None

    def _load_ic_index(self):
        """加载IC索引"""
        index_file = self.main_work_path / "ic_index.json"

        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self._ic_index = json.load(f)
                logger.info(f"IC索引加载完成，共 {len(self._ic_index)} 条记录")
            except Exception as e:
                raise ValueError(f"加载IC索引失败: {e}")
        else:
            self._ic_index = {}

    def _save_ic_index(self):
        """保存IC索引"""
        index_file = self.main_work_path / "ic_index.json"

        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self._ic_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存IC索引失败: {e}")

    def get_latest_calculation_date(self, factor_name: str, stock_pool: str) -> Optional[str]:
        """获取因子的最新计算时点"""
        pattern = f"{factor_name}_{stock_pool}_"
        latest_date = None

        for key, metadata in self._ic_index.items():
            if key.startswith(pattern):
                calc_date = metadata['calculation_date']
                if latest_date is None or calc_date > latest_date:
                    latest_date = calc_date

        return latest_date

    def cleanup_old_snapshots(self, keep_months: int = 36):
        """清理过期的IC快照"""
        cutoff_date = datetime.now() - relativedelta(months=keep_months)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        removed_count = 0
        keys_to_remove = []

        for key, metadata in self._ic_index.items():
            if metadata['calculation_date'] < cutoff_str:
                try:
                    # 删除文件
                    filepath = Path(metadata['filepath'])
                    if filepath.exists():
                        filepath.unlink()

                    keys_to_remove.append(key)
                    removed_count += 1

                except Exception as e:
                    logger.error(f"删除快照失败 {key}: {e}")

        # 更新索引
        for key in keys_to_remove:
            del self._ic_index[key]

        self._save_ic_index()
        logger.info(f"清理完成，删除 {removed_count} 个过期快照")


def run_cal_and_save_rolling_ic_by_snapshot_config_id(snapshot_config_id, factor_names):
    manager = ConfigSnapshotManager()
    pool_index,s,e ,config_evaluation= manager.get_snapshot_config_content_details(snapshot_config_id)
    version = f'{s}_{e}'
    config = ICCalculationConfig(
        lookback_months=12,
        forward_periods=config_evaluation['forward_periods'],
        min_require_observations=120,
        calculation_frequency='M'
    )
    if 'o2o' not in config_evaluation['returns_calculator']:
        raise ValueError("之前的测试 计算收益率不是按照c2c来的，现在无法滚动 ")
    manager = RollingICManager('o2o', config,version)

    resultLoadManager = ResultLoadManager(calcu_return_type='o2o', version=version,
                                          is_raw_factor=False)

    stock_pool_index = pool_index

    snapshots = manager.calculate_and_store_rolling_ic(
        factor_names, stock_pool_index, s, e,
        resultLoadManager, True
    )
    print(f"计算完成，共生成 {sum(len(snaps) for snaps in snapshots.values())} 个IC快照")
if __name__ == '__main__':
    # 使用并发执行器进行批量计算
    from projects._03_factor_selection.utils.efficiency_engineering.concurrent_executor import run_concurrent_factors
    
    snapshot_config_id = '20250906_045625_05e460ab'
    df = pd.read_csv(r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factor_manager\selector\o2o_v3.csv')
    factor_names = df['factor_name'].unique().tolist()
    
    logger.info(f"📊 开始批量计算 {len(factor_names)} 个因子的滚动IC")
    factor_names = ['lqs_orthogonal_v1']
    # # 并发执行 - 单因子模式，适合内存充足的情况
    # successful_results, failed_factors = run_concurrent_factors(
    #     factor_names=factor_names,
    #     snapshot_config_id=snapshot_config_id,
    #     max_workers=3,  # 根据机器配置调整
    #     execution_mode="chunked"  # 或 "chunked" 用于分组执行
    # )
    #
    # logger.info(f"🎉 批量计算完成!")
    # logger.info(f"✅ 成功: {len(successful_results)} 个因子")
    # logger.info(f"❌ 失败: {len(failed_factors)} 个因子")
    #
    # if failed_factors:
    #     logger.warning("失败的因子:")
    #     for factor, error in failed_factors:
    #         logger.warning(f"  - {factor}: {error}")
    
    # 单个测试用法(保留原有方式)
    run_cal_and_save_rolling_ic_by_snapshot_config_id('20250909_125913_b5be5b49',['vwap_deviation_20d'] )