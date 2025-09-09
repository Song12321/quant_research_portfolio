import json
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from typing import Optional
from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd

from projects._03_factor_selection.config_manager.base_config import INDEX_CODES, workspaces_result_dir
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import ConfigSnapshotManager
from projects._03_factor_selection.config_manager.function_load.load_config_file import _load_local_config_functional
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.utils.factor_scoring_v33_final import calculate_factor_score_v33
from projects._03_factor_selection.visualization_manager import VisualizationManager
from quant_lib import logger


@dataclass
class FactorStats:
    factor_name: str
    ic_mean_21d: float
    ic_ir_21d: float
    detail_score_21d: float
    top_q_turnover_dict: dict
    # periods_data: Dict[str, Dict]  # 各周期数据
    # avg_ic_with_sign: float  # 带符号
    # avg_ir_ir_with_sign: float
    # avg_ic_abs: float  # 平均IC绝对值
    # avg_ir_abs: float  # 平均IR绝对值
    # best_period_ic_ir: float  # ir所在 表现最佳的周期val
    # nw_t_stat_series_mean: float
    # avg_stability: float  # 平均稳定性
    # avg_ic_volatility: float  # 平均IC波动率
    # detail_score_21d: float  # 多周期综合评分
    # snapshot_count: int  # 快照数量
    # time_range: Tuple[str, str]  # 时间范围
    #
    # # 实盘交易成本控制
    # # avg_daily_rank_change: float = 0.0    # 平均月度换手率
    # daily_rank_change_mean: float
    # daily_turnover_trend: float
    # daily_turnover_volatility: float
    # turnover_adjusted_score: float = 0.0  # 换手率调整后评分


@dataclass
class SelectionConfig:
    """滚动IC筛选配置"""
    # 基本筛选门槛
    min_snapshots: int = 3  # 最少快照数量
    min_ic_abs_mean: float = 0.01  # 滚动IC均值绝对值门槛
    min_ir_abs_mean: float = 0.15  # 滚动IR均值绝对值门槛
    min_ic_stability: float = 0.4  # IC稳定性门槛（方向一致性）
    max_ic_volatility: float = 0.05  # IC波动率上限

    # 多周期权重配置
    decay_rate: float = 0.75  # 衰减率，越小权重衰减越慢
    prefer_short_term: bool = True  # 偏向短期

    # 类别内选择
    max_factors_per_category: int = 10  # 每类最多因子数
    min_category_score: float = 10.0  # 类别最低评分

    # 最终筛选
    max_final_factors: int = 30  # 最多选择因子数

    # 相关性控制（三层决策哲学）
    high_corr_threshold: float = 0.7  # 高相关阈值（红色警报：二选一）
    medium_corr_threshold: float = 0.3  # 中低相关分界（黄色预警：正交化战场）
    enable_orthogonalization: bool = True  # 是否启用中相关区间正交化

    # 层次聚类配置
    clustering_method: str = 'graph'  # 聚类方法: 'graph'(图算法) 或 'hierarchical'(层次聚类)
    hierarchical_distance_threshold: float = 0.3  # 层次聚类距离阈值
    hierarchical_linkage_method: str = 'ward'  # 连接方法: 'ward', 'complete', 'average'
    max_clusters: int = None  # 最大簇数量限制 (None表示使用距离阈值)

    # 实盘交易成本控制（换手率一等公民）
    max_turnover_rate: float = 0.15  # 最大换手率阈值（月度）
    turnover_weight: float = 0.25  # 换手率在综合评分中的权重
    enable_turnover_penalty: bool = False  # 是否启用换手率惩罚 todo 后续在补充

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
    max_turnover_mean_daily: float = 0.15  # 硬门槛：日均换手率不得超过1% (约等于月度21%)
    max_turnover_trend_daily: float = 0.00005  # 硬门槛：换手率每日恶化趋势不得超过0.002%
    max_turnover_vol_daily: float = 0.015  # 硬门槛：换手率波动率不得超过1.5%

class FactorSelector:
    def __init__(self,snapshot_config_id,config: SelectionConfig):
        self.snap_config_id = snapshot_config_id
        manager = ConfigSnapshotManager()
        pool_index, s, e, config_evaluation = manager.get_snapshot_config_content_details(snapshot_config_id)
        version = f'{s}_{e}'
        self.start_date = s
        self.end_date = e
        self.pool_index = pool_index

        self.resultLoadManager = ResultLoadManager(pool_index = pool_index,s=s,e=e,version=version)
        self.config = config or SelectionConfig()
        # 假设因子测试中所有可能用到的周期都在这里定义
        self.ALL_PERIODS = ['1d', '5d', '10d', '21d', '40d', '60d', '120d']
        self.visualization_manager = VisualizationManager(
        )
        self.factor_categories = self.build_factor_categorie_maps()

        # 函数1: 只负责加载，不再负责对齐

    def load_all_factor_data(self,factor_names: List[str]) -> Dict[str, pd.DataFrame]:
        """仅加载所有因子数据到字典中，不进行对齐"""
        factor_data_dict = {}
        for factor_name in factor_names:
            try:
                factor_data = self._load_factor_data(factor_name)
                if factor_data is not None and not factor_data.empty:
                    factor_data_dict[factor_name] = factor_data
                else:
                    raise ValueError(f"  ⚠️ {factor_name}: 数据加载失败或为空")
            except Exception as e:
                raise ValueError(f"  ❌ {factor_name}: 数据加载异常 - {e}")
                continue

        if len(factor_data_dict) < 2:
            raise ValueError("⚠️ 有效因子数量不足，无法计算相关性")

        return factor_data_dict

    def _load_factor_data(self, factor_name: str) -> Optional[pd.DataFrame]:
        return self.resultLoadManager.get_factor_data(factor_name)
    def build_factor_icir_data(self,
                                   run_version: str = 'latest') :
        base_path = workspaces_result_dir / self.resultLoadManager.pool_index
        ret = {}
        for factor_dir in base_path.iterdir():
            if not factor_dir.is_dir(): continue
            one_period = {}
            factor_name = factor_dir.name
            for period in self.ALL_PERIODS:
                # 1. 为当前因子和周期构建一个完整的指标行
                current_period_row = self._build_single_period_row(factor_dir, period, run_version)
                if current_period_row == None:
                    continue
                ic_ir = current_period_row['ic_ir_processed_o2o']
                ic_mean = current_period_row['ic_mean_processed_o2o']
                ic_t_stat = current_period_row['ic_t_stat_processed_o2o']
                one_period[period] = {'ic_mean':ic_mean,'ic_ir':ic_ir, 'ic_t_stat':ic_t_stat}
            if  len (one_period)!=0:
                ret[factor_name] = one_period
        return ret
    # 最新版评价因子！挑选因子！
    # 全局ic ir 进行评价！
    def get_base_passed_factors(self
                                ):
        all_factors_summary_data = self.build_factor_icir_data()
        # --- 2. 执行筛选与画像 ---
        elite_factor_reports = profile_elite_factors(
            all_factors_summary=all_factors_summary_data
        )
        names = []
        # --- 3. 查看精英因子的深度画像报告 ---
        for factor_name, report in elite_factor_reports.items():
            names.append(factor_name)
            print(f"\n----- {factor_name} 精英因子报告 -----")
            # 使用json美化输出
            print(json.dumps(report, indent=4, ensure_ascii=False))
        print("\n" + "=" * 50)
        print(f"因子list：：{names}")
        return names

    def _process_red_zone_clusters(
            self,
            candidate_factors: List[str],
            correlation_matrix: pd.DataFrame,
            qualified_factors: Dict[str, FactorStats]
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

            def dfs(node, current_cluster):  # node:需给这个node找帮凶， 都放在这个cluster中
                if node in visited:
                    return
                visited.add(node)  # 染黑，下次进来发现！已经被处理
                current_cluster.add(node)
                for neighbor in high_corr_graph[
                    node]:  # 找出与之相关的，B C ，B又去找与B相关的xx ，（简直就是连根拔起，然后放入一个集合，最后可能多个集合，我们只要每个集合的高分选手！
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
                    score = qualified_factors[factor].detail_score_21d['Final_Score']
                    cluster_scores.append((factor, score))
                else:
                    cluster_scores.append((factor, 0.0))

            # 按评分排序，选择最高者
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            champion = cluster_scores[0][0]  # 高相关里 最厉害的
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

            logger.info(f"  🔥 集群{i + 1}: {len(cluster)}个因子 → 选择 {champion}，淘汰 {losers}")

        # Step 4: 处理无高相关的独立因子（直接幸存）
        independent_factors = [f for f in candidate_factors if f not in processed_factors]
        survivors.extend(independent_factors)

        for factor in independent_factors:
            logger.info(f"  ✅ 独立因子: {factor} 直接幸存")

        logger.info(f"🚨 红色区域处理完成: 发现 {len(clusters)} 个高相关集群，{len(independent_factors)} 个独立因子")
        logger.info(f"   最终幸存者: {len(survivors)} 个")

        return survivors, decisions

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
                        for j in range(i + 1, len(cluster_factors)):
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

    def _process_clusters_hierarchical(
            self,
            candidate_factors: List[str],
            correlation_matrix: pd.DataFrame,
            qualified_factors: Dict[str, FactorStats]
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
                        for j in range(i + 1, len(cluster_factors)):
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

    def apply_correlation_control(
            self,
            candidate_factors: List[str],
            qualified_factors: Dict[str, FactorStats]
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
        else:  # todo 对比看看 新方法结果一致不
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

        logger.info(
            f"  📊 正交化处理结果: {len(red_zone_survivors)} → {len(final_factors)} + {len(orthogonalization_plan)} 个正交化计划")

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
            factor_data_dict = self.load_all_factor_data(factor_names)

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

    def _process_yellow_zone_orthogonalization(
            self,
            red_zone_survivors: List[str],
            qualified_factors: Dict[str, FactorStats]
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
            score1 = qualified_factors[factor1].detail_score_21d['Final_Score'] if factor1 in qualified_factors else 0.0
            score2 = qualified_factors[factor2].detail_score_21d['Final_Score'] if factor2 in qualified_factors else 0.0

            if score1 >= score2:
                base_factor, target_factor = factor1, factor2
            else:
                base_factor, target_factor = factor2, factor1

            # 生成正交化计划
            orthogonal_name = f"{target_factor}_orth_vs_{base_factor}"  # base 高分！

            orthogonalization_plan.append({
                'original_factor': target_factor,
                'base_factor': base_factor,
                'orthogonal_name': orthogonal_name,
                'correlation': corr,
                'base_score': qualified_factors[
                    base_factor].detail_score_21d['Final_Score'] if base_factor in qualified_factors else 0.0,
                'target_score': qualified_factors[
                    target_factor].detail_score_21d['Final_Score'] if target_factor in qualified_factors else 0.0
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

            logger.info(
                f"  🔄 正交化计划: {target_factor} → {orthogonal_name} (基于 {base_factor}，相关性={corr:.3f})")

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
    #todo 后面检验真确性
    def screen_factors_by_recent_rolling_ic(
            self,
            phase1_passed_factors: List[str],  # 输入是第一阶段筛选出的精英因子列表
            force_generate: bool = False  # 这个参数用于控制是否重新加载数据
    ) -> List[str]:
        """
        【Phase 2】对已经通过全样本检验的精英因子，根据其近期滚动IC表现进行优中选优。
        Args:
            phase1_passed_factors: 第一阶段筛选出的、简历过硬的因子列表。
        Returns:
            List[str]: 通过了近期状态检验的、最终合格的因子列表。
        """
        # --- 1. 定义滚动筛选的配置和门槛 ---
        ROLLING_SCREENING_CONFIG = {
            "rolling_window_size": 12,  # 回看窗口：12个周期 (对于月度IC序列，即12个月)
            "min_rolling_icir_threshold": 0.25,  # 近期滚动ICIR的最低门槛
            "min_rolling_ic_mean_threshold": 0.01  # 近期滚动IC均值的最低门槛 (确保方向正确)
        }
        logger.info("--- 开始阶段二：基于近期滚动IC表现进行优中选优 ---")
        passed_phase2_factors = []

        # 遍历每一个“简历”过硬的因子
        for factor_name in phase1_passed_factors:

            # --- 步骤 1: 加载该因子的“黄金IC序列” ---
            # 这个序列是我们之前通过 calculate_non_overlapping_ic_series 生成的、
            # 贯穿整个历史的、干净的月度IC序列。
            # 在实际系统中，这里会有缓存逻辑。
            golden_ic_series = self.resultLoadManager.get_ic_series_by_period(
                self.TARGET_UNIVERSE,
                factor_name,
                period_days=21  # 假设我们分析的是月度IC序列
            )

            if golden_ic_series is None or len(golden_ic_series) < ROLLING_SCREENING_CONFIG["rolling_window_size"]:
                logger.warning(f"因子 {factor_name} 的IC序列过短，无法进行滚动分析，已跳过。")
                continue

            # --- 步骤 2: 计算滚动统计指标 ---
            # 使用pandas的 .rolling() 方法进行计算
            window_size = ROLLING_SCREENING_CONFIG["rolling_window_size"]
            rolling_ic_mean = golden_ic_series.rolling(window=window_size).mean()
            rolling_ic_std = golden_ic_series.rolling(window=window_size).std()

            # 计算滚动的ICIR序列
            rolling_icir = rolling_ic_mean / rolling_ic_std

            # --- 步骤 3: 提取最新的滚动值作为“近期表现” ---
            # .iloc[-1] 可以获取到时间序列的最后一个值
            latest_rolling_ic_mean = rolling_ic_mean.iloc[-1]
            latest_rolling_icir = rolling_icir.iloc[-1]

            # --- 步骤 4: 执行筛选 ---
            # 核心决策逻辑：近期表现是否达标？
            # 注意：我们这里也应该使用绝对值，以容纳反向因子

            icir_passed = abs(latest_rolling_icir) >= ROLLING_SCREENING_CONFIG["min_rolling_icir_threshold"]

            # 同时，要确保近期表现的方向与长期方向一致
            # 我们用全样本均值的符号代表长期方向
            long_term_direction = np.sign(golden_ic_series.mean())
            mean_passed = (latest_rolling_ic_mean * long_term_direction) >= ROLLING_SCREENING_CONFIG[
                "min_rolling_ic_mean_threshold"]

            if icir_passed and mean_passed:
                logger.info(f"  > ✅ 因子 {factor_name} 通过近期状态检验 (滚动ICIR={latest_rolling_icir:.2f})")
                passed_phase2_factors.append(factor_name)
            else:
                logger.info(f"  > ❌ 因子 {factor_name} 未通过近期状态检验 (滚动ICIR={latest_rolling_icir:.2f})，被剔除。")

        return passed_phase2_factors
    def get_passed_factor_names(self,   need_filter_rencent_bad: bool = False,force_generate:bool=False) -> List[str]:
        passed_factor_names = self.get_base_passed_factors()
        if need_filter_rencent_bad:
            return self.screen_factors_by_recent_rolling_ic(passed_factor_names,force_generate)
        return passed_factor_names

    def select_category_champions(self, passed_factor_stats) -> Dict[str, List[str]]:
        """
        类别内冠军选择
        Args:
            passed_factor_names: 通过基本筛选的因子
             for :n个类别"
                类内排名逻辑：按换手率加权的周期衰减总ic分数
                每个类别只要2个
        Returns:
            Dict[category, List[factor_names]]: 各类别的冠军因子
        """
        logger.info("开始类别内冠军选择...")
        category_champions = {}
        # 注意 遍历的是类别！，而不是因子，所以务必需要保证类别在config配置文件！
        for category, factor_list in self.factor_categories.items():
            # 找到该类别中的合格因子
            category_factors = {
                name: stats for name, stats in passed_factor_stats.items()
                if name in factor_list
            }

            if not category_factors:
                continue

            # 按换手率调整后评分排序（实盘导向优化）
            sorted_factors = sorted(
                category_factors.items(),
                key=lambda x: x[1].detail_score_21d['Final_Score'] if self.config.enable_turnover_penalty else x[ #第一个final_score 记得切换为 换手率*原始分数 todo
                    1].detail_score_21d['Final_Score'],
                reverse=True
            )

            # 选择前N名
            max_count = min(len(sorted_factors), self.config.max_factors_per_category)
            champions = [name for name, _ in sorted_factors[:max_count]]

            if champions:
                category_champions[category] = champions
                logger.info(f"{category}: {len(champions)} 个冠军")
                for name in champions:
                    stats = passed_factor_stats[name]
                    direction = "+" if np.sign(stats.ic_mean_21d) > 0 else "-"
                    score_used = stats.detail_score_21d if self.config.enable_turnover_penalty else stats.detail_score_21d
                    logger.info(
                        f"  {direction} {name}: 调整后21d评分={score_used} (top_q_21d换手率={stats.top_q_turnover_dict['21d']})")

        return category_champions

    def _generate_selection_report(self, candidate_factors: List[str],
                                   qualified_factors: Dict[str, FactorStats],
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
            scores = [stats.detail_score_21d['Final_Score'] for stats in qualified_factors.values()]
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
                    'final_score_21d': qualified_factors[factor].detail_score_21d['Final_Score'],
                    # 'avg_ic_abs': qualified_factors[factor].avg_ic_abs,
                    # 'avg_ir_abs': qualified_factors[factor].avg_ir_abs,
                    # 'avg_stability': qualified_factors[factor].avg_stability,
                    # 'snapshot_count': qualified_factors[factor].snapshot_count,
                    # 'time_range': qualified_factors[factor].time_range
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

    def run_complete_selection(self,pool_index, force_generate: bool = False) -> Tuple[
        List[str], Dict[str, Any]]:
        """
        第一步：全样本“硬筛选” (Phase 1 - 看简历):

        （全样本IC均值、ICIR、Newey-West T值），对所有备选因子进行一次残酷的“资格认证”。

        目的： 确保进入下一轮的，都是在过去数年完整历史中，被证明了“基因”优秀的因子。

        第二步：滚动表现“优中选优” (Phase 2 - 看状态):

        在第一步筛选出的“精英池”内部，我们才开始考察它们近期的滚动IC表现。

        目的： 从一群“基因”都很好的因子中，挑选出那些“近期状态”也正佳的。

        第三步：类别内选择 (Intra-Category Selection):

        流程不变。

        第四步：相关性控制 (Correlation Control):

        流程不变。
        Args:
            factor_names: 候选因子列表
        Returns:
            Tuple[List[str], Dict]: (选中因子列表, 详细报告)
        """
        logger.info("=" * 60)
        logger.info("开始基于滚动IC的完整因子筛选")
        logger.info("=" * 60)

        # 第一步 筛选（base+近期表现
        passed_factor_names = self.get_passed_factor_names( False, force_generate)
        
        passed_factor_stats = self.build_stats_dict(passed_factor_names)
        if not passed_factor_names:
            logger.warning("警告：没有因子通过基础IC筛选")
            return [], {}

        # 第二步：类别内选择
        category_champions = self.select_category_champions(passed_factor_stats)

        if not category_champions:
            logger.warning("警告：没有类别冠军")
            return [], {}

        # 第三步：初步最终选择 （只是过滤数量的过滤而已），限制最多八个
        preliminary_selection = self.generate_final_selection(category_champions, passed_factor_stats)

        # 第四步：三层相关性控制哲学
        final_selection, correlation_report = self.apply_correlation_control(  # debug here
            preliminary_selection, passed_factor_stats
        )

        # 生成详细报告
        report = self._generate_selection_report(
            passed_factor_names, passed_factor_stats, category_champions, final_selection, correlation_report
        )

        logger.info("=" * 60)
        logger.info("滚动IC因子筛选完成！")
        logger.info(f"推荐用于IC加权合成: {final_selection}")
        logger.info("=" * 60)

        return final_selection, report

    def generate_final_selection(self, category_champions: Dict[str, List[str]],
                                 qualified_factors: Dict[str, FactorStats]) -> List[str]:
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
        all_champions.sort(key=lambda x: x[1].detail_score_21d['Final_Score'], reverse=True)

        # 选择前N名
        max_selection = min(len(all_champions), self.config.max_final_factors)
        final_selection = [name for name, _ in all_champions[:max_selection]]

        logger.info(f"最终选择 {len(final_selection)} 个因子:")
        for i, (name, stats) in enumerate(all_champions[:max_selection], 1):

            logger.info(f"   因子:{name}-------------------------------")
            logger.info(f"       评分: {stats.detail_score_21d['Final_Score']:.1f}")
            logger.info(f"       IC_mean_21d: {stats.ic_mean_21d:.3f}, IC_IR_21d: {stats.ic_ir_21d:.2f}")
            logger.info(f"      细节分数: {stats.detail_score_21d['Final_Score']}")

        return final_selection
    def run_factor_analysis(self, TARGET_STOCK_POOL: str, top_n_final: int = 5, correlation_threshold: float = 0.5,
                            run_version: str = None):
        RESULTS_PATH = workspaces_result_dir

        # --- 第一、二级火箭: 构建多周期冠军排行榜 ---
        champion_leaderboard = self.build_champion_leaderboard(
            results_path=RESULTS_PATH,
            target_stock_pool=TARGET_STOCK_POOL,
            run_version=run_version
        )
        print("\n--- 因子冠军排行榜 (已选出每个因子的最佳周期) ---")

        print(champion_leaderboard.head(10))

        # --- 第三级火箭: 从冠军排行榜中，筛选出最终的、多样化的顶级因子 ---
        # top_factors_df = self.get_top_factors(
        #     leaderboard_df=champion_leaderboard,
        #     results_path=RESULTS_PATH,
        #     stock_pool_index=TARGET_STOCK_POOL,
        #     quality_score_threshold=0.0,  # 建议设置一个有意义的门槛分，比如40分
        #     top_n_final=top_n_final,
        #     correlation_threshold=correlation_threshold
        # )
        print("\n--- 最终入选的顶级因子详情 (Diversified Top Factors) ---")
        print(champion_leaderboard)

        # --- 后续步骤: 为最终入选的因子生成详细报告 ---
        # ... (这里的逻辑与你之前的版本类似, 可以复用)
        logger.info("\n--- 开始为顶级因子生成详细报告 ---")
        for _, factor_row in champion_leaderboard.iterrows():
            factor_name = factor_row['factor_name']
            best_period = factor_row['best_period']

            print(f"正在为因子 '{factor_name}' (最佳周期: {best_period}) 生成报告...")
            print(f"正在为因子 '{factor_name}' 生成报告...")
            # 2. 生成您需要的报告
            viz_manager = self.visualization_manager
            # --- 选项 A：生成最全面的“业绩报告” ---
            viz_manager.plot_performance_report(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='o2o',
                run_version='latest'
            )

            # --- 选项 B：生成“特性诊断报告”，深入了解因子自身属性 ---
            viz_manager.plot_characteristics_report(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='o2o',
                run_version='latest'
            )

            # --- 选项 C：生成“归因面板”，直观对比预处理前后的效果 ---
            viz_manager.plot_attribution_panel(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='o2o',
                run_version='latest'
            )

            # --- 选项 D：生成“核心摘要”，用于快速浏览关键业绩 ---
            viz_manager.plot_ic_quantile_panel(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='o2o',
                run_version='latest'
            )
            # # 4.1 生成主报告 (3x2 统一评估报告)
            # # 绘图函数现在需要从硬盘加载数据，我们只需告知关键信息
            # self.visualization_manager.plot_unified_factor_report(
            #     backtest_base_on_index=TARGET_STOCK_POOL,
            #     factor_name=factor_name,
            #     results_path=RESULTS_PATH,  # <--- 传入成果库的根路径
            #     # 你可以决定主报告默认使用C2C还是O2C的结果
            #     default_config='o2o'
            # )
            #
            # # 4.2 调用新的分层净值报告函数
            # self.visualization_manager.plot_diagnostics_report(
            #     backtest_base_on_index=TARGET_STOCK_POOL,
            #     factor_name=factor_name,
            #     results_path=RESULTS_PATH,
            #     default_config='o2o'
            # )
            # # 调用新的归因分析面板函数
            # self.visualization_manager.plot_attribution_panel(
            #     backtest_base_on_index=TARGET_STOCK_POOL,
            #     factor_name=factor_name,
            #     results_path=RESULTS_PATH,
            #     default_config='o2o'
            # )
            #

    def _build_single_period_row(self, factor_dir: Path, period: str, run_version: str) -> Dict | None:
        """【辅助函数】为单个因子、单个周期构建用于打分的宽表行"""

        def _find_and_load_stats(factor_dir: Path, config_name: str, version: str = 'latest') -> Dict | None:
            config_path = factor_dir / config_name
            if not config_path.is_dir(): return None
            version_dirs = [d for d in config_path.iterdir() if d.is_dir()]
            if not version_dirs: return None
            target_version_path = sorted(version_dirs)[-1] if version == 'latest' else config_path / version
            if not target_version_path.exists(): return None
            summary_file = target_version_path / 'summary_stats.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f: return json.load(f)
            return None

        stats_o2o = _find_and_load_stats(factor_dir, 'o2o', run_version)
        if not stats_o2o: return None

        row = {'factor_name': factor_dir.name}
        for r_type, stats_data in [('o2o', stats_o2o)]:
            for d_type in ['raw', 'processed']:
                try:
                    ic_stats = stats_data.get(f'ic_analysis_{d_type}', {}).get(period, {})
                    q_stats = stats_data.get(f'quantile_backtest_{d_type}', {}).get(period, {})
                    if not ic_stats or not q_stats: continue  # 如果该周期数据不完整，则返回None

                    row[f'ic_mean_{d_type}_{r_type}'] = ic_stats.get('ic_mean')
                    row[f'ic_ir_{d_type}_{r_type}'] = ic_stats.get('ic_ir')
                    row[f'ic_t_stat_{d_type}_{r_type}'] = ic_stats.get('ic_t_stat')

                    row[f'tmb_sharpe_{d_type}_{r_type}'] = q_stats['tmb_sharpe']
                    row[f'tmb_max_drawdown_{d_type}_{r_type}'] = q_stats['tmb_max_drawdown']
                    row[f'monotonicity_spearman_{d_type}_{r_type}'] = q_stats['monotonicity_spearman']
                except:
                    continue

            fm_stats = stats_data.get('fama_macbeth', {}).get(period, {})
            row[f'fm_t_statistic_processed_{r_type}'] = fm_stats.get('t_statistic')

        return row

    def build_factor_ic_data(self,
                             run_version: str = 'latest') :
        base_path = workspaces_result_dir / self.resultLoadManager.pool_index
        ret = {}
        for factor_dir in base_path.iterdir():
            if not factor_dir.is_dir(): continue
            one_period = {}
            factor_name = factor_dir.name
            for period in self.ALL_PERIODS:
                # 1. 为当前因子和周期构建一个完整的指标行
                current_period_row = self._build_single_period_row(factor_dir, period, run_version)
                if current_period_row == None:
                    continue
                ic_ir = current_period_row['ic_ir_processed_o2o']
                ic_mean = current_period_row['ic_mean_processed_o2o']
                ic_t_stat = current_period_row['ic_t_stat_processed_o2o']
                one_period[period] = {'ic_mean':ic_mean,'ic_ir':ic_ir, 'ic_t_stat':ic_t_stat}
            if  len (one_period)!=0:
                ret[factor_name] = one_period
        return ret
    def build_champion_leaderboard(self, results_path: str, target_stock_pool: str,
                                   run_version: str = 'latest') -> pd.DataFrame:
        """
        【V4.0-多周期冠军版】 - 实现了第一和第二级火箭
        1. 扫描指定股票池下的所有因子。
        2. 对每个因子，遍历其所有测试周期，找到得分最高的“最佳周期”。
        3. 将所有因子的“冠军版本”汇总成一个排行榜。
        """
        logger.info(f"正在为股票池 [{target_stock_pool}] 构建多周期冠军排行榜...")
        champions_data = []
        base_path = Path(results_path) / target_stock_pool

        for factor_dir in base_path.iterdir():
            if not factor_dir.is_dir(): continue
            factor_name = factor_dir.name

            highest_score = -1
            best_period_champion_row = None

            # --- 第一级火箭：因子内部的“周期选美” ---
            for period in self.ALL_PERIODS:
                # 1. 为当前因子和周期构建一个完整的指标行
                current_period_row = self._build_single_period_row(factor_dir, period, run_version)
                if current_period_row is None:
                    logger.info(f"  > 因子 {factor_name} 在周期 {period} 数据不完整，已跳过。")
                    continue

                # 2. 为该周期的表现打分
                scores = calculate_factor_score_v33(current_period_row)

                # 3. 选出冠军
                if scores['Final_Score'] > highest_score:
                    highest_score = scores['Final_Score']
                    # 记录冠军信息：合并指标和分数，并加上最佳周期
                    best_period_champion_row = {
                        **current_period_row,
                        **scores,
                        'best_period': period
                    }

            # 选美结束后，记录冠军档案
            if best_period_champion_row:
                champions_data.append(best_period_champion_row)
                logger.info(f"✓ 因子 {factor_name} 的最佳周期为 [ {best_period_champion_row['best_period']} ], "
                            f"最高分: {best_period_champion_row['Final_Score']:.2f}")
            else:
                logger.warning(f"✗ 未能为因子 {factor_name} 在任何周期找到完整的测试结果。")

        # --- 第二级火箭：构建冠军排行榜 ---
        if not champions_data:
            raise ValueError(f"在路径 {base_path} 下，没有找到任何可以生成冠军排行榜的因子。")

        final_leaderboard = pd.DataFrame(champions_data).set_index('factor_name', drop=False)
        ##

        [['Final_Score','ic_mean_processed_o2o', 'ic_ir_processed_o2o', 'tmb_sharpe_processed_o2o',
         'tmb_max_drawdown_processed_o2o', 'monotonicity_spearman_processed_o2o', 'fm_t_statistic_processed_o2o',
         'Prediction_Score', 'Strategy_Score', 'Stability_Score', 'Purity_Score', 'Composability_Score', 'Final_Score','factor_name', 'ic_mean_raw_o2o', 'ic_ir_raw_o2o', 'tmb_sharpe_raw_o2o', 'tmb_max_drawdown_raw_o2o',
         'monotonicity_spearman_raw_o2o',
         'Grade', 'Factor_Direction', 'Composability_Passed', 'best_period']]
        #
        ret  = final_leaderboard.sort_values(by='Final_Score', ascending=False)
        return ret

    def get_top_factors(self, leaderboard_df: pd.DataFrame, results_path: str, stock_pool: str,
                        quality_score_threshold: float, top_n_final: int, correlation_threshold: float,
                        run_version: str = 'latest') -> pd.DataFrame:
        """
        【V2.0-升级版】从冠军排行榜中，筛选出最终的、多样化的顶级因子。
        """
        logger.info(f"--- 第三级火箭: 开始筛选多样化的顶级因子 ---")

        # 1. 质量筛选
        candidate_df = leaderboard_df[leaderboard_df['Final_Score'] >= quality_score_threshold].copy()
        if candidate_df.empty:
            logger.warning(f"没有因子的综合得分超过 {quality_score_threshold}。")
            return pd.DataFrame()
        logger.info(f"通过最低分数阈值，筛选出 {len(candidate_df)} 个高质量候选因子。")

        # 2. 多样化筛选 (去相关性)
        # 【核心升级】调用新版加载函数，该函数能处理不同的最佳周期
        factor_returns_matrix = self.load_fm_returns_for_champions(
            candidate_df=candidate_df,
            results_path=results_path,
            stock_pool=stock_pool,
            config='o2o',
            run_version=run_version
        )
        correlation_matrix = factor_returns_matrix.corr()

        final_selected_factors = []
        # 贪心算法：从得分最高的因子开始 (candidate_df已按分数排序)
        for factor_name in candidate_df.index:
            if len(final_selected_factors) >= top_n_final: break
            if not final_selected_factors:
                final_selected_factors.append(factor_name)
                continue

            correlations_with_selected = correlation_matrix.loc[factor_name, final_selected_factors].abs()
            if correlations_with_selected.max() < correlation_threshold:
                final_selected_factors.append(factor_name)

        logger.info(f"--- 筛选完成 ---")
        logger.info(f"最终选出 {len(final_selected_factors)} 个多样化顶级因子：{final_selected_factors}")

        return leaderboard_df.loc[final_selected_factors]

    def load_fm_returns_for_champions(self, candidate_df: pd.DataFrame, results_path: str, stock_pool: str,
                                      config: str, run_version: str) -> pd.DataFrame:
        """
        【V3.0-升级版】辅助函数：为冠军因子加载F-M收益序列，用于计算相关性。
        能够根据每个因子的 'best_period' 加载对应的收益文件。
        """
        all_returns = {}
        base_results_path = Path(results_path)

        # 遍历冠军因子DataFrame的每一行
        for factor_name, row in candidate_df.iterrows():
            period = row['best_period']  # <-- 【核心】获取该因子的最佳周期

            # --- 版本定位逻辑 ---
            factor_path = base_results_path / stock_pool / factor_name / config
            if not factor_path.is_dir(): continue
            version_dirs = [d for d in factor_path.iterdir() if d.is_dir()]
            if not version_dirs: continue
            target_version_path = sorted(version_dirs)[-1] if run_version == 'latest' else factor_path / run_version
            if not target_version_path.exists(): continue

            # --- 使用最佳周期构建动态文件路径 ---
            file_path = target_version_path / f"fm_returns_series_{period}.parquet"
            if file_path.exists():
                return_series = pd.read_parquet(file_path).squeeze()
                all_returns[factor_name] = return_series
            else:
                logger.warning(f"警告: 未找到文件: {file_path}")

        if not all_returns:
            logger.error(f"未能为任何候选因子加载F-M收益序列。")
            return pd.DataFrame()

        return pd.DataFrame(all_returns)

    def build_factor_categorie_maps(self):
        # 读取配置
        config = _load_local_config_functional()
        factor_definitions = config['factor_definition']

        maps = defaultdict(list)
        for factor in factor_definitions:
            maps[factor['style_category']].append(factor['name'])

        return dict(maps)
    def build_stats_dict(self, factor_names):
        ret = {}
        for f in factor_names:
            ret[f] = self.build_stats(f)
        return ret
    def build_stats(self, factor_name):
        summary_stats = self.resultLoadManager.get_summary_stats(factor_name)
        score = score_factor_from_stats_for_21d(summary_stats)
         # 构建结果
        factor_stats = FactorStats(
            factor_name=factor_name,
            ic_mean_21d=summary_stats['ic_analysis_processed']['21d']['ic_mean'],
            ic_ir_21d=summary_stats['ic_analysis_processed']['21d']['ic_ir'],
            detail_score_21d=score,
            top_q_turnover_dict=summary_stats['turnover'],#todo 记得后续改成：top_q_turnover
            # periods_data=aggregated_periods,
            # avg_ic_with_sign=avg_ic_with_sign,
            # avg_ir_ir_with_sign=avg_ic_ir_with_sign,
            # avg_ic_abs=avg_ic_abs,
            # avg_ir_abs=avg_ir_abs,
            # best_period_ic_ir=best_period_ic_ir,
            # nw_t_stat_series_mean=nw_t_stat_series_mean,
            # avg_stability=np.mean(all_stabilities) if all_stabilities else 0.0,
            # avg_ic_volatility=np.mean(all_ic_stds) if all_ic_stds else 0.0,
            # detail_score_21d=detail_score_21d,
            # snapshot_count=len(dates_range),
            # time_range=(min(dates_range), max(dates_range)) if dates_range else ('', ''),
            # # 将三个核心换手率指标填入返回结构
            # daily_rank_change_mean=final_turnover_stats['avg_daily_rank_change'],
            # daily_turnover_trend=final_turnover_stats['daily_turnover_trend'],
            # daily_turnover_volatility=final_turnover_stats['daily_turnover_volatility'],
            # turnover_adjusted_score=turnover_adjusted_score
        )
        return factor_stats


# 维持配置不变，因为我们会在代码中处理方向
PHASE1_SCREENING_CONFIG = {
    'min_full_sample_icir_abs': 0.4,   # 修正：我们现在关心ICIR的绝对值
    'min_full_sample_ic_mean_abs': 0.02, # 修正：IC均值的绝对值也应达标
    'min_newey_west_t_stat_abs': 1.96, # T值的绝对值要显著 (95%置信度)
    'min_win_rate': 0.55               # 胜率依然重要
}


def screen_factor_phase1(
        summary_row: Union[pd.Series, dict],
        config: Dict = None
) -> Tuple[bool, Dict]:
    """
    【V2版：因子准入筛选函数 - 方向中性】
    此版本基于ICIR的【绝对值】进行筛选，能同时识别正向和反向的有效因子。

    Returns:
        Tuple[bool, Dict]:
        - is_passed (bool): 是否通过筛选。
        - screening_results (Dict): 包含核心指标和【因子方向】的字典。
    """
    if config is None:
        config = PHASE1_SCREENING_CONFIG

    ic_mean = summary_row.get('full_sample_ic_mean', 0)
    ic_ir = summary_row.get('full_sample_icir', 0)
    nw_t_stat = summary_row.get('full_sample_nw_t_stat', 0)
    win_rate = summary_row.get('full_sample_win_rate', 0)

    # --- 核心修正 1：判断因子方向 ---
    # np.sign()会返回1, -1,或0。如果ic_mean接近0，我们默认为正向1。
    factor_direction = np.sign(ic_mean) if abs(ic_mean) > 1e-6 else 1

    # --- 核心修正 2：基于绝对值进行筛选 ---
    ic_mean_abs = abs(ic_mean)
    ic_ir_abs = abs(ic_ir)
    nw_t_stat_abs = abs(nw_t_stat)

    # 胜率需要根据方向重新计算：(IC * 方向) > 0 的比例
    # 假设 summary_row 里的 win_rate 是基于ic_mean方向算的，这里直接用

    screening_results = {
        'IC Mean': ic_mean,
        'ICIR': ic_ir,
        'NW T-stat': nw_t_stat,
        'Win Rate': win_rate,
        'Factor Direction': int(factor_direction)  # 新增：输出因子方向
    }

    if ic_ir_abs < config['min_full_sample_icir_abs']:
        screening_results['failure_reason'] = f"|ICIR| < {config['min_full_sample_icir_abs']}"
        return False, screening_results

    if ic_mean_abs < config['min_full_sample_ic_mean_abs']:
        screening_results['failure_reason'] = f"|IC Mean| < {config['min_full_sample_ic_mean_abs']}"
        return False, screening_results

    if nw_t_stat_abs < config['min_newey_west_t_stat_abs']:
        screening_results['failure_reason'] = f"|NW T-stat| < {config['min_newey_west_t_stat_abs']}"
        return False, screening_results

    if win_rate < config['min_win_rate']:
        screening_results['failure_reason'] = f"Win Rate < {config['min_win_rate']}"
        return False, screening_results

    return True, screening_results


def _generate_factor_profile_v4(
        factor_name: str,
        factor_stats: Dict[str, Dict]
) -> Dict:
    """
    【V4 最终版辅助函数】为一个通过筛选的因子生成深度画像和诊断结论。

    核心改进：
    1. 将 10d 周期的数据整合进短期效应的诊断逻辑中。
    2. 提供更丰富、更细致的短期效应画像（如“经典反转后走强”）。
    """
    profile = {
        "因子名称": factor_name,
        "决策指标 (21d)": {},
        "辅助诊断": {},
        "最终画像结论": "有待评估"
    }

    # --- 1. 提取所有周期的关键指标 ---
    icir_dict = {
        p: factor_stats.get(f'{p}d', {}).get("ic_ir", 0)
        for p in [1, 5, 10, 21, 40, 60, 120]
    }

    icir_21d = icir_dict[21]
    profile["决策指标 (21d)"]["21d 全样本ICIR"] = f"{icir_21d:.4f} (✅ 决策通过)"

    # --- 2. 【V4修正】诊断短期效应 (Short-term Effect), 引入10d数据 ---
    icir_1d = icir_dict[1]
    icir_5d = icir_dict[5]
    icir_10d = icir_dict[10]

    short_term_diagnosis_text = f"ICIR_1d={icir_1d:.2f}, ICIR_5d={icir_5d:.2f}, ICIR_10d={icir_10d:.2f}"

    # 建立更精细的判断逻辑
    if icir_1d < -0.05 and icir_10d > 0.02:
        short_term_conclusion = " (诊断：经典的短期反转后走强，形态非常健康)"
    elif icir_1d > 0.1 and icir_5d > 0.1 and icir_10d > 0.05:
        short_term_conclusion = " (⚠️ 警告：存在持续的强短期动量，高度疑似追高型因子)"
    elif icir_1d < -0.1 and icir_10d < -0.05:
        short_term_conclusion = " (⚠️ 警告：短期反转效应过强且持续，可能侵蚀中期信号)"
    else:
        short_term_conclusion = " (诊断：短期效应不明显或形态不典型)"
    profile["辅助诊断"]["短期效应 (1d, 5d, 10d)"] = short_term_diagnosis_text + short_term_conclusion

    # --- 3. 诊断信号持久性 (IC Decay) ---
    abs_icir_21d = abs(icir_21d)
    benchmark_icir = abs_icir_21d if abs_icir_21d > 1e-6 else 0.01

    decay_ratio_40d = abs(icir_dict[40]) / benchmark_icir
    decay_ratio_60d = abs(icir_dict[60]) / benchmark_icir
    decay_ratio_120d = abs(icir_dict[120]) / benchmark_icir

    persistence_diagnosis_text = (f"ICIR_40d={icir_dict[40]:.2f}, "
                                  f"ICIR_60d={icir_dict[60]:.2f}, "
                                  f"ICIR_120d={icir_dict[120]:.2f}")

    if decay_ratio_120d > 0.6:
        persistence_conclusion = " (诊断：信号非常持久，衰减极慢，顶级长效因子)"
    elif decay_ratio_60d < 0.3:
        persistence_conclusion = " (诊断：信号在中期(60d)衰减严重，不适合长周期持有)"
    elif decay_ratio_40d < 0.5:
        persistence_conclusion = " (诊断：信号在初期(40d)衰减较快，偏向中短周期)"
    else:
        persistence_conclusion = " (诊断：信号正常衰减，符合中长期因子特征)"
    profile["辅助诊断"]["信号持久性 (40d, 60d, 120d)"] = persistence_diagnosis_text + persistence_conclusion

    # --- 4. 【V4修正】形成最终结论 ---
    final_conclusion = "表现合格的中长期因子，可作为备选纳入合成池。"  # 默认结论

    if "顶级长效因子" in persistence_conclusion and "经典" in short_term_conclusion:
        final_conclusion = "顶级长效因子。信号持久且呈现健康的‘反转后走强’形态，Alpha来源干净。强烈建议作为核心基石。"
    elif "衰减严重" in persistence_conclusion or "衰减较快" in persistence_conclusion:
        final_conclusion = "中短周期因子。虽然通过了21d筛选，但其长期有效性存疑，在月度调仓策略中需谨慎使用或低配。"
    elif "警告：存在持续的强短期动量" in short_term_conclusion:
        final_conclusion = "可能被动量污染的因子。其Alpha来源不纯粹，稳定性风险较高，建议进一步做剥离分析或直接放弃。"
    elif "警告：短期反转效应过强且持续" in short_term_conclusion:
        final_conclusion = "短期反转特征过强，风险较高。虽然21d表现合格，但可能侵蚀了部分中期收益，需谨慎评估。"

    profile["最终画像结论"] = final_conclusion
    profile['ic_不同周期表现'] = show_diff_period_ic(factor_stats)

    return profile

def show_diff_period_ic(factor_stats):
    ic_dict = {
        p: round(factor_stats.get(f'{p}d', {}).get("ic_mean", 0), 3)
        for p in [1, 5, 10, 21, 40, 60, 120]
    }
    return ic_dict

# --- 1. 定义一个更全面的、多维度的筛选配置 ---
PHASE1_CONFIG_V3 = {
    'decision_period': 21,
    'min_icir_abs': 0.32,
    'min_ic_mean_abs': 0.02,
    'min_nw_t_stat_abs': 1.96
}


def profile_elite_factors(
        all_factors_summary: Dict[str, Dict],
        config: Dict = None
) -> Dict[str, Dict]:
    """
    """
    if config is None:
        config = PHASE1_CONFIG_V3
    decision_period_str = f"{config['decision_period']}d"

    print(
        f"筛选标准: |ICIR| >= {config['min_icir_abs']}, |IC Mean| >= {config['min_ic_mean_abs']}, |T-stat| >= {config['min_nw_t_stat_abs']}")

    factor_profiles = {}

    for factor_name, factor_stats in all_factors_summary.items():
        print(f"\n正在评估因子: {factor_name}...")

        stats_for_decision = factor_stats.get(decision_period_str)

        if not stats_for_decision:
            print(f"  > ❌ 筛选失败: 缺少决策周期 {decision_period_str} 的统计数据。")
            continue

        ic_mean = stats_for_decision.get('ic_mean', 0)
        ic_ir = stats_for_decision.get('ic_ir', 0)
        nw_t_stat = stats_for_decision.get('ic_t_stat', 0)#下此切换从ic_nw_t_stat todo

        # --- 执行“三道防火墙”检验 ---
        passed_effectiveness = abs(ic_mean) >= config['min_ic_mean_abs']
        passed_stability = abs(ic_ir) >= config['min_icir_abs']
        passed_significance = abs(nw_t_stat) >= config['min_nw_t_stat_abs']

        if passed_effectiveness and passed_stability and passed_significance:
            print(
                f"  > ✅ 通过所有筛选 (|IC Mean|={abs(ic_mean):.4f}, |ICIR|={abs(ic_ir):.4f}, |T-stat|={abs(nw_t_stat):.2f})")

            # 对通过的因子进行深度画像
            profile = _generate_factor_profile_v4(factor_name, factor_stats)
            factor_profiles[factor_name] = profile
        else:
            # 提供更详细的失败原因
            failure_reasons = []
            if not passed_effectiveness: failure_reasons.append(f"有效性不足(|IC Mean|={abs(ic_mean):.4f})")
            if not passed_stability: failure_reasons.append(f"稳定性不足(|ICIR|={abs(ic_ir):.4f})")
            if not passed_significance: failure_reasons.append(f"显著性不足(|T-stat|={abs(nw_t_stat):.2f})")
            print(f"  > ❌ 筛选失败: {', '.join(failure_reasons)}。")

    print("\n" + "=" * 50)
    print(f"筛选完成! 共 {len(factor_profiles)} 个因子进入精英池。")
    print("=" * 50)
    return factor_profiles


#
# def profile_elite_factors(
#         all_factors_summary: Dict[str, Dict],
#         decision_period: int = 21,
#         icir_threshold: float = 0.4
# ) -> Dict[str, Dict]:
#     """
#     【V2修正版主函数】执行两步走的因子筛选和画像流程。
#
#     核心修正：
#     1. 基于ICIR的【绝对值】进行硬性门槛筛选，以识别正向和反向因子。
#     2. 确保从嵌套字典中正确提取ic_ir值。
#     """
#     print(f"--- 开始因子筛选与画像 (V2版-方向中性) | 决策周期: {decision_period}d | |ICIR|门槛: {icir_threshold} ---")
#     factor_profiles = {}
#
#     for factor_name, factor_stats in all_factors_summary.items():
#         print(f"\n正在评估因子: {factor_name}...")
#
#         # --- 第一步：硬性门槛筛选 ---
#         decision_key = f'{decision_period}d'
#         icir_for_decision = factor_stats.get(decision_key)
#
#         if not icir_for_decision:
#             print(f"  > ❌ 筛选失败: 缺少决策周期 {decision_key} 的统计数据。")
#             continue
#
#
#         # --- 【核心逻辑修正】 ---
#         # 基于ICIR的绝对值进行判断
#         if abs(icir_for_decision) >= icir_threshold:
#             print(f"  > ✅ 通过硬性筛选 (|ICIR|={abs(icir_for_decision):.4f})")
#
#             # --- 第二步：对通过筛选的因子，进行“深度画像” ---
#             # _generate_factor_profile_v2 函数能正确处理正负ICIR并给出画像
#             profile = _generate_factor_profile_v2(factor_name, factor_stats)
#             factor_profiles[factor_name] = profile
#         else:
#             print(f"  > ❌ 筛选失败: |{decision_key} ICIR| ({abs(icir_for_decision):.4f}) 未达到门槛 {icir_threshold}。")
#
#     print("\n" + "=" * 50)
#     print(f"筛选完成! 共 {len(factor_profiles)} 个因子进入精英池。")
#     print("=" * 50)
#     return factor_profiles


def _normalize_score(value: float, worse_val: float, best_val: float) -> float:
    """一个简单的线性评分函数，将指标值映射到50-100分。"""
    # 处理反向指标的情况，确保 best_val 总是较大的那个
    if worse_val > best_val:
        value, worse_val, best_val = -value, -worse_val, -best_val

    if value >= best_val: return 100.0
    if value <= worse_val: return 50.0
    return 50 + 50 * (value - worse_val) / (best_val - worse_val)


def score_factor_from_stats_for_21d(
        factor_data: Dict,
        config: Dict = None
) -> Dict:
    """
    【定制版因子打分函数】
    根据你提供的特定数据结构，对一个已经通过硬性筛选的精英因子进行多维度打分。
    """
    # --- 1. 定义评分标准和权重 ---
    if config is None:
        config = {
            'power': {'worse': 0.02, 'best': 0.06},
            'stability_icir': {'worse': 0.3, 'best': 0.8},
            'stability_tstat': {'worse': 1.96, 'best': 3.0},
            'character_decay': {'worse': 0.2, 'best': 0.8},
            'cost_turnover': {'worse': 1.5, 'best': 0.5},  # 修正：turnover_mean 的量级是1.5左右
            'weights': {'power': 0.4, 'stability': 0.3, 'character': 0.15, 'cost': 0.15},
            'character_weights': {'decay': 0.7, 'reversal': 0.3}
        }

    # --- 2. 从你的数据结构中，安全地提取所有需要的原始指标 ---
    ic_stats = factor_data.get('ic_analysis_processed', {})
    turnover_stats = factor_data.get('turnover', {})

    stats_1d = ic_stats.get('1d', {})
    stats_21d = ic_stats.get('21d', {})
    stats_120d = ic_stats.get('120d', {})
    turnover_21d = turnover_stats.get('21d', {})

    ic_mean_21d = stats_21d.get('ic_mean', 0)
    ic_ir_21d = stats_21d.get('ic_ir', 0)
    # 适配：使用你数据中已有的 ic_t_stat
    t_stat_21d = stats_21d.get('ic_t_stat', 0)

    ic_ir_1d = stats_1d.get('ic_ir', 0)
    ic_ir_120d = stats_120d.get('ic_ir', 0)

    # 适配：使用21d周期的 turnover_mean
    avg_monthly_turnover = 0.5 # todo 注意后续重跑数据 注释取消掉 turnover_21d.get('turnover_mean', 1.0)  # 如果缺失，给一个中性偏高的惩罚值

    # --- 3. 计算各维度得分 ---

    # 维度一：核心预测能力
    power_score = _normalize_score(abs(ic_mean_21d), config['power']['worse'], config['power']['best'])

    # 维度二：信号稳定性
    stability_icir_score = _normalize_score(abs(ic_ir_21d), config['stability_icir']['worse'],
                                            config['stability_icir']['best'])
    stability_tstat_score = _normalize_score(abs(t_stat_21d), config['stability_tstat']['worse'],
                                             config['stability_tstat']['best'])
    stability_score = (stability_icir_score + stability_tstat_score) / 2

    # 维度三：因子“品格”
    benchmark_icir_abs = abs(ic_ir_21d) if abs(ic_ir_21d) > 1e-6 else 1.0
    decay_ratio = abs(ic_ir_120d) / benchmark_icir_abs
    decay_score = _normalize_score(decay_ratio, config['character_decay']['worse'], config['character_decay']['best'])

    reversal_score = 90.0 if ic_ir_1d < 0 else (70.0 if ic_ir_1d <= 0.1 else 30.0)
    character_score = (decay_score * config['character_weights']['decay'] +
                       reversal_score * config['character_weights']['reversal'])

    # 维度四：交易成本 (反向指标)
    cost_score = _normalize_score(avg_monthly_turnover, config['cost_turnover']['worse'],
                                  config['cost_turnover']['best'])

    # --- 4. 计算最终总分 ---
    final_score = (power_score * config['weights']['power'] +
                   stability_score * config['weights']['stability'] +
                   character_score * config['weights']['character'] +
                   cost_score * config['weights']['cost'])

    return {
        'Final_Score': round(final_score, 2),
        'Power_Score': round(power_score, 2),
        'Stability_Score': round(stability_score, 2),
        'Character_Score': round(character_score, 2),
        'Cost_Score': round(cost_score, 2)
    }


if __name__ == '__main__':
    config = SelectionConfig() #自己搭配！
    x = FactorSelector('20250906_045625_05e460ab',config)
    print(1)
    #
    #
    TARGET_UNIVERSE = INDEX_CODES['ZZ800']  # 以中证300为主战场
    # TARGET_UNIVERSE = INDEX_CODES['ZZ500']  # 以中证1000为主战场
    # TARGET_UNIVERSE = INDEX_CODES['ZZ800']  # 以中证1000为主战场
    #
    # selector.run_factor_analysis(
    #     TARGET_STOCK_POOL=TARGET_UNIVERSE,
    #     top_n_final=400,
    #     correlation_threshold=0.0,
    #     run_version='latest'
    # )
    # get_base_passed_factors(TARGET_UNIVERSE)

    x.run_complete_selection(TARGET_UNIVERSE)