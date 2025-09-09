import json
from pathlib import Path
from typing import Union, Dict, Any, List

import numpy as np
import pandas as pd
from jupyterlab.pytest_plugin import workspaces_dir

from projects._03_factor_selection.config_manager.base_config import INDEX_CODES, workspaces_result_dir
from projects._03_factor_selection.utils.factor_scoring_v33_final import calculate_factor_score_v33
from quant_lib import logger

from projects._03_factor_selection.visualization_manager import VisualizationManager
from typing import Union, Dict, Tuple

#
# def calculate_factor_score_ultimate(summary_row: Union[pd.Series, dict]) -> pd.Series:
#     def get_metric(key: str, default=0.0):
#         val = summary_row.get(key)
#         return default if pd.isna(val) else val
#
#     ic_ir_processed_o2c = get_metric('ic_ir_processed_o2c')
#     ic_mean_processed_o2c = get_metric('ic_mean_processed_o2c')
#     tmb_sharpe_proc_o2c = get_metric('tmb_sharpe_processed_o2c')
#     fm_t_stat_proc_o2c = get_metric('fm_t_statistic_processed_o2c')
#     tmb_max_drawdown_proc_o2c = get_metric('tmb_max_drawdown_processed_o2c')
#     monotonicity_spearman_proc_o2c = get_metric('monotonicity_spearman_processed_o2c', None)
#     tmb_sharpe_raw_o2c = get_metric('tmb_sharpe_raw_o2c')
#     factor_direction = 1
#     if ic_mean_processed_o2c < -1e-4:
#         factor_direction = -1
#     elif abs(ic_mean_processed_o2c) <= 1e-4 and fm_t_stat_proc_o2c < 0:
#         factor_direction = -1
#     base_score = 0
#     adj_ic_mean = ic_mean_processed_o2c * factor_direction
#     if adj_ic_mean > 0.05:
#         base_score += 20
#     elif adj_ic_mean > 0.03:
#         base_score += 15
#     elif adj_ic_mean > 0.01:
#         base_score += 10
#     adj_ic_ir = ic_ir_processed_o2c * factor_direction
#     if adj_ic_ir > 0.5:
#         base_score += 20
#     elif adj_ic_ir > 0.3:
#         base_score += 15
#     elif adj_ic_ir > 0.1:
#         base_score += 10
#     t_abs = abs(fm_t_stat_proc_o2c)
#     if t_abs > 3.0:
#         base_score += 30
#     elif t_abs > 2.0:
#         base_score += 25
#     elif t_abs > 1.5:
#         base_score += 15
#     adj_tmb_sharpe = tmb_sharpe_proc_o2c * factor_direction
#     perf_score = 0
#     if adj_tmb_sharpe > 1.0:
#         perf_score = 20
#     elif adj_tmb_sharpe > 0.5:
#         perf_score = 15
#     elif adj_tmb_sharpe > 0.2:
#         perf_score = 10
#     if tmb_max_drawdown_proc_o2c < -0.5: perf_score -= 5
#     base_score += max(0, perf_score)
#     if pd.notna(monotonicity_spearman_proc_o2c) and abs(monotonicity_spearman_proc_o2c) >= 0.5:
#         base_score += abs(monotonicity_spearman_proc_o2c) * 10
#     robustness_penalty = 0
#     purity_penalty = 0
#     if tmb_sharpe_raw_o2c * factor_direction > 0.3:
#         denominator = max(abs(tmb_sharpe_raw_o2c), 1e-6)
#         decay_ratio = (tmb_sharpe_raw_o2c - tmb_sharpe_proc_o2c) / denominator
#         if decay_ratio > 0.5: purity_penalty += 15
#         if decay_ratio > 0.8: purity_penalty += 25
#     final_score = base_score - robustness_penalty - purity_penalty
#     return pd.Series({
#         'Base_Score': base_score,
#         'Robustness_Penalty': robustness_penalty,
#         'Purity_Penalty': purity_penalty,
#         'Final_Score': max(0, final_score)
#     })


class FactorSelectorV2:
    def __init__(self):
        # self.visualizationManager = VisualizationManager()
        # 假设因子测试中所有可能用到的周期都在这里定义
        self.ALL_PERIODS = ['1d', '5d', '10d', '21d', '40d', '60d', '120d']
        self.visualization_manager = VisualizationManager(
        )

        print("FactorSelectorV2 (专业级因子筛选平台) 已准备就绪。")

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

    def build_factor_icir_data(self, target_stock_pool: str,
                                   run_version: str = 'latest') :
        base_path = workspaces_result_dir / target_stock_pool
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
    'min_icir_abs': 0.4,
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




#最新版评价因子！挑选因子！
#全局ic ir 进行评价！
def get_passed_factors(
        summary_row: Union[pd.Series, dict]=None,
        config: Dict = None
):

    selector = FactorSelectorV2()
    all_factors_summary_data = selector.build_factor_icir_data(TARGET_UNIVERSE)

    # --- 2. 执行筛选与画像 ---
    elite_factor_reports = profile_elite_factors(
        all_factors_summary=all_factors_summary_data
    )
    names=[]
    # --- 3. 查看精英因子的深度画像报告 ---
    for factor_name, report in elite_factor_reports.items():
        names.append(factor_name)
        print(f"\n----- {factor_name} 精英因子报告 -----")
        # 使用json美化输出
        print(json.dumps(report, indent=4, ensure_ascii=False))
    print("\n" + "=" * 50)
    print(f"因子list：：{names}")
if __name__ == '__main__':

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
    get_passed_factors()
