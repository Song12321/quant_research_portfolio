"""
可视化管理器 - 统一管理所有图表和报告生成

提供标准化的可视化接口，支持：
1. 单因子测试结果可视化
2. 多因子优化结果可视化
3. 性能对比图表
4. 交互式仪表板
5. 专业报告生成

Author: Quantitative Research Team
Date: 2024-12-19
"""
import glob
import os
import re

import matplotlib.dates as mdates  # 导入日期格式化模块

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import gridspec
import matplotlib
matplotlib.use("TkAgg")   #  for fix ：'FigureCanvas'. Did you mean: 'FigureCanvasAgg'?
from matplotlib.font_manager import FontProperties
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import warnings
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger, log_warning

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from quant_lib.utils.json_utils import load_json_with_numpy

# 配置日志
logger = setup_logger(__name__)

##fname 为你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径，这里放到工程本地目录下。
cn_font = FontProperties(
    fname=r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\quant_lib\font\SourceHanSansSC-Regular.otf",
    size=12)
x1 = np.array([1, 2, 3, 4])
y2 = np.array([6, 2, 13, 10])

plt.plot(x1, y2)
plt.xlabel("X轴", fontproperties=cn_font)
plt.ylabel("Y轴", fontproperties=cn_font)
plt.title("测试", fontproperties=cn_font)


class VisualizationManager:
    """
    可视化管理器 - 统一管理所有图表生成
    
    功能：
    1. 单因子测试结果可视化
    2. 多因子优化结果可视化
    3. 性能对比和分析图表
    4. 交互式图表和仪表板
    """

    def __init__(self, output_dir: str = "visualizations", style: str = "default"):
        """
        初始化可视化管理器

        Args:
            output_dir: 图表输出目录
            style: 图表样式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置图表样式
        self._setup_style(style)
        sns.set_palette("husl")

        # 颜色配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }

        # logger.info(f"可视化管理器初始化完成，输出目录: {output_dir}")

    def _setup_style(self, style: str):
        """
        设置matplotlib样式

        Args:
            style: 样式名称
        """
        try:
            # 尝试使用指定的样式
            if style == "seaborn":
                # 如果是seaborn样式，使用seaborn-v0_8样式或默认样式
                available_styles = plt.style.available
                if 'seaborn-v0_8' in available_styles:
                    plt.style.use('seaborn-v0_8')
                elif 'seaborn-whitegrid' in available_styles:
                    plt.style.use('seaborn-whitegrid')
                else:
                    # 手动设置类似seaborn的样式
                    plt.rcParams.update({
                        'figure.facecolor': 'white',
                        'axes.facecolor': 'white',
                        'axes.edgecolor': 'black',
                        'axes.linewidth': 0.8,
                        'axes.grid': True,
                        'grid.color': 'gray',
                        'grid.alpha': 0.3,
                        'grid.linewidth': 0.5,
                        'font.size': 10,
                        'axes.labelsize': 10,
                        'axes.titlesize': 12,
                        'xtick.labelsize': 9,
                        'ytick.labelsize': 9,
                        'legend.fontsize': 9
                    })
            else:
                plt.style.use(style)
        except OSError:
            # 如果样式不存在，使用默认样式
            log_warning(f"样式 '{style}' 不可用，使用默认样式")
            plt.style.use('default')
    # ==========================================================================================
    #  一、核心公开方法 (Public Methods)
    # ==========================================================================================

    def plot_performance_report(self, backtest_base_on_index: str, factor_name: str, results_path: str,
                                default_config: str = 'o2o', run_version: str = 'latest') -> Optional[str]:
        """【业绩报告】生成单因子综合表现报告 (3x2布局)，专注盈利能力。"""
        logger.info(f"为因子 {factor_name} 生成综合表现报告...")
        data = self._load_report_data(backtest_base_on_index, factor_name, results_path, default_config, run_version)
        if not data: return None

        fig = plt.figure(figsize=(24, 30))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3)
        fig.suptitle(f'单因子 "{factor_name}" 综合表现报告\n(版本: {data["run_version_str"]})', fontproperties=cn_font,
                     fontsize=32, y=0.98)

        self._plot_ic_analysis_subplot(fig.add_subplot(gs[0, 0]), data, title="A. 因子IC序列与累计IC")
        self._plot_quantile_net_value_subplot(fig.add_subplot(gs[0, 1]), data,
                                              title=f'B. 分层累计净值 ({data["best_period"]})')
        self._plot_ic_vs_fm_subplot(fig.add_subplot(gs[1, 0]), data)
        self._plot_rolling_ic_subplot(fig.add_subplot(gs[1, 1]), data)
        self._plot_icir_sharpe_subplot(fig.add_subplot(gs[2, 0]), data)
        self._plot_summary_table_subplot(fig.add_subplot(gs[2, 1]), data)

        return self._save_figure(fig, data, "performance_report")

    def plot_characteristics_report(self, backtest_base_on_index: str, factor_name: str, results_path: str,
                                    default_config: str = 'o2o', run_version: str = 'latest') -> Optional[str]:
        """【特性报告】生成因子特性诊断报告 (2x2布局)，专注因子自身内在属性。"""
        logger.info(f"为因子 {factor_name} 生成特性诊断报告...")
        data = self._load_report_data(backtest_base_on_index, factor_name, results_path, default_config, run_version)
        if not data: return None

        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle(f'单因子 "{factor_name}" 特性诊断报告\n(版本: {data["run_version_str"]})', fontproperties=cn_font,
                     fontsize=28, y=0.97)

        self._plot_autocorrelation_subplot(fig.add_subplot(gs[0, 0]), data)
        self._plot_turnover_subplot(fig.add_subplot(gs[0, 1]), data)
        self._plot_style_exposure_subplot(fig.add_subplot(gs[1, 0]), data)
        self._plot_factor_distribution_subplot(fig.add_subplot(gs[1, 1]), data)

        return self._save_figure(fig, data, "characteristics_report")

    def plot_attribution_panel(self, backtest_base_on_index: str, factor_name: str, results_path: str,
                               default_config: str = 'o2o', run_version: str = 'latest') -> Optional[str]:
        """【归因报告】生成对比Raw vs. Processed分层表现的归因面板 (1x2布局)。"""
        logger.info(f"为因子 {factor_name} 生成因子归因分析面板...")
        data = self._load_report_data(backtest_base_on_index, factor_name, results_path, default_config, run_version)
        # 【容错】如果任一数据缺失，则无法生成此对比报告
        if not data or data.get('q_daily_returns_raw') is None or data.get('q_daily_returns_proc') is None:
            log_warning(f"因子 {factor_name} 缺少Raw或Processed每日分层收益数据，无法生成归因面板。")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(25, 8), sharey=True)
        self._plot_daily_quantile_subplot(axes[0], data['q_daily_returns_raw'], "A. 原始因子 (Raw Factor) 分层回测（每日收益累乘）")
        self._plot_daily_quantile_subplot(axes[1], data['q_daily_returns_proc'],
                                          "B. 纯净因子 (Processed Factor) 分层回测（每日收益累乘")
        fig.suptitle(f"因子 [{factor_name}] 价值归因分析: 处理前 vs. 处理后\n(版本: {data['run_version_str']})",
                     fontsize=20, y=1.02, fontproperties=cn_font)

        return self._save_figure(fig, data, "attribution_panel")

    def plot_ic_quantile_panel(self, backtest_base_on_index: str, factor_name: str, results_path: str,
                               default_config: str = 'o2o', run_version: str = 'latest') -> Optional[str]:
        """【核心摘要】生成IC分析和分层回测的核心摘要面板 (1x2布局)。"""
        logger.info(f"为因子 {factor_name} 生成IC与分层核心摘要面板...")
        data = self._load_report_data(backtest_base_on_index, factor_name, results_path, default_config, run_version)
        if not data: return None

        fig, axes = plt.subplots(1, 2, figsize=(25, 8))
        self._plot_ic_analysis_subplot(axes[0], data, title="A. 因子IC序列与累计IC")
        self._plot_quantile_net_value_subplot(axes[1], data, title=f'B. 最佳周期：分层累计净值 ({data["best_period"]})')
        fig.suptitle(f"因子 [{factor_name}] 核心分析面板\n(版本: {data['run_version_str']})", fontsize=20, y=1.02,
                     fontproperties=cn_font)

        return self._save_figure(fig, data, "ic_quantile_panel")

        # ==========================================================================================
        #  二、私有辅助函数 (Private Helper Methods)
        # ==========================================================================================

    def _save_figure(self, fig: plt.Figure, data: Dict, report_type: str) -> str:
        self._save_self_figure(fig, data, report_type)
        self._save_collective_figure(fig, data, report_type)
        

    def _save_collective_figure(self, fig: plt.Figure, data: Dict, report_type: str) -> str:
        """【私有】统一的图形保存函数。"""
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self_report_dir = data['collective_report_pics_path'] / report_type
            self_report_dir.mkdir(parents=True, exist_ok=True)
            pic_desc =  f"{data['factor_name']}_{report_type}_{data['default_config']}_{data['run_version_str']}.png"
            save_path = self_report_dir /pic_desc
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            path_str = str(save_path)
            logger.info(f"✓ {report_type} 已保存至: {path_str}")
            return path_str
        except Exception as e:
            logger.error(f"保存图表 {report_type} 时出错: {e}")
            return ""
        finally:
            plt.close(fig)

    def _save_self_figure(self, fig: plt.Figure, data: Dict, report_type: str) -> str:
        """【私有】统一的图形保存函数。"""
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self_report_dir = data['base_path'] / 'reports'
            self_report_dir.mkdir(parents=True, exist_ok=True)
            pic_desc =  f"{data['factor_name']}_{report_type}_{data['default_config']}_{data['run_version_str']}.png"
            save_path = self_report_dir /pic_desc
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            path_str = str(save_path)
            logger.info(f"✓ {report_type} 已保存至: {path_str}")
            return path_str
        except Exception as e:
            logger.error(f"保存图表 {report_type} 时出错: {e}")
            return ""
        finally:
            plt.close(fig)

    def _load_report_data(self, backtest_base_on_index, factor_name, results_path, default_config, run_version) -> \
    Optional[Dict]:
        """【私有】统一的数据加载和预处理中心。"""
        try:
            base_path = Path(results_path) / backtest_base_on_index / factor_name
            collective_report_pics = Path(results_path) / backtest_base_on_index / 'collective_report_pics'
            config_path = base_path / default_config
            target_version_path = self._find_target_version_path(config_path, run_version)
            if not target_version_path: raise FileNotFoundError(f"在 {config_path} 中未找到版本 '{run_version}'")

            stats = load_json_with_numpy(target_version_path / 'summary_stats.json')
            data = {'stats': stats, 'base_path': base_path,'collective_report_pics_path':collective_report_pics , 'target_version_path': target_version_path,
                    'run_version_str': target_version_path.name, 'factor_name': factor_name,
                    'default_config': default_config}

            ic_stats_proc = stats.get('ic_analysis_processed', {})
            q_stats_proc = stats.get('quantile_backtest_processed', {})
            fm_stats = stats.get('fama_macbeth', {})
            data['best_period'] ='21d' #self._find_best_period_by_rank(ic_stats_proc, q_stats_proc, fm_stats)

            def load_optional_parquet(file_name):
                path = target_version_path / file_name
                return pd.read_parquet(path) if path.exists() else None

            best_p = data['best_period']
            data['ic_series_proc_all'] = {int(re.search(r'(\d+)', p.name).group(1)): pd.read_parquet(p) for p in
                                          target_version_path.glob("ic_series_processed_*d.parquet")}
            data['processed_factor'] = load_optional_parquet("processed_factor.parquet")
            data['q_returns_proc'] = load_optional_parquet(f"quantile_returns_processed_{best_p}.parquet")
            data['ic_series_proc'] = data['ic_series_proc_all'].get(int(best_p[:-1]))
            data['fm_returns'] = load_optional_parquet(f"fm_returns_series_{best_p}.parquet")

            # 【容错加载】智能加载可选的“原始因子”和每日收益数据
            data['q_returns_raw'] = load_optional_parquet(f"quantile_returns_raw_{best_p}.parquet")
            data['q_daily_returns_raw'] = load_optional_parquet('q_daily_returns_df_raw.parquet')
            data['q_daily_returns_proc'] = load_optional_parquet('q_daily_returns_df_processed.parquet')

            return data
        except Exception as e:
            log_warning(f"加载报告数据时出错: {e}")
            return None

    def _find_target_version_path(self, config_path: Path, version: str) -> Optional[Path]:
        """
        【私有】【V2-鲁棒版】在配置路径下查找版本目录。
        'latest' 将根据文件夹的【修改时间】确定，而非名称排序。
        """
        if not config_path.is_dir(): return None
        version_dirs = [d for d in config_path.iterdir() if d.is_dir()]
        if not version_dirs: return None

        if version == 'latest':
            # 按目录的最后修改时间降序排序，第一个就是最新的
            latest_dir = sorted(version_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
            return latest_dir
        else:
            path_to_find = config_path / version
            return path_to_find if path_to_find in version_dirs else None

    def _find_best_period_by_rank(self, ic_stats: Dict, q_stats: Dict, fm_stats: Dict) -> str:
        """【私有】通过对多个核心指标进行综合排名，来选择最佳周期。"""
        if not ic_stats: return "21d"
        periods = list(ic_stats.keys())
        if not periods: return "21d"
        icir = pd.Series({p: ic_stats.get(p, {}).get('ic_ir', -np.inf) for p in periods})
        sharpe = pd.Series({p: q_stats.get(p, {}).get('tmb_sharpe', -np.inf) for p in periods})
        fmt = pd.Series({p: abs(fm_stats.get(p, {}).get('t_statistic', 0)) for p in periods})
        combined_rank = icir.rank(ascending=False) * 0.4 + sharpe.rank(ascending=False) * 0.4 + fmt.rank(
            ascending=False) * 0.2
        return combined_rank.idxmin()

        # ==========================================================================================
        #  三、模块化私有绘图函数 (Private Plotting Helpers)
        # ==========================================================================================

    def _plot_ic_analysis_subplot(self, ax: plt.Axes, data: Dict, title: str):
        """【私有绘图】绘制IC序列与累计IC图。"""
        ic_series_all = data.get('ic_series_proc_all', {})
        if not ic_series_all: return
        target_period = data.get('best_period', '21d')
        target_period_int = int(target_period[:-1])

        ax_twin = ax.twinx()

        target_ic_series = ic_series_all.get(target_period_int)
        if target_ic_series is not None:
            # ==========================================================
            # --- 【核心修正点】 ---
            # 使用 .flatten() 将 .values 从二维数组转为一维，解决TypeError
            ax.bar(target_ic_series.index, target_ic_series.values.flatten(),
                   color='cornflowerblue', alpha=0.7, label=f'IC序列 ({target_period})', width=1.5)
            # ==========================================================

        color_cycle = plt.cm.get_cmap('viridis', len(ic_series_all))
        for i, (period_int, ic_series) in enumerate(sorted(ic_series_all.items())):
            ic_series.cumsum().plot(ax=ax_twin, label=f'累计IC ({period_int}d)', color=color_cycle(i), lw=2.0)

        ax.set_title(title, fontproperties=cn_font, fontsize=16)
        ax.set_ylabel('IC值 (单期)', fontproperties=cn_font, color='royalblue')
        ax_twin.set_ylabel('累计IC', fontproperties=cn_font)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc='upper left', prop=cn_font)

    def _plot_quantile_net_value_subplot(self, ax: plt.Axes, data: Dict, title: str):
        """【私有绘图】绘制分层累计净值图(含Raw对比)。"""
        q_returns_proc = data.get('q_returns_proc')
        q_returns_raw = data.get('q_returns_raw')
        best_period = data.get('best_period', '21d')
        if q_returns_proc is None: return

        net_worth_df_proc = (1 + q_returns_proc).cumprod()
        ax_twin = ax.twinx()

        if 'TopMinusBottom' in net_worth_df_proc.columns:
            ax_twin.fill_between(net_worth_df_proc.index, 1, net_worth_df_proc['TopMinusBottom'], color='grey',
                                 alpha=0.3, label='多空组合 (纯净)')

        quantile_cols = sorted([col for col in net_worth_df_proc.columns if col.startswith('Q')])
        colors = plt.cm.get_cmap('coolwarm', len(quantile_cols))
        for i, col in enumerate(quantile_cols):
            net_worth_df_proc[col].plot(ax=ax, label=f'{col} (纯净)', color=colors(i), lw=2.0)

        # 【容错】仅当raw数据存在时，才绘制对比曲线
        if q_returns_raw is not None and 'TopMinusBottom' in q_returns_raw.columns:
            (1 + q_returns_raw['TopMinusBottom']).cumprod().plot(ax=ax_twin, label='多空组合 (原始)', linestyle='--',
                                                                 color='black', lw=1.5)

        ax.set_title(title, fontproperties=cn_font, fontsize=16)
        ax.set_ylabel('每日分层累计净值', fontproperties=cn_font)
        ax_twin.set_ylabel('多空组合累计净值', fontproperties=cn_font, color='gray')
        lines, labels = ax.get_legend_handles_labels();
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', prop=cn_font)

    def _plot_icir_sharpe_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制核心指标对比图 (纯净 vs. 原始)。"""
        stats = data['stats']
        ic_stats_proc, q_stats_proc = stats.get('ic_analysis_processed', {}), stats.get('quantile_backtest_processed',
                                                                                        {})
        ic_stats_raw, q_stats_raw = stats.get('ic_analysis_raw', {}), stats.get('quantile_backtest_raw', {})
        if not ic_stats_proc: return

        periods_numeric = sorted([int(p[:-1]) for p in ic_stats_proc.keys()])
        periods_str = [f'{p}d' for p in periods_numeric]
        icir_proc = [ic_stats_proc.get(p, {}).get('ic_ir', np.nan) for p in periods_str]
        sharpe_proc = [q_stats_proc.get(p, {}).get('tmb_sharpe', np.nan) for p in periods_str]
        ax.plot(periods_numeric, icir_proc, marker='o', lw=2.5, label='ICIR (纯净)')
        ax_twin = ax.twinx()
        ax_twin.plot(periods_numeric, sharpe_proc, marker='s', linestyle='-', color='C1', label='分层Sharpe (纯净)')

        # 【容错】仅当raw数据存在时，才绘制对比曲线
        if ic_stats_raw and q_stats_raw:
            icir_raw = [ic_stats_raw.get(p, {}).get('ic_ir', np.nan) for p in periods_str]
            sharpe_raw = [q_stats_raw.get(p, {}).get('tmb_sharpe', np.nan) for p in periods_str]
            ax.plot(periods_numeric, icir_raw, marker='o', lw=1.5, linestyle='--', color='C0', alpha=0.7,
                    label='ICIR (原始)')
            ax_twin.plot(periods_numeric, sharpe_raw, marker='s', lw=1.5, linestyle=':', color='C1', alpha=0.7,
                         label='分层Sharpe (原始)')

        ax.set_title('核心指标对比 (纯净 vs. 原始)', fontproperties=cn_font, fontsize=16)
        ax.set_xlabel('持有周期 (天)', fontproperties=cn_font);
        ax.set_ylabel('ICIR', fontproperties=cn_font)
        ax_twin.set_ylabel('分层Sharpe', fontproperties=cn_font)
        lines, labels = ax.get_legend_handles_labels();
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best', prop=cn_font)

    def _plot_rolling_ic_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制滚动IC图。"""
        ic_series_all = data.get('ic_series_proc_all', {})
        if not ic_series_all: return
        for period_int, ic_series in sorted(ic_series_all.items()):
            ic_series.rolling(window=120).mean().plot(ax=ax, label=f'滚动IC ({period_int}d, W=120d)')
        ax.set_title('因子滚动IC (稳定性)', fontproperties=cn_font, fontsize=16)
        ax.axhline(0, color='black', linestyle='--', lw=1)
        ax.legend(prop=cn_font)

    def _plot_autocorrelation_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制因子自相关性图。"""
        processed_factor = data.get('processed_factor')
        if processed_factor is None or processed_factor.empty: return
        mean_factor = processed_factor.mean(axis=1).dropna()
        if len(mean_factor) > 1:
            pd.plotting.autocorrelation_plot(mean_factor, ax=ax)
        ax.set_title('因子自相关性 (持续性)', fontproperties=cn_font, fontsize=16)

    def _plot_turnover_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制诊断图：年化换手率。"""
        turnover_stats = data['stats'].get('turnover', {})
        if not turnover_stats: return
        turnover_data = {p: d['turnover_annual'] for p, d in turnover_stats.items()}

        # --- 【核心修正点】 ---
        # 使用 idx.map() 来正确处理索引排序，确保X轴按周期从小到大显示
        pd.Series(turnover_data).sort_index(
            key=lambda idx: idx.map(lambda label: int(label.replace('d', '')))
        ).plot(kind='bar', ax=ax, alpha=0.7)
        # --- 【修正结束】 ---

        ax.set_title('年化换手率 (交易成本)', fontproperties=cn_font, fontsize=16)
        ax.set_ylabel('年化换手率', fontproperties=cn_font)
        ax.tick_params(axis='x', rotation=0)

    def _plot_style_exposure_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制风格暴露分析图。"""
        style_corr = data['stats'].get('style_correlation', {})
        if not style_corr: return
        pd.Series(style_corr).sort_values(ascending=True).plot(kind='barh', ax=ax)
        ax.axvline(0, color='black', linestyle='--', lw=1)
        ax.grid(True, axis='x')
        ax.set_title('风格暴露分析 (独特性)', fontproperties=cn_font, fontsize=16)

    def _plot_factor_distribution_subplot(self, ax: plt.Axes, data: Dict):
        """【私有绘图】绘制因子值分布图。"""
        processed_factor = data.get('processed_factor')
        if processed_factor is None or processed_factor.empty: return
        sample_factor = processed_factor.stack().sample(n=min(50000, len(processed_factor.stack())), random_state=42)
        sns.histplot(sample_factor, kde=True, ax=ax, stat="density")
        ax.set_title('因子值分布 (截面)', fontproperties=cn_font, fontsize=16)
        ax.set_xlabel('因子值', fontproperties=cn_font)
        ax.set_ylabel('密度', fontproperties=cn_font)

        # ... 其他绘图辅助函数 ...

    def _plot_ic_vs_fm_subplot(self, ax: plt.Axes, data: Dict):
        best_period, ic_series, fm_returns = data.get('best_period'), data.get('ic_series_proc'), data.get('fm_returns')
        if ic_series is None or fm_returns is None: return
        ax_twin = ax.twinx()
        ic_series.cumsum().plot(ax=ax, label=f'累计IC ({best_period})', lw=2.5, color='C0')
        (1 + fm_returns).cumprod().plot(ax=ax_twin, label=f'F-M纯净收益 ({best_period})', lw=2.5, color='C1',
                                        linestyle='--')
        ax.set_title(f'IC vs. F-M Alpha ({best_period})', fontproperties=cn_font, fontsize=16)
        ax.set_ylabel('累计IC', fontproperties=cn_font)
        ax_twin.set_ylabel('F-M纯净收益', fontproperties=cn_font, color='gray')
        lines, labels = ax.get_legend_handles_labels();
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', prop=cn_font)

    def _plot_summary_table_subplot(self, ax: plt.Axes, data: Dict):
        stats = data['stats']
        ic_stats_proc, q_stats_proc = stats.get('ic_analysis_processed', {}), stats.get('quantile_backtest_processed',
                                                                                        {})
        fm_stats, turnover_stats = stats.get('fama_macbeth', {}), stats.get('turnover', {})
        if not ic_stats_proc: return
        periods_str = sorted(ic_stats_proc.keys(), key=lambda x: int(x[:-1]))
        summary_data = []
        for p in periods_str:
            summary_data.append([
                p, f"{ic_stats_proc.get(p, {}).get('ic_ir', np.nan):.2f}",
                f"{q_stats_proc.get(p, {}).get('tmb_sharpe', np.nan):.2f}",
                f"{fm_stats.get(p, {}).get('t_statistic', np.nan):.2f}"
            ])
        columns = ['周期', 'ICIR', '分层Sharpe', 'F-M t值']
        ax.axis('off')
        table = ax.table(cellText=summary_data, colLabels=columns, loc='center')
        ax.set_title('核心指标汇总', fontproperties=cn_font, fontsize=18, y=0.85)
        table.auto_set_font_size(False);
        table.set_fontsize(14);
        table.scale(1.0, 2.5)
        for cell in table.get_celld().values(): cell.set_text_props(fontproperties=cn_font)

    def _plot_daily_quantile_subplot(self, ax: plt.Axes, returns_df: Optional[pd.DataFrame], title: str):
        if returns_df is None or returns_df.empty:
            ax.text(0.5, 0.5, "数据为空", ha='center', va='center', fontproperties=cn_font)
            return
        net_worth_df = (1 + returns_df).cumprod()
        ax_twin = ax.twinx()
        if 'TopMinusBottom' in returns_df.columns:
            tmb_net_worth = net_worth_df['TopMinusBottom']
            ax_twin.fill_between(tmb_net_worth.index, 1, tmb_net_worth, color='lightgray', alpha=0.8, label='多空组合')
        quantile_cols = sorted([col for col in net_worth_df.columns if col.startswith('Q')])
        colors = plt.cm.get_cmap('viridis', len(quantile_cols))
        for i, col in enumerate(quantile_cols):
            ax.plot(net_worth_df.index, net_worth_df[col], color=colors(i), label=f'{col}', linewidth=2)
        ax.set_title(title, fontsize=16, fontproperties=cn_font)
        ax.set_ylabel('每日分层累计净值', fontproperties=cn_font)
        ax_twin.set_ylabel('多空组合累计净值', fontproperties=cn_font, color='gray')
        lines, labels = ax.get_legend_handles_labels();
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', prop=cn_font)


def extrat_day_map_df(folder_path, name_prefix):
    folder = Path(folder_path)
    ic_dict = {}

    # 遍历匹配 ic_series_processed_xd.parquet 文件
    for file in folder.glob(f"{name_prefix}_*d.parquet"):
        # 用正则提取天数
        match = re.search(r"quantile_returns_processed_(\d+)d\.parquet", file.name)
        if name_prefix == 'ic_series_processed':
            match = re.search(r"ic_series_processed_(\d+)d\.parquet", file.name)
        if match:
            days = int(match.group(1))
            df = pd.read_parquet(file)
            ic_dict[days] = df

    return ic_dict
