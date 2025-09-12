from typing import Callable  # 引入Callable来指定函数类型的参数

import numpy as np
import pandas as pd
import pandas_ta as ta

from data.local_data_load import load_index_daily, load_cashflow_df, load_income_df, \
    load_balancesheet_df, load_fina_indicator_df
from projects._03_factor_selection.factor_manager.factor_calculator.momentum.sw_factor import \
    IndustryMomentumFactor
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap
from projects._03_factor_selection.utils.date.trade_date_utils import map_ann_dates_to_tradable_dates
from quant_lib import logger


## 数据统一 tushare 有时候给元 千元 万元!  现在需要达成:统一算元!
#remind:  turnover_rate 具体处理 都要/100
# total_mv, circ_mv 具体处理 *10000
# amount 具体处理  *1000
class FactorCalculator:
    """
    【新增】因子计算器 (Factor Calculator)
    这是一个专门负责具体因子计算逻辑的类。
    它将所有的计算细节从 FactorManager 中分离出来，使得代码更清晰、更易于扩展。
    只做纯粹的计算，shift 以及对齐where股票池，下游自己处理！！！ remind
    """

    ##
    #
    # caclulate_函数 ，无需关心对齐，反正下游会align where 动态股票池！
    #  注意：涉及到rolling shift 操作，需要关注数据连续性。要考虑填充！
    #   ---ex：close可能是空的，因为停牌日就是空的，是nan，我们可以适当ffill
    #
    # #

    def __init__(self, factor_manager):
        """
        初始化因子计算器。

        Args:
            factor_manager: FactorManager 的实例。计算器需要通过它来获取依赖的因子。
        """
        # 注意：这里持有 FactorManager 的引用，以便在计算衍生因子时，
        # 可以通过 factor_manager.get_raw_factor() 来获取基础因子，并利用其缓存机制。
        self.factor_manager = factor_manager
        print("FactorCalculator (因子计算器) 已准备就绪。")



    # === 规模 (Size) ===
    def _calculate_log_circ_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_raw_factor('circ_mv').copy()
        # 2. 【核心步骤】因为市值是“状态量”，所以使用ffill填充非交易日
        #    这确保了公司价值状态的连续性。
        circ_mv_df.ffill(limit=65)
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)

        factor_df = circ_mv_df.apply(np.log)
        return factor_df
    def _calculate_log_total_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_raw_factor('total_mv').copy()
        # 2. 【核心步骤】因为市值是“状态量”，所以使用ffill填充非交易日
        #    这确保了公司价值状态的连续性。
        circ_mv_df.ffill(limit=65)
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        # 使用 pandas 自带 log 函数，保持类型一致
        factor_df = circ_mv_df.apply(np.log)
        return factor_df

    # === 价值 (Value) - 【V2.0 - 第一性原理版】 ===

    def _calculate_bm_ratio(self) -> pd.DataFrame:
        """计算账面市值比 (Book-to-Market Ratio)。"""
        return self._create_financial_ratio_factor(
            numerator_factor='total_equity',
            denominator_factor='total_mv',
            numerator_must_be_positive=True
        )

    def _calculate_ep_ratio(self) -> pd.DataFrame:
        """
        【V2.0 - 第一性原理版】计算盈利收益率 (Earnings Yield)。
        E/P = net_profit_ttm / total_mv
        """
        return self._create_financial_ratio_factor(
            numerator_factor='net_profit_ttm',
            denominator_factor='total_mv',
            numerator_must_be_positive=False  # 盈利可以是负数
        )

    def _calculate_sp_ratio(self) -> pd.DataFrame:
        """
        【- 第一性原理版】计算销售收益率 (Sales Yield)。
        S/P = total_revenue_ttm / total_mv
        """
        return self._create_financial_ratio_factor(
            numerator_factor='total_revenue_ttm',
            denominator_factor='total_mv',
            numerator_must_be_positive=True
        )
    #ok
    def _calculate_cfp_ratio(self) -> pd.DataFrame:
        """
        【  第一性原理版】计算现金流市值比 (Cash Flow Yield)。
        CF/P = cashflow_ttm / total_mv
        """
        return self._create_financial_ratio_factor(
            numerator_factor='cashflow_ttm',
            denominator_factor='total_mv',
            numerator_must_be_positive=False  # 经营现金流可以是负数
        )

    # === 质量 (Quality) ===
    #ok
    def _calculate_roa_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的总资产报酬率 (ROA_TTM)。

        金融逻辑:
        ROA衡量公司利用所有资产（包括负债）创造利润的能力。高ROA代表
        更轻资产的运营模式或更高的资产效率。

        公式: net_profit_ttm / total_assets
        """
        logger.info("--- 开始计算最终因子: roa_ttm ---")
        return self._create_financial_ratio_factor('net_profit_ttm','total_assets')
    def _calculate_roe_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月的净资产收益率 (ROE_TTM)。

        金融逻辑:
        ROE是衡量公司为股东创造价值效率的核心指标。高ROE意味着公司能用更少的
        股东资本创造出更多的利润，是“好生意”的标志。

        注意: 这是一个依赖财报数据的复杂因子，其计算逻辑与 cashflow_ttm 类似。
              你需要确保你的 DataManager 能够提供包含 'net_profit' 和 'total_equity'
              的季度财务报表数据。
        """

        # --- 步骤一：获取分子和分母 ---
        # 调用我们刚刚实现的两个生产级函数
        net_profit_ttm_df = self._calculate_net_profit_ttm()
        quarterly_equity_df = self._calculate_total_equity()

        # --- 步骤二：对齐数据 ---
        # align确保两个DataFrame的索引和列完全一致，避免错位计算
        # join='inner'会取两个因子都存在的股票和日期，是最安全的方式
        profit_aligned, equity_aligned = net_profit_ttm_df.align(quarterly_equity_df, join='inner', axis=None)
        equity_lagged_4q = equity_aligned.shift(4)
        # 平均净资产 = (期初 + 期末) / 2
        average_equity = (equity_aligned + equity_lagged_4q) / 2
        # --- 步骤三：风险控制与计算 ---
        # 核心风控：股东权益可能为负（公司处于资不抵债状态）。
        # 在这种情况下，ROE的计算没有经济意义，且会导致计算错误。
        # 我们将分母小于等于0的地方替换为NaN，这样除法结果也会是NaN。
        # 例如，2021年-2023年，一些陷入困境的地产公司净资产可能为负，其ROE必须被视为无效值。
        average_equity_safe = average_equity.where(average_equity > 0, np.nan)
        roe_ttm_df = profit_aligned / average_equity_safe

        # --- 步骤四：后处理 ---
        # 尽管我们处理了分母为0的情况，但仍可能因浮点数问题产生无穷大值。
        # 统一替换为NaN，确保因子数据的干净。
        roe_ttm_df=roe_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=False)
        return roe_ttm_df


    def _calculate_gross_margin_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的销售毛利率 (Gross Margin TTM)。
        公式: (Revenue TTM - Operating Cost TTM) / Revenue TTM
        """
        print("--- 开始计算最终因子: gross_margin_ttm ---")

        # --- 步骤一：获取分子和分母的组成部分 ---
        revenue_ttm_df = self._calculate_total_revenue_ttm()
        op_cost_ttm_df = self._calculate_op_cost_ttm()

        # --- 步骤二：对齐数据 ---
        # 确保revenue和op_cost的索引和列完全一致，避免错位计算
        revenue_aligned, op_cost_aligned = revenue_ttm_df.align(op_cost_ttm_df, join='inner', axis=None)

        # --- 步骤三：风险控制与计算 ---
        # 核心风控：分母(营业收入)可能为0或负数(在极端或错误数据情况下)。
        # 我们将分母小于等于0的地方替换为NaN，这样除法结果也会是NaN，避免产生无穷大值。
        revenue_aligned_safe = revenue_aligned.where(revenue_aligned > 0, np.nan)

        print("1. 计算 Gross Margin TTM，并对分母进行风险控制(>0)...")
        gross_margin_ttm_df = (revenue_aligned - op_cost_aligned) / revenue_aligned_safe

        # --- 步骤四：后处理 (可选但推荐) ---
        # 理论上，毛利率不应超过100%或低于-100%太多，但极端情况可能出现。
        # 这里可以根据需要进行clip或winsorize，但暂时保持原样以观察原始分布。
        # 再次确保没有无穷大值。
        return  gross_margin_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=False)

    def _calculate_debt_to_assets(self) -> pd.DataFrame:
        """
        【生产级】计算每日可用的最新资产负债率。
        公式: Total Debt / Total Assets
        """
        print("--- 开始计算最终因子: debt_to_assets ---")

        # --- 步骤一：获取分子和分母 ---
        total_debt_df = self._calculate_total_debt()
        total_assets_df = self._calculate_total_assets()

        # --- 步骤二：对齐数据 ---
        debt_aligned, assets_aligned = total_debt_df.align(total_assets_df, join='inner', axis=None)

        # --- 步骤三：风险控制与计算 ---
        # 核心风控：分母(总资产)可能为0或负数。
        assets_aligned_safe = assets_aligned.where(assets_aligned > 0, np.nan)

        print("1. 计算 Debt to Assets，并对分母进行风险控制(>0)...")
        debt_to_assets_df = debt_aligned / assets_aligned_safe

        # --- 步骤四：后处理 ---
        return debt_to_assets_df.replace([np.inf, -np.inf], np.nan, inplace=False)
    # === 成长 (Growth) ===
    def _calculate_net_profit_growth_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月归母净利润的同比增长率(TTM YoY Growth)。
        """
        logger.info("    > 正在计算因子: net_profit_growth_ttm...")

        # 直接调用通用的TTM增长率计算引擎
        return self._calculate_financial_ttm_growth_factor(
            factor_name='net_profit_growth_ttm',
            ttm_factor_name='net_profit_ttm'  # 指定依赖的TTM因子名
        )

    def _calculate_revenue_growth_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月营业总收入的同比增长率(TTM YoY Growth)。
        """
        # 同样调用通用的TTM增长率计算引擎
        return self._calculate_financial_ttm_growth_factor(
            factor_name='revenue_growth_ttm',
            ttm_factor_name='total_revenue_ttm'  # 指定依赖的TTM因子名
        )

    #ok
    def _calculate_net_profit_growth_yoy(self) -> pd.DataFrame:
        """
        【生产级】计算单季度归母净利润的同比增长率 (YoY)。
        """
        return self._create_general_quarterly_factor_engine(
            factor_name='net_profit_growth_yoy',
            data_loader_func=load_income_df,
            source_column='n_income_attr_p',
            calculation_logic_func=self._yoy_growth_logic
        )
    #ok
    def _calculate_total_revenue_growth_yoy(self) -> pd.DataFrame:
        """
        【生产级】计算单季度营业收入的同比增长率 (YoY)。
        """
        return self._create_general_quarterly_factor_engine(
            factor_name='total_revenue_growth_yoy',
            data_loader_func=load_income_df,
            source_column='total_revenue',
            calculation_logic_func=self._yoy_growth_logic
        )
 
    # === 动量与反转 (Momentum & Reversal) ===
    ###材料 是否需要填充 介绍:
    ##
    # ：动量/反转类因子
    # 因子示例: _calculate_momentum_120d, _calculate_reversal_21d, _calculate_momentum_12_1, _calculate_momentum_20d
    #
    # 计算特性: 它们的核心逻辑是 price(t) / price(t-N) - 1。这类计算对价格的绝对时间间隔非常敏感。
    #
    # 推荐使用: self.factor_manager.get_raw_factor('close_hfq') (未经填充的版本)
    #
    # 理由:
    #
    # 想象一下momentum_12_1的计算：close_hfq.shift(21) / close_hfq.shift(252) - 1。
    #
    # 如果一只股票在t-252之后停牌了半年，然后复牌。如果你使用了close_hfq_filled，那么close_hfq.shift(252)取到的就是一个非常“陈腐”的、半年前的价格。用这个陈腐价格计算出的动量值，其经济学意义是存疑的。
    #
    # 更稳健的做法是使用未经填充的close_hfq。如果在t-21或t-252的任一时间点，股票是停牌的（值为NaN），那么最终的动量因子值也应该是NaN。我们宁愿在没有可靠数据时得到一个NaN，也不要一个基于陈腐数据计算出的错误值。#
    def _calculate_momentum_120d(self) -> pd.DataFrame:
        """
        计算120日（约半年）动量/累计收益率。

        金融逻辑:
        捕捉市场中期的价格惯性，即所谓的“强者恒强，弱者恒弱”的趋势。
        这是构建趋势跟踪策略的基础。
        """
        logger.info("    > 正在计算因子: momentum_120d...")
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()

        # 2. 计算120个交易日前的价格到今天的收益率
        #    使用 .pct_change() 是最直接且能处理NaN的pandas原生方法
        momentum_df = close_df.pct_change(periods=120)

        logger.info("    > momentum_120d 计算完成。")
        return momentum_df
    def _calculate_momentum_pct_60d(self) -> pd.DataFrame:
        """
        计算70日（约半年）动量/累计收益率。
        金融逻辑:
        捕捉市场中期的价格惯性，即所谓的“强者恒强，弱者恒弱”的趋势。
        这是构建趋势跟踪策略的基础。
        """
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()
        # 2. 计算120个交易日前的价格到今天的收益率
        #    使用 .pct_change() 是最直接且能处理NaN的pandas原生方法
        momentum_df = close_df.pct_change(periods=60)
        return momentum_df

    def _calculate_reversal_21d(self) -> pd.DataFrame:
        """
        计算21日（约1个月）反转因子。
        金融逻辑:
        A股市场存在显著的短期均值回归现象。即过去一个月涨幅过高的股票，
        在未来倾向于下跌；反之亦然。因此，我们将短期收益率取负，
        得到的分数越高，代表其反转（上涨）的可能性越大。
        """
        logger.info("    > 正在计算因子: reversal_21d...")
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()
        # 2. 计算21日收益率
        return_21d = close_df.pct_change(periods=21)
        # 3. 将收益率取负，即为反转因子
        return   -return_21d

    def _calculate_reversal_5d(self) -> pd.DataFrame:
        """
        计算周 反转因子。
        金融逻辑:
        A股市场存在显著的短期均值回归现象。即过去一个月涨幅过高的股票，
        在未来倾向于下跌；反之亦然。因此，我们将短期收益率取负，
        得到的分数越高，代表其反转（上涨）的可能性越大。
        """
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()
        return_21d = close_df.pct_change(periods=5)
        # 3. 将收益率取负，即为反转因子
        return -return_21d
    def _calculate_momentum_12_1(self) -> pd.DataFrame:
        """
        计算过去12个月剔除最近1个月的累计收益率 (Momentum 12-1)。
        金融逻辑:
        这是最经典的动量因子，由Jegadeesh和Titman提出。它剔除了最近一个月的
        短期反转效应，旨在捕捉更稳健的中期价格惯性。
        """
        # 1. 获取收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        # close_df.ffill(axis=0, inplace=True) #反驳：如果人家停牌一年，你非fill前一年的数据，那误差太大了 不行！
        # 2. 计算 T-21 (约1个月前) 的价格 与 T-252 (约1年前) 的价格之间的收益率
        #    shift(21) 获取的是约1个月前的价格
        #    shift(252) 获取的是约12个月前的价格
        momentum_df = close_df.shift(21) / close_df.shift(252) - 1
        return momentum_df
    def _calculate_momentum_12_2(self) -> pd.DataFrame:
        """
        计算过去12个月剔除最近1个月的累计收益率 (Momentum 12-1)。
        金融逻辑:
        这是最经典的动量因子，由Jegadeesh和Titman提出。它剔除了最近2个月的
        短期反转效应，旨在捕捉更稳健的中期价格惯性。
        """
        # 1. 获取收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        # close_df.ffill(axis=0, inplace=True) #反驳：如果人家停牌一年，你非fill前一年的数据，那误差太大了 不行！
        # 2. 计算 T-21 (约1个月前) 的价格 与 T-252 (约1年前) 的价格之间的收益率
        #    shift(21) 获取的是约1个月前的价格
        #    shift(252) 获取的是约12个月前的价格
        momentum_df = close_df.shift(21*2) / close_df.shift(252) - 1
        return momentum_df

    def _calculate_momentum_1d(self) -> pd.DataFrame:
                return self.get_momentum_n_d(1)
    def _calculate_momentum_5d(self) -> pd.DataFrame:
        return self.get_momentum_n_d(5)
    def get_momentum_n_d(self,n_day) -> pd.DataFrame:
        close_hfq = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        mom_nd = close_hfq.pct_change(periods=n_day)
        return mom_nd
    def _calculate_momentum_20d(self) -> pd.DataFrame:
        """
        计算20日动量/收益率。

        金融逻辑:
        捕捉短期（约一个月）的价格惯性，即所谓的“强者恒强”。
        """
        print("    > 正在计算因子: momentum_20d...")
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)

        momentum_df = close_df.pct_change(periods=20)
        return momentum_df

    # === 风险 (Risk) ===

    def _calculate_beta(self, benchmark_index, window: int = 60,
                        min_periods: int = 20) -> pd.DataFrame:
        """
        1. 从FactorManager获取个股和指定的市场收益率。
        2. 准备数据缓冲期。
        3. 调用纯函数 `calculate_rolling_beta_pure` 进行计算。
        """
        logger.info(f"调度Beta计算任务 (基准: {benchmark_index}, 窗口: {window}天)...")

        # --- 1. 获取原材料 ---
        stock_returns = self.factor_manager.get_raw_factor('pct_chg')
        market_returns = self._calculate_market_pct_chg(index_code=benchmark_index)

        # --- 2. 准备 ---
        config = self.factor_manager.data_manager.config['backtest']
        start_date, end_date = config['start_date'], config['end_date']

        buffer_days = int(window * 1.7) + 5
        buffer_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')

        stock_returns_buffered = stock_returns.loc[buffer_start_date:end_date] #必须多点前缀数据,后面好进行rolling
        market_returns_buffered = market_returns.loc[buffer_start_date:end_date]

        # --- 3. 执行计算 ---
        beta_df_full = calculate_rolling_beta_pure(
            stock_returns=stock_returns_buffered,
            market_returns=market_returns_buffered,
            window=window,
            min_periods=min_periods
        )
        return beta_df_full

        # 重构后的波动率因子

    def _calculate_volatility_90d(self) -> pd.DataFrame:
        """
       计算90日年化波动率。

       金融逻辑:
       衡量个股在过去约半年内的价格波动风险。经典的“低波动异象”认为，
       低波动率的股票长期来看反而有更高的风险调整后收益。

       数据处理逻辑:
       - 停牌期间的收益率为NaN，这是正确的，不应该填充为0
       - rolling.std()会自动忽略NaN值进行计算
       - min_periods=60确保至少有60个有效交易日才计算波动率
       """
        return self._create_rolling_volatility_factor(window=90, min_periods=45)

    def _calculate_volatility_120d(self) -> pd.DataFrame:
        return self._create_rolling_volatility_factor(window=120, min_periods=60)

    def _calculate_volatility_40d(self) -> pd.DataFrame:
        return self._create_rolling_volatility_factor(window=40, min_periods=20)


    # #换手率因子
    # def _calculate_turnover_20d_std(self) -> pd.DataFrame:
    #

    def _calculate_turnover_42d_std_252d_std_ratio(self) -> pd.DataFrame:
        """
        【新增】计算换手率波动性的变化率因子。
        金融逻辑:
        衡量个股最近2个月的日换手率波动性，相对于其过去2年的“正常”波动水平的变化。
        一个远大于0的值，可能表示该股票近期交易行为剧变，风险或关注度急剧增加。
        一个接近0或为负的值，表示近期交易行为相对平稳或萎缩。
        计算公式:
        (最近2个月换手率标准差 / 最近2年换手率标准差) - 1
        """
        logger.info("    > 正在计算因子: turnover_std_ratio (换手率波动率变化)...")

        # --- 1. 获取原材料：日换手率 ---
        # 我们使用原始的、未经填充的换手率，因为rolling计算会自动处理NaN，
        # 填充为0反而会人为降低波动率的计算结果。
        turnover_df = self.factor_manager.get_raw_factor('turnover_rate').copy()

        # --- 2. 定义时间窗口 (交易日) ---
        short_term_window = 42  # 约2个月
        long_term_window = 504  # 约2年

        # 设定计算所需的最小样本数，通常为窗口的70%-80%
        min_periods_short = int(short_term_window * 0.8)
        min_periods_long = int(long_term_window * 0.8)

        # --- 3. 计算短期波动率（分子） ---
        std_short_term = turnover_df.rolling(
            window=short_term_window,
            min_periods=min_periods_short
        ).std()

        # --- 4. 计算长期波动率（分母） ---
        std_long_term = turnover_df.rolling(
            window=long_term_window,
            min_periods=min_periods_long
        ).std()

        # --- 5. 风险控制与最终计算 ---
        # a) 对齐数据，确保计算的准确性
        std_short_aligned, std_long_aligned = std_short_term.align(std_long_term, join='inner', axis=None)

        # b) 【核心风控】分母（长期波动率）可能为0或极小，必须处理以防除0错误
        # 我们将小于1e-6的值视同为0，并替换为NaN，避免产生无穷大(inf)的因子值
        std_long_safe = std_long_aligned.where(std_long_aligned > 1e-6)

        # c) 执行最终计算
        turnover_std_ratio = (std_short_aligned / std_long_safe) - 1

        # d) 再次清理可能产生的无穷大值
        final_factor = turnover_std_ratio.replace([np.inf, -np.inf], np.nan)
        return final_factor

    # === 流动性 (Liquidity) ===
    def _calculate_rolling_mean_turnover_rate(self, window: int, min_periods: int) -> pd.DataFrame:
        """【私有引擎】计算滚动平均换手率（以小数形式）。"""

        # 1. 获取原始换手率数据（其中停牌日为NaN） ---现在直接ffill（0） 符合金融要求
        turnover_df_ = self.factor_manager.get_raw_factor('turnover_rate')

        # 2. 在填充后的、代表了真实交易活动的数据上进行滚动计算
        mean_turnover_df = turnover_df_.rolling(window=window, min_periods=min_periods).mean()

        return mean_turnover_df

    # --- 现在，原来的两个函数可以简化为下面这样 ---

    def _calculate_turnover_rate_90d_mean(self) -> pd.DataFrame:
        """计算90日滚动平均换手率。"""
        logger.info("    > 正在计算因子: turnover_rate_90d_mean...")
        # 直接调用通用引擎
        return self._calculate_rolling_mean_turnover_rate(window=90, min_periods=60)

    def _calculate_turnover_rate_monthly_mean(self) -> pd.DataFrame:
        """计算月度（21日）滚动平均换手率。"""
        logger.info("    > 正在计算因子: turnover_rate_monthly_mean...")
        # 直接调用通用引擎
        return self._calculate_rolling_mean_turnover_rate(window=21, min_periods=15)


    def _calculate_ln_turnover_value_90d(self) -> pd.DataFrame:
        """
        计算90日日均成交额的对数。

        金融逻辑:
        日均成交额直接反映了资产的流动性容量。成交额越大的股票，能容纳的资金规模越大，
        交易时的冲击成本也越低。取对数是为了使数据分布更接近正态，便于进行回归分析。
        """
        logger.info("    > 正在计算因子: ln_turnover_value_90d...")
        # 1. 获取日成交额数据 (单位：元)
        amount_df = self.factor_manager.get_raw_factor('amount').copy()

        # 2. 计算90日滚动平均成交额
        mean_amount_df = amount_df.rolling(window=90, min_periods=60).mean()

        # 3. 【核心风控】在取对数前，必须确保数值为正。
        #    使用 .where() 方法，将所有小于等于0的值替换为NaN，避免log函数报错。
        mean_amount_positive = mean_amount_df.where(mean_amount_df > 0)

        # 4. 计算对数
        ln_turnover_value_df = np.log(mean_amount_positive)

        logger.info("    > ln_turnover_value_90d 计算完成。")
        return ln_turnover_value_df
    def _calculate_turnover_t1_div_t20d_avg(self) -> pd.DataFrame:
        """
        逻辑：成交额突然放大，往往伴随趋势/事件驱动。
        """
        # 1. 获取日成交额数据 (单位：元)
        amount_df = self.factor_manager.get_raw_factor('amount').copy()
        return amount_df / amount_df.rolling(20).mean()
    def _calculate_turnover_reversal_20d(self) -> pd.DataFrame:
        """
        高换手率本身与小市值股和特定行业高度相关，建议中性化。。
        """
        # 1. 获取日成交额数据 (单位：元)
        amount_df = self.factor_manager.get_raw_factor('amount').copy()
        return amount_df / amount_df.rolling(20).mean()


    def _calculate_amihud_liquidity(self) -> pd.DataFrame:
        """
       计算Amihud非流动性指标 - 最终生产版。

       处理流程:
       1. 计算原始日度Amihud指标。
       2. 对数变换(log1p)处理分布形状。
       3. 滚动平均以平滑信号。
       4. 截面标准化(Z-Score)处理数据尺度，使其对回归模型友好。
       """

        logger.info("    > 正在计算因子: amihud_liquidity (非流动性) -...")

        # 步骤 1: 计算原始日度Amihud
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy()
        amount_df = self.factor_manager.get_raw_factor('amount').copy()

        amount_in_yuan = amount_df
        amount_in_yuan_safe = amount_in_yuan.where(amount_in_yuan > 0)
        daily_amihud_df = pct_chg_df.abs() / amount_in_yuan_safe

        # 2. 对数变换
        # 使用 np.log1p()，它计算的是 log(1 + x)，可以完美处理x接近0的情况。
        # 这是处理这类因子的标准做法。
        log_amihud_df = np.log1p(daily_amihud_df)

        # 3. 【】: 滚动平滑
        # 使用过去一个月（约20个交易日）的平均值来代表当天的流动性水平
        # 这会使因子信号更稳定，减少日常噪声。
        smoothed_log_amihud_df = log_amihud_df.rolling(window=20, min_periods=12).mean()
        # 2. 乘以一个足够大的常数进行缩放
        #    例如，1e10可以将 e-11 级别的值放大到 0.1 级别
        scaling_factor = 1e10
        amihud_scaled_df = smoothed_log_amihud_df * scaling_factor
        return  amihud_scaled_df
    #类别： 资金流向类 money_flow
    def _calculate_large_trade_ratio_10d(self) -> pd.DataFrame:
        """
        【需特殊数据】计算10日大单或特大单净买入额占比。
        金融逻辑:
        大单或特大单的交易行为通常被认为是机构或主力资金的动向。持续的大单净流入
        可能预示着有备而来的建仓行为，是一个积极信号。大单或特大单净流入占比，反映了机构或主力资金的动向。持续的大单净流入可能预示着股价的上涨。
        """

        # 1. 加载资金流数据并计算每日大单净买入额
        moneyflow_df =None #load_moneyflow_df() #todo 下载  load
        # 注意：Tushare的moneyflow单位是“万元”，我们的amount因子单位是“元”，需要统一
        moneyflow_df['net_lg_amount'] = (moneyflow_df['buy_lg_amount'] + moneyflow_df['buy_elg_amount'] -
                                         moneyflow_df['sell_lg_amount'] - moneyflow_df['sell_elg_amount']) * 10000

        net_lg_amount_wide = moneyflow_df.pivot_table(
            index='trade_date', columns='ts_code', values='net_lg_amount'
        )

        # 2. 获取总成交额 (单位: 元)
        amount_df = self.factor_manager.get_raw_factor('amount')

        # 3. 计算10日滚动求和
        net_lg_sum_10d = net_lg_amount_wide.rolling(window=10, min_periods=7).sum()
        amount_sum_10d = amount_df.rolling(window=10, min_periods=7).sum()

        # 4. 风险控制与计算
        amount_sum_safe = amount_sum_10d.where(amount_sum_10d > 0)

        return (net_lg_sum_10d / amount_sum_safe).replace([np.inf, -np.inf], np.nan)
    ##财务basic数据

    def _calculate_cashflow_ttm(self) -> pd.DataFrame:
        """
           【】计算滚动12个月的经营活动现金流净额 (TTM)。
           输入:
           - cashflow_df: 原始现金流量表数据，包含['ann_date', 'ts_code', 'end_date', 'n_cashflow_act']
           - all_trading_dates: 一个包含所有交易日日期的pd.DatetimeIndex，用于构建最终的日度因子矩阵。
           输出:
           - 一个以交易日为索引(index)，股票代码为列(columns)的日度TTM因子矩阵。
           """
        return  self._calculate_financial_ttm_factor('cashflow_ttm',load_cashflow_df,'n_cashflow_act')

    def _calculate_net_profit_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月的归母净利润 (Net Profit TTM)。
        该函数逻辑与 _calculate_cashflow_ttm 完全一致，仅替换数据源和字段。
        """
        return  self._calculate_financial_ttm_factor('net_profit_ttm',load_income_df,'n_income_attr_p')

    def _calculate_total_equity(self) -> pd.DataFrame:
        """
        【生产级】获取每日可用的最新归母所有者权益。
        这是一个时点数据，无需计算TTM，但需要执行公告日对齐流程。
        """
        ret  = self._calculate_financial_snapshot_factor('total_equity',load_balancesheet_df,'total_hldr_eqy_exc_min_int')
        return ret

    def _calculate_total_revenue_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的营业总收入 (TTM)。
        利用通用TTM引擎计算得出。
        """
        # print("--- 调用通用引擎计算: revenue_ttm ---")
        return self._calculate_financial_ttm_factor(
            factor_name='total_revenue_ttm',
            data_loader_func=load_income_df,
            source_column='total_revenue'  # Tushare利润表中的“营业总收入”字段
        )

    def _calculate_op_cost_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的营业总成本 (TTM)。
        利用通用TTM引擎计算得出。
        """
        # print("--- 调用通用引擎计算: op_cost_ttm ---")
        return self._calculate_financial_ttm_factor(
            factor_name='op_cost_ttm',
            data_loader_func=load_income_df,
            source_column='oper_cost'  # Tushare利润表中的“减:营业成本”字段
        )

    def _calculate_total_debt(self) -> pd.DataFrame:
        """【生产级】获取每日可用的最新总负债。"""
        print("--- 调用Snapshot引擎计算: total_debt ---")
        return self._calculate_financial_snapshot_factor(
            factor_name='total_debt',
            data_loader_func=load_balancesheet_df,
            source_column='total_liab'  # Tushare资产负债表中的“负债合计”字段
        )
    #对齐方案：事件类型
    def _calculate_total_assets(self) -> pd.DataFrame:
        """
        【生产级】获取每日可用的最新总资产。
        这是一个“时点”或“存量”指标，直接从最新的资产负债表中获取。

        此方法将作为计算其他财务比率（如资产负债率）的分母。
        """
        logger.info("--- 调用Snapshot引擎计算: total_assets ---")

        # 我们将直接调用通用的“时点”因子计算引擎，
        # 只需要告诉它要加载哪个数据表、并使用其中的哪一列即可。
        return self._calculate_financial_snapshot_factor(
            factor_name='total_assets',
            data_loader_func=load_balancesheet_df,  # 指定加载资产负债表
            source_column='total_assets'              # 指定使用资产负债表中的“资产总计”字段
        )

    def _calculate_net_profit_single_q_long(self) -> pd.DataFrame:
        """
        【内部函数】计算单季度归母净利润的长表。
        这是计算同比增长率的基础。
        """
        # print("--- 调用通用引擎计算: net_profit_single_q ---")
        return self._calculate_financial_single_q_factor(
            factor_name='net_profit_single_q',
            data_loader_func=load_income_df,
            source_column='n_income_attr_p'  # 确认使用归母净利润
        )

        #  三、新增进阶因子 (Advanced Factors)
        # =========================================================================

        # === 质量类深化 (Advanced Quality) ===

    def _calculate_operating_accruals(self) -> pd.DataFrame:
        """
        计算经营性应计利润 (Operating Accruals)。
        公式: (净利润TTM - 经营活动现金流TTM) / 总资产
        这是一个反向指标，值越高，利润质量越差，未来反转（下跌）风险越高。
        """
        logger.info("    > 正在计算因子: operating_accruals (经营性应计利润)...")

        # 1. 获取所需的基础因子
        net_profit_ttm = self.factor_manager.get_raw_factor('net_profit_ttm')
        cashflow_ttm = self.factor_manager.get_raw_factor('cashflow_ttm')
        total_assets = self.factor_manager.get_raw_factor('total_assets')

        # 2. 对齐数据 (核心修正部分)
        # 使用 reindex 的方式对齐多个DataFrame，这是更稳健和清晰的做法

        # 2.1 找到所有数据源索引的交集，这等价于 'inner' join
        common_index = net_profit_ttm.index.intersection(cashflow_ttm.index)
        common_index = common_index.intersection(total_assets.index)

        # 2.2 使用共同索引来对齐所有DataFrame
        profit_aligned = net_profit_ttm.reindex(common_index)
        cash_aligned = cashflow_ttm.reindex(common_index)
        assets_aligned = total_assets.reindex(common_index)

        # 3. 风险控制：总资产必须为正，防止除以0或负数
        # 使用 .where 方法，不满足条件的项会被设置为NaN，这很安全
        assets_aligned_safe = assets_aligned.where(assets_aligned > 0)

        # 4. 计算应计利润
        # 对齐后，可以直接进行元素级(element-wise)计算
        accruals = (profit_aligned - cash_aligned) / assets_aligned_safe

        # 5. 清理计算过程中可能产生的无穷大值
        accruals = accruals.replace([np.inf, -np.inf], np.nan)

        return accruals

    def _calculate_earnings_stability(self) -> pd.DataFrame:
        """
        计算盈利稳定性 (Earnings Stability) - 修正版。
        使用变异系数的倒数，衡量盈利的相对稳定性，剔除规模效应。
        公式: abs(滚动平均净利润) / 滚动净利润标准差
        这是一个正向指标，值越高，盈利相对越稳定。
        """
        return self._create_general_quarterly_factor_engine(
            factor_name='earnings_stability',
            data_loader_func=load_income_df,
            source_column='n_income_attr_p',
            calculation_logic_func=self._earnings_stability_logic
        )

    #质量类因子（---杠杆因子

    def _calculate_debt_252d_ratio(self) -> pd.DataFrame:
        """
        计算财务杠杆变动率因子 (LVGI - Leverage Growth Index)。
        因子类别: 质量类因子
        金融逻辑:
        衡量公司资产负债率的年度变化趋势。一个增长的比率（>1）可能表示
        公司正在增加杠杆，财务风险上升；一个降低的比率（<1）则可能表示
        公司正在去杠杆，财务状况更稳健。
        计算公式:
        本期(年报)资产负债率 / 上期(年报)资产负债率
        """

        # --- 1. 获取基础原材料：每日更新的资产负债率因子 ---
        debt_to_assets_df = self.factor_manager.get_raw_factor('debt_to_assets').copy()

        # --- 2. 获取上一期（年报）的数据 ---
        debt_to_assets_last_year = debt_to_assets_df.shift(252)

        # --- 3. 风险控制与最终计算 ---
        # a) 对齐数据，确保计算的准确性
        current_leverage, prior_leverage = debt_to_assets_df.align(
            debt_to_assets_last_year, join='inner', axis=None
        )
        # b) 【核心风控】分母（上一期资产负债率）可能为0或极小，必须处理以防除0错误。
        # 我们将小于1e-6的值视同为0，并替换为NaN，避免产生无穷大(inf)的因子值。
        prior_leverage_safe = prior_leverage.where(prior_leverage > 1e-6)

        # c) 执行最终计算
        leverage_change_ratio = current_leverage / prior_leverage_safe
        # d) 再次清理可能产生的无穷大值
        final_factor = leverage_change_ratio.replace([np.inf, -np.inf], np.nan)
        return final_factor
    #量价因子
    def _calculate_single_day_vpt(self) -> pd.DataFrame:
        """
        【新增】计算单日价量趋势因子 (Volume-Price Trend)。
        金融逻辑:
        结合价格变动方向和成交量大小，衡量资金推动价格上涨或下跌的力度。
        正值代表放量上涨，负值代表放量下跌。
        计算公式:
        （今日收盘价 - 昨日收盘价）/ 昨日收盘价 * 当日成交量
        等价于： pct_chg * vol_hfq
        """
        logger.info("    > 正在计算因子: single_day_vpt...")

        # --- 1. 获取基础原材料 ---
        # a) 每日涨跌幅 (已经包含了价格变动信息)
        pct_chg = self.factor_manager.get_raw_factor('pct_chg').copy()

        # b) 后复权成交量 (消除送转股等事件对成交量的影响)
        vol_hfq = self.factor_manager.get_raw_factor('vol_hfq').copy()

        # --- 2. 对齐数据 ---
        # 使用 align 确保两个DataFrame在计算前具有完全相同的索引和列
        pct_chg_aligned, vol_hfq_aligned = pct_chg.align(vol_hfq, join='inner', axis=None)

        # --- 3. 核心计算 ---
        raw_vpt_factor = pct_chg_aligned * vol_hfq_aligned

        return raw_vpt_factor
    #复合类因子
    # def _calculate_mom5d_mom1d_vol_wei_3_3_4_combo(self) -> pd.DataFrame:
    #     """
    #     【新增】计算一个由短期动量和成交量加权合成的复合因子。
    #
    #     金融逻辑:
    #     综合考量价格的短期趋势强度（1日和5日动量）和市场的参与热度（成交量），
    #     旨在捕捉那些有资金参与的、正在启动的短期趋势。
    #
    #     计算公式:
    #     - 0.3 * Z-Score(5日收益率)
    #     - 0.3 * Z-Score(1日收益率)
    #     - 0.4 * Z-Score(后复权成交量)
    #     """
    #     """
    #      【V2 修正版】计算一个由短期动量和成交量加权合成的复合因子。
    #
    #      核心修正：
    #      使用 reindex 方法来稳健地对齐三个及以上的DataFrame。
    #      """
    #
    #     # --- 1. 获取所有基础原材料 ---
    #     close_hfq = self.factor_manager.get_raw_factor('close_hfq').copy()
    #     vol_hfq = self.factor_manager.get_raw_factor('vol_hfq').copy()
    #
    #     # --- 2. 计算三大核心组件 ---
    #     mom_5d = close_hfq.pct_change(periods=5)
    #     mom_1d = close_hfq.pct_change(periods=1)
    #     volume = vol_hfq
    #
    #     # --- 3. 对所有组件进行截面标准化 (Z-Score) ---
    #     z_mom_5d = cross_sectional_zscore(mom_5d)
    #     z_mom_1d = cross_sectional_zscore(mom_1d)
    #     z_volume = cross_sectional_zscore(volume)
    #
    #     # a) 找到所有因子共有的日期索引
    #     common_index = z_mom_5d.index.intersection(z_mom_1d.index).intersection(z_volume.index)
    #
    #     # b) 找到所有因子共有的股票代码列
    #     common_columns = z_mom_5d.columns.intersection(z_mom_1d.columns).intersection(z_volume.columns)
    #
    #     # c) 将所有DataFrame规整到这个共同的形状上
    #     z_mom_5d_aligned = z_mom_5d.reindex(index=common_index, columns=common_columns)
    #     z_mom_1d_aligned = z_mom_1d.reindex(index=common_index, columns=common_columns)
    #     z_volume_aligned = z_volume.reindex(index=common_index, columns=common_columns)
    #
    #     # --- 5. 按指定权重进行加权合成 ---
    #     weights = {'mom5d': 0.3, 'mom1d': 0.3, 'vol': 0.4}
    #     composite_factor = (
    #             weights['mom5d'] * z_mom_5d_aligned +
    #             weights['mom1d'] * z_mom_1d_aligned +
    #             weights['vol'] * z_volume_aligned
    #     )
    #
    #     return composite_factor

        # === 新增情绪类因子 (Sentiment) ===
    ##
    # 滚动技术指标类 (价格材料 必须喂给它是连续的
    # 因子示例: _calculate_rsi, _calculate_cci
    #
    # 计算特性: 它们的算法（尤其是在pandas_ta这样的库中）通常假定输入的时间序列是连续的。数据的中断（NaN）会导致指标计算中断，产生非常稀疏的因子值。
    #
    # 推荐使用: self.factor_manager.get_raw_factor(('close_hfq_filled', 10)) (经过填充的版本)
    #
    # 理由:
    #
    # 这是一个实用性和纯粹性之间的权衡。
    #
    # 为了得到一个更连续、在策略中更“可用”的因子信号，我们主动选择接受一个假设：“短期停牌（如10天内），股票的状态可以被认为是其停牌前的延续”。
    #
    # 我们用一个带limit的ffill来填充短期的NaN，以确保技术指标能够连续计算。这是一个主动的、有意识的建模选择。#
    def _calculate_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        计算RSI (相对强弱指数)。
        衡量股价的超买超卖状态，是经典的反转信号。
        """
        logger.info(f"    > 正在计算因子: RSI (window={window})...")
        close_df = self.factor_manager.get_raw_factor(('close_hfq_filled', 10))

        # 使用 pandas_ta 库，通过 .apply 在每一列（每只股票）上独立计算
        rsi_df = close_df.apply(lambda x: ta.rsi(x, length=window), axis=0)

        return rsi_df

    def _calculate_cci(self, window: int = 20) -> pd.DataFrame:
        """
        计算CCI (顺势指标)。
        衡量股价是否超出其正常波动范围，可用于捕捉趋势的开启或反转。
        """
        logger.info(f"    > 正在计算因子: CCI (window={window})...")
        high_df = self.factor_manager.get_raw_factor(('high_hfq_filled',10))
        low_df = self.factor_manager.get_raw_factor(('low_hfq_filled',10))
        close_df = self.factor_manager.get_raw_factor(('close_hfq_filled',10))
        # CCI需要三列数据，我们按股票逐一计算
        cci_results = {}
        for stock_code in close_df.columns:
            # 确保该股票在所有价格数据中都存在
            if stock_code in high_df.columns and stock_code in low_df.columns:
                cci_series = ta.cci(
                    high=high_df[stock_code],
                    low=low_df[stock_code],
                    close=close_df[stock_code],
                    length=window
                )
                cci_results[stock_code] = cci_series

        cci_df = pd.DataFrame(cci_results)
        return cci_df

    ###惊喜
    # (确保在文件顶部导入):

    ##
    # 核心逻辑：SUE衡量的是盈利的“惊喜”程度。
    #
    # 市场如何反映“惊喜”：一家公司发布了超预期的财报，市场最直接的反应是什么？股价会跳空高开，并在接下来几天持续上涨。这种现象被称为“盈余公告后漂移”(Post-Earnings Announcement Drift, PEAD)，是金融学里最著名、最稳健的异象之一。
    #
    # 计算财报发布日后几天的累计收益率(财报后漂移。
    # pead Post-Earnings Announcement Drift) #
    def _calculate_pead(self) -> pd.DataFrame:
        """
        【V3.0 - 引擎版】计算SUE因子 (标准化盈利意外)。
        公式: earnings_surprise_numerator / total_mv
        """
        logger.info("  > [引擎版] 正在计算 PEAD (SUE) 因子...")

        # 直接调用我们强大的金融比率引擎，一行代码完成所有对齐、风控和计算
        return self._create_financial_ratio_factor(
            numerator_factor='earnings_surprise_numerator',  # 使用我们刚刚创建的新因子
            denominator_factor='total_mv',
            numerator_must_be_positive=False  # 盈利意外可以是负数
        )
    ##
    # 核心逻辑：分析师评级调整反映的是“聪明钱”对公司基本面预期的持续改善。
    #
    # 市场如何反映“持续改善”：一家基本面持续向好的公司，它的股价走势通常不是暴涨暴跌，而是稳步、持续地上涨。这种上涨通常伴随着较低的波动。这被称为“高质量的动量”。
    #
    # 风险调整后动量 /计算风险调整后的动量。#
    def _calculate_quality_momentum(self) -> pd.DataFrame:
        """
        计算风险调整后的动量因子 (Quality Momentum)。
        逻辑: 120日动量 / 90日波动率。
        作为“分析师评级上调”的代理，寻找那些稳步上涨的股票。
        """
        logger.info("    > 正在计算代理因子: Quality Momentum...")

        # 1. 获取动量和波动率因子
        momentum_120d = self.factor_manager.get_raw_factor('momentum_120d')
        volatility_90d = self.factor_manager.get_raw_factor('volatility_90d')

        # 2. 对齐数据
        mom_aligned, vol_aligned = momentum_120d.align(volatility_90d, join='inner', axis=None)

        # 3. 风险控制：波动率必须为正
        vol_aligned_safe = vol_aligned.where(vol_aligned > 0)

        # 4. 计算风险调整后的动量
        quality_momentum_df = mom_aligned / vol_aligned_safe

        quality_momentum_df = quality_momentum_df.replace([np.inf, -np.inf], np.nan)
        return quality_momentum_df
    ######################

    ##辅助函数
    #ok
    def _calculate_market_pct_chg(self, index_code) -> pd.Series:
        """【新增】根据指定的指数代码，计算其总回报收益率。"""
        """
           【V2.0 - 权威版】
           根据指数的不复权点位和分红数据，计算真实的总回报收益率。
           确保与个股pct_chg的计算逻辑完全统一。
           """
        logger.info(f"  > 正在基于第一性原理，计算市场基准 [{index_code}] 的权威pct_chg...")

        # --- 1. 获取最基础的原材料 ---
        # a) 获取指数的不复权日线数据 (需要你的DataManager支持)

        # b) 获取指数的分红事件 (需要你的DataManager支持)
        #    对于宽基指数，Tushare通常在 index_daily 接口中直接提供总回报的pct_chg
        #    但最严谨的做法，是获取其对应的ETF的分红数据，或使用总回报指数
        #    这里我们做一个简化，直接使用Tushare index_daily 中那个质量较高的pct_chg
        #    这是一种在严谨性和工程便利性上的权衡。

        index_daily_total_return = load_index_daily(index_code)
        market_pct_chg = index_daily_total_return['pct_chg'] / 100.0

        # 确保返回的Series有名字，便于后续join
        market_pct_chg.name = index_code

        return market_pct_chg
    #分类 行业~~~
    def _calculate_sw_l1_momentum_21d(self, pointInTimeIndustryMap: PointInTimeIndustryMap):
        fa = IndustryMomentumFactor(pointInTimeIndustryMap)
        ret = fa.compute(self.factor_manager.data_manager._prebuffer_trading_dates, 'l1_code')
        return ret



    #伞兵函数 共一个将使用 没啥复用意义，只是清晰而已，



    def _calculate_earnings_surprise_numerator(self) -> pd.DataFrame:
        """
        【V3.0 - 核心组件】计算盈利意外的分子 (TTM归母净利同比变动额)。
        公式: net_profit_ttm(t) - net_profit_ttm(t-1_year)
        这是一个事件驱动的日度因子。
        """
        logger.info("  > [核心组件] 正在计算: earnings_surprise_numerator...")

        # 1. 获取TTM归母净利润因子。
        #    这个因子已经是通过我们修正后的引擎计算出来的，它已经是每日频率、且无前视偏差的了。
        net_profit_ttm_df = self.factor_manager.get_raw_factor('net_profit_ttm')

        # 2. 获取大约一年前的TTM归母净利润。
        #    在日度数据上，shift(252) 是对一年前最常用的近似。
        net_profit_ttm_last_year_df = net_profit_ttm_df.shift(252)

        # 3. 计算两者差额，即为盈利意外的绝对值
        surprise_numerator_df = net_profit_ttm_df - net_profit_ttm_last_year_df

        return surprise_numerator_df
    # --- 为引擎创建可复用的“计算逻辑”函数 ---
    def _yoy_growth_logic(self, df: pd.DataFrame, col_name: str, factor_name: str) -> pd.DataFrame:
        df_sorted = df.sort_values(by=['ts_code', 'end_date'])
        last_year_q = df_sorted.groupby('ts_code')[col_name].shift(4)
        last_year_q_safe = last_year_q.where(last_year_q > 0)
        df[factor_name] = df_sorted[col_name] / last_year_q_safe - 1
        return df

    def _earnings_stability_logic(self, df: pd.DataFrame, col_name: str, factor_name: str) -> pd.DataFrame:

        # 【核心修正】必须先按股票和时间排序，确保rolling操作的时序正确性
        df_sorted = df.sort_values(by=['ts_code', 'end_date'])

        grouped = df_sorted.groupby('ts_code')[col_name]
        rolling_stats = grouped.rolling(window=20, min_periods=12)

        mean_val = rolling_stats.mean().reset_index(level=0, drop=True)
        std_val = rolling_stats.std().reset_index(level=0, drop=True)

        # 后续逻辑保持不变...
        std_safe = std_val.clip(lower=1e-6)
        mean_safe = mean_val.where(abs(mean_val) > 1000, 0)

        # 注意：因为df_sorted和df的索引是一样的，这里可以直接赋值
        df[factor_name] = abs(mean_safe) / std_safe
        return df
    ##以下是模板
    def _calculate_vwap_hfq(self) -> pd.DataFrame:
        """
        【V2.0 - 基于第一性原理】计算每日的后复权VWAP (vwap_hfq)。

        金融逻辑:
        VWAP (成交量加权平均价) 是当天交易的平均成本。为与后复权收盘价(close_hfq)
        进行可比的计算，必须将当日的原始VWAP通过后复权因子进行调整。

        计算步骤:
        1. 从 'daily' 接口获取原始成交额(amount)和成交量(vol)。
        2. 计算当日不复权VWAP: vwap_raw = (amount * 1000) / (vol * 100)。
        3. 获取后复权因子: hfq_adj_factor。
        4. 计算后复权VWAP: vwap_hfq = vwap_raw * hfq_adj_factor。
        """
        logger.info("    > [第一性原理] 正在计算因子: vwap_hfq...")

        # 1. 获取最基础的日线数据
        # 假设 FactorManager 可以直接获取 amount 和 vol
        amount_df = self.factor_manager.get_raw_factor('amount')
        vol_df = self.factor_manager.get_raw_factor('vol_hfq')

        # 2. 对齐数据
        amount_aligned, vol_aligned = amount_df.align(vol_df, join='inner', axis=None)

        # 3. 风险控制：成交量必须大于0
        vol_aligned_safe = vol_aligned.where(vol_aligned > 0)

        # 4. 计算当日不复权VWAP
        vwap_raw = (amount_aligned) / (vol_aligned_safe )

        # 5. 获取后复权因子并对齐
        hfq_adj_factor = self.factor_manager.get_raw_factor('hfq_adj_factor')
        vwap_raw_aligned, adj_factor_aligned = vwap_raw.align(hfq_adj_factor, join='inner', axis=None)

        # 6. 计算后复权VWAP
        vwap_hfq = vwap_raw_aligned * adj_factor_aligned

        return vwap_hfq.replace([np.inf, -np.inf], np.nan)
    def _create_scaffold_and_merge_quarterly_data(self,
                                                  long_df: pd.DataFrame,
                                                  date_col: str = 'end_date') -> pd.DataFrame:
        """
        【底层核心工具】为季度长表数据创建时间脚手架并合并。

        功能: 解决因财报发布不规律导致的季度时间序列“跳跃”问题。

        Args:
            long_df (pd.DataFrame): 包含 'ts_code' 和日期列的原始长表数据。
            date_col (str): 用于创建时间范围的日期列名，通常是 'end_date'。

        Returns:
            pd.DataFrame: 一个将原始数据合并到完整季度时间线上的新DataFrame。
                          缺失的季度会以行的形式存在，但数据列为NaN。
        """
        if long_df.empty:
            return long_df

        # 1. 为每只股票创建一个完整的、连续的季度日期范围
        scaffold_df = long_df.groupby('ts_code')[date_col].agg(['min', 'max'])
        full_date_dfs = []
        for ts_code, row in scaffold_df.iterrows():
            date_range = pd.date_range(start=row['min'], end=row['max'], freq='Q-DEC')
            full_date_dfs.append(pd.DataFrame({'ts_code': ts_code, date_col: date_range}))

        if not full_date_dfs:
            return long_df  # 如果没有有效日期范围，返回原始df

        full_dates_scaffold = pd.concat(full_date_dfs)

        # 2. 将原始数据左合并到脚手架上，让缺失的季度显式化为NaN
        merged_df = pd.merge(full_dates_scaffold, long_df, on=['ts_code', date_col], how='left')

        return merged_df
    def _create_rolling_volatility_factor(self, window: int, min_periods: int) -> pd.DataFrame:
        """【V3.0 通用滚动波动率引擎】计算指定窗口的年化波动率。"""
        logger.info(f"    > [波动率引擎] 正在计算 {window}日 年化波动率...")
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy(deep=True)
        rolling_std_df = pct_chg_df.rolling(window=window, min_periods=min_periods).std()
        annualized_vol_df = rolling_std_df * np.sqrt(252)
        return annualized_vol_df
    def _create_general_quarterly_factor_engine(self,
                                                factor_name: str,
                                                data_loader_func: Callable[[], pd.DataFrame],
                                                source_column: str,
                                                calculation_logic_func: Callable[[pd.DataFrame, str, str], pd.DataFrame]
                                                ) -> pd.DataFrame:
        """
        【通用季度因子引擎】根据指定的单季度财务数据和计算逻辑，延展为每日因子。
        """
        logger.info(f"--- [通用季度引擎] 正在为因子 '{factor_name}' 执行计算 ---")

        # 步骤一：获取基础的单季度数据长表
        single_q_col = f"{source_column}_single_q"
        long_df = self._get_single_q_long_df(
            data_loader_func=data_loader_func,
            source_column=source_column,
            single_q_col_name=single_q_col
        )

        # 步骤二：应用传入的、自定义的计算逻辑
        long_df_calculated =  calculation_logic_func(long_df, single_q_col, factor_name)

        # 步骤三：格式化为日度因子矩阵
        final_long_df = long_df_calculated[['ts_code', 'ann_date', 'end_date', factor_name]].dropna()
        if final_long_df.empty:
            raise ValueError(f"警告: 因子 '{factor_name}' 的计算逻辑没有产生任何有效数据点。")

        final_long_df['ann_date'] = pd.to_datetime(final_long_df['ann_date'])
        final_long_df['trade_date'] = map_ann_dates_to_tradable_dates(
            ann_dates=final_long_df['ann_date'],
            trading_dates=self.factor_manager.data_manager._prebuffer_trading_dates
        )
        final_long_df = final_long_df.sort_values(by=['ts_code', 'end_date'])
        final_wide = final_long_df.pivot_table(
            index='trade_date',  # 已修正
            columns='ts_code',
            values=factor_name,
            aggfunc='last'
        )

        daily_factor_df = _broadcast_ann_date_to_daily(final_wide, self.factor_manager.data_manager._prebuffer_trading_dates)

        logger.info(f"--- [通用季度引擎] 因子 '{factor_name}' 计算完成 ---")
        return daily_factor_df
    #ok
    def _create_financial_ratio_factor(self,
                                       numerator_factor: str,
                                       denominator_factor: str,
                                       numerator_must_be_positive: bool = False) -> pd.DataFrame:
        """
        【V3.0 通用金融比率引擎】根据指定的分子和分母因子，计算比率类因子。

        Args:
            numerator_factor (str): 分子因子的名称。
            denominator_factor (str): 分母因子的名称。
            numerator_must_be_positive (bool): 是否要求分子也必须为正数。

        Returns:
            pd.DataFrame: 计算出的比率因子。
        """
        logger.info(f"  > [比率引擎] 正在计算: {numerator_factor} / {denominator_factor}...")

        # 1. 获取分子和分母 (它们已经是每日对齐的、无前视偏差的因子)
        numerator_df = self.factor_manager.get_raw_factor(numerator_factor).copy(deep=True)
        denominator_df = self.factor_manager.get_raw_factor(denominator_factor).copy(deep=True)

        # 2. 对齐数据 (使用 inner join 保证数据质量)
        num_aligned, den_aligned = numerator_df.align(denominator_df, join='inner', axis=None)

        # 3. 防止除0
        den_positive = den_aligned.where(den_aligned > 0)

        num_final = num_aligned
        if numerator_must_be_positive:
            num_final = num_aligned.where(num_aligned > 0)

        # 4. 计算并返回
        ratio_df = num_final / den_positive
        return ratio_df.replace([np.inf, -np.inf], np.nan)

    # --- 私有的、可复用的计算引擎 ---
    def _calculate_financial_ttm_growth_factor(self,
                                               factor_name: str,
                                               ttm_factor_name: str,
                                               lookback_days: int = 252) -> pd.DataFrame:
        """
        【通用TTM增长率计算引擎】
        根据指定的TTM因子，计算其同比增长率。
        公式: (Current TTM / Last Year's TTM) - 1

        Args:
            factor_name (str): 最终生成的因子名称（用于日志记录）。
            ttm_factor_name (str): 依赖的TTM因子的名称。
            lookback_days (int): 回溯周期，默认为252个交易日（约一年）。

        Returns:
            pd.DataFrame: 计算出的TTM同比增长率因子矩阵。
        """
        logger.info(f"      > [引擎] 正在为 {ttm_factor_name} 计算TTM同比增长率...")

        # --- 步骤一：获取当期的TTM因子数据 ---
        ttm_df = self.factor_manager.get_raw_factor(ttm_factor_name)

        # --- 步骤二：获取一年前（回溯期）的TTM因子数据 ---
        ttm_last_year = ttm_df.shift(lookback_days)

        # --- 步骤三：【核心风险控制】---
        # 金融逻辑：增长率的计算只有在分子(当期)和分母(去年同期)都为正时才有意义。
        # 这确保我们只比较“从盈利到盈利”的情况，避免了由盈转亏等情况带来的噪音。
        logger.info("        > 正在进行风险控制，确保分子和分母均为正数...")
        ttm_df_safe = ttm_df.where(ttm_df > 0)
        ttm_last_year_safe = ttm_last_year.where(ttm_last_year > 0)

        # --- 步骤四：计算同比增长率 ---
        growth_df = (ttm_df_safe / ttm_last_year_safe) - 1

        # --- 步骤五：后处理 ---
        # 防御性编程：清除计算过程中可能意外产生的无穷大值。
        # 采用重新赋值，避免使用 inplace=True。
        growth_df = growth_df.replace([np.inf, -np.inf], np.nan)

        logger.info(f"--- [引擎] 因子: {factor_name} 计算完成 ---")
        return growth_df

        ###A股市场早期，或一些公司在特定时期，只会披露年报和半年报，而缺少一季报和三季报的累计值。这会导致在我们的完美季度时间标尺上出现NaN。
        ### 所以这就是解决方案：实现了填充 跳跃的季度区间，新增填充的列：filled_col ，计算就在filled_col上面做diff。然后在平滑diff上做rolling。done
        ## 季度性数据ttm通用计算， 模板计算函数 ok (且是安全ffill ann_Date
    def _calculate_financial_ttm_factor(self,
                                        factor_name: str,
                                        data_loader_func: Callable[[], pd.DataFrame],
                                        source_column: str) -> pd.DataFrame:
        """
        【通用生TTM因子计算引擎】(已重构)
        计算滚动12个月(TTM)的因子值。
        """
        print(f"--- [引擎] 开始计算TTM因子: {factor_name} ---")

        # --- 步骤一： 获取单季度数据 ---
        single_q_col_name = f"{source_column}_single_q"
        single_q_long_df = self._get_single_q_long_df(
            data_loader_func=data_loader_func,
            source_column=source_column,
            single_q_col_name=single_q_col_name
        )

        # --- 步骤二：在单季度数据的基础上，计算TTM ---
        single_q_long_df[factor_name] = single_q_long_df.groupby('ts_code')[single_q_col_name].rolling(
            window=4, min_periods=4
        ).sum().reset_index(level=0, drop=True)

        # --- 步骤三：格式化为日度因子矩阵 (Pivot -> Reindex -> ffill) ---
        ttm_long_df = single_q_long_df[['ts_code', 'ann_date', 'end_date', factor_name]].dropna()#factor_name是rooling 计算出来的，因为min_periods 所以有三行nan ，在这里会被移除行
        if ttm_long_df.empty:
            raise ValueError(f"警告: 计算因子 {factor_name} 后没有产生任何有效的TTM数据点。")

        ttm_long_df = ttm_long_df.sort_values(by=['ts_code', 'end_date'])
        ttm_long_df['ann_date'] = pd.to_datetime(ttm_long_df['ann_date'])
        ttm_long_df['trade_date'] = map_ann_dates_to_tradable_dates(
            ann_dates=ttm_long_df['ann_date'],
            trading_dates = self.factor_manager.data_manager._prebuffer_trading_dates
        )

        ttm_wide = ttm_long_df.pivot_table(
            index='trade_date', #以ann_date 作为索引，这是无规则的index。假设100只股票，可能同一天有发布报告的股票只有一只
            columns='ts_code',
            values=factor_name,
            aggfunc='last'
        )#执行完之后的ttm_Wide 可能到处都是nan，原因：（以index='trade_date' （ann_date） 作为索引，这是无规则的index。假设100只股票，可能同一天有发布报告的股票只有一只）
        # ttm_daily = (ttm_wide.reindex(self.factor_manager.data_manager._prebuffer_trading_dates) #注意 满目苍翼的ttm_wide然后还被对齐索引（截断，）从trading开始日开始截，万一刚好交易日这一天 数值为nan，那么后面ffill也是nan，直到下一个有效ann_date
        #              .ffill()) #强化理解。ann_date [0701,0801],但传入的tradingList是0715，那么 reindex之后，就是nan ffill这个nan，跟0701ann的值矛盾！
        #解决办法如下：这是解决所有财报类因子“期初NaN”问题的最终解决方案。
        ret = _broadcast_ann_date_to_daily(ttm_wide, self.factor_manager.data_manager._prebuffer_trading_dates)

        print(f"--- [引擎] 因子: {factor_name} 计算完成 ---")

        return ret

    # 加载财报中的“时点”数据，并将其正确地映射到每日的时间序列上。 ok
    #流程:两个报告期间作了填充
    #对齐方案：事件类型
    def _calculate_financial_snapshot_factor(self,
                                             factor_name: str,
                                             data_loader_func: Callable[[], pd.DataFrame],
                                             source_column: str) -> pd.DataFrame:
        """
        【通用生产级“时点”因子计算引擎】
        根据指定的财务报表数据和字段，获取最新的“时点”因子值。
        适用于资产、负债、股东权益等“存量”指标。

        参数:
        - factor_name (str): 你想生成的最终因子名称，如 'total_assets'。
        - data_loader_func (Callable): 一个无参数的函数，用于加载原始财务数据DataFrame。
                                      例如: self.data_manager.load_balancesheet_df
        - source_column (str): 原始财务数据中的时点字段名。
                               例如: 'total_assets'

        返回:
        - 一个以交易日为索引(index)，股票代码为列(columns)的日度时点因子矩阵。
        """
        print(f"--- [通用引擎] 开始计算Snapshot因子: {factor_name} ---")

        # 使用传入的函数加载数据
        financial_df = data_loader_func()

        # 步骤一：选择数据并确保有效性
        snapshot_long_df = financial_df[['ts_code', 'ann_date', 'end_date', source_column]].copy(deep=True)
        snapshot_long_df=snapshot_long_df.dropna(inplace=False)
        if snapshot_long_df.empty:
            raise ValueError(f"警告: 计算因子 {factor_name} 时，从 {source_column} 字段未获取到有效数据。")
            # 步骤二：【核心修正】应用“公告日转交易日”模块
        snapshot_long_df['ann_date'] = pd.to_datetime(snapshot_long_df['ann_date'])
        snapshot_long_df['trade_date'] = map_ann_dates_to_tradable_dates(
            ann_dates=snapshot_long_df['ann_date'],
            trading_dates=self.factor_manager.data_manager._prebuffer_trading_dates
        )

        # 步骤二：透视
        snapshot_long_df=snapshot_long_df.sort_values(by=['ts_code', 'end_date'], inplace=False).copy(deep=True)
        snapshot_wide = snapshot_long_df.pivot_table(
            index='trade_date',
            columns='ts_code',
            values=source_column,
            aggfunc='last'
        )
        # --- 步骤三：【核心修正】使用合并索引的方法，进行稳健的重索引和填充 ---
        # 1. 获取交易日历
        trading_dates = self.factor_manager.data_manager._prebuffer_trading_dates

        # 2. 将稀疏的“公告日”索引与密集的“交易日”索引合并，并排序
        combined_index = snapshot_wide.index.union(trading_dates)

        # 3. 将 snapshot_wide 扩展到这个超级索引上，然后进行前向填充
        snapshot_filled_on_super_index = snapshot_wide.reindex(combined_index).ffill()

        # 4. 最后，从这个填充好的、完整的DataFrame中，只选取我们需要的交易日
        snapshot_daily = snapshot_filled_on_super_index.loc[trading_dates]

        print(f"--- [通用引擎] 因子: {factor_name} 计算完成 ---")
        return snapshot_daily

    def _calculate_financial_single_q_factor(self,
                                             factor_name: str,
                                             data_loader_func: Callable[[], pd.DataFrame],
                                             source_column: str) -> pd.DataFrame:
        """
        【通用生产级“单季度”因子计算引擎】(已重构)
        获取单季度的因子值的长表DataFrame。
        """
        print(f"--- [引擎] 开始准备单季度长表: {factor_name} ---")

        # 直接调用底层零件函数，获取单季度长表
        single_q_long_df = self._get_single_q_long_df(
            data_loader_func=data_loader_func,
            source_column=source_column,
            single_q_col_name=factor_name  # 输出列名就是我们想要的因子名
        )

        return single_q_long_df

    def _get_single_q_long_df(self,
                              data_loader_func: Callable[[], pd.DataFrame],
                              source_column: str,
                              single_q_col_name: str) -> pd.DataFrame:
        """
                【底层零件 - V2.0 重构版】从累计值财报数据中，计算出单季度值的长表。
                """
        print(f"    >  正在从 {source_column} 计算 {single_q_col_name}...")

        financial_long_df = data_loader_func()

        # 步骤一：【调用新工具】创建脚手架并合并，解决季度跳跃问题
        merged_df = self._create_scaffold_and_merge_quarterly_data(financial_long_df, 'end_date')

        # 步骤二：在时间连续的DataFrame上安全地计算单季度值
        filled_col = f"{source_column}_filled"
        # 使用ffill填充缺失财报季度的累计值
        merged_df[filled_col] = merged_df.groupby('ts_code')[source_column].ffill()
        merged_df[single_q_col_name] = merged_df.groupby('ts_code')[filled_col].diff()

        # 修正第一个季度被diff成NaN的问题
        is_q1 = merged_df['end_date'].dt.month == 3
        # 只在原始数据真实存在的地方进行修正
        merged_df.loc[is_q1 & merged_df[source_column].notna(), single_q_col_name] = merged_df.loc[is_q1, source_column]

        # 步骤三：整理并返回
        single_q_long_df = merged_df[['ts_code', 'ann_date', 'end_date', single_q_col_name]].copy()
        # 确保公告日和计算值都存在 (ann_date在缺失的季度行为NaN)
        single_q_long_df.dropna(subset=[single_q_col_name, 'ann_date'], inplace=True)

        return single_q_long_df
    #ok 能对上 聚宽数据
    def _calculate_pct_chg(self) -> pd.DataFrame:
        close_hfq = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        ret  = close_hfq.pct_change()
        return  ret


    #daily_hfq亲测 后复权的close是可用的，因为涨跌幅跟聚宽一模一样！ 我们直接用，不需要下面这样复杂的计算！
    # #ok 对的上daily的pct_chg字段（ pct_chg, float, 涨跌幅【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收
    # #也能和 t_bao_pct_chg 计算出来的数据对上！
    # def _calculate_pct_chg(self) -> pd.DataFrame:
    #     """
    #        根据“总回报恒等式”，直接从不复权价和分红送股事件计算真实总回报率。
    #        """
    #     logger.info("  > 正在基于第一性原理，计算【最终版】权威 pct_chg...")
    # 
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')
    #     pre_close_raw = close_raw.shift(1)
    #     dividend_events = load_dividend_events_long()
    # 
    #     # 【调试输出】
    #     logger.info(f"  > close_raw形状: {close_raw.shape}")
    # 
    #     # 构建分红矩阵（未对齐）
    #     cash_div_matrix_raw = dividend_events.pivot_table(index='ex_date', columns='ts_code',
    #                                                       values='cash_div_tax').reindex(close_raw.index).fillna(0)
    #     stk_div_matrix_raw = dividend_events.pivot_table(index='ex_date', columns='ts_code', values='stk_div').reindex(
    #         close_raw.index).fillna(0)
    # 
    #     logger.info(f"  > 分红矩阵原始形状: cash_div={cash_div_matrix_raw.shape}, stk_div={stk_div_matrix_raw.shape}")
    # 
    #     # 【关键修复】强制对齐到close_raw的列，避免形状不匹配
    #     target_stocks = close_raw.columns
    #     cash_div_matrix = cash_div_matrix_raw.reindex(columns=target_stocks, fill_value=0)
    #     stk_div_matrix = stk_div_matrix_raw.reindex(columns=target_stocks, fill_value=0)
    # 
    #     logger.info(f"  > 对齐后形状: cash_div={cash_div_matrix.shape}, stk_div={stk_div_matrix.shape}")
    # 
    #     # 验证形状一致性
    #     assert close_raw.shape == cash_div_matrix.shape == stk_div_matrix.shape, \
    #         f"形状不匹配: close_raw={close_raw.shape}, cash_div={cash_div_matrix.shape}, stk_div={stk_div_matrix.shape}"
    # 
    #     # 核心公式: (今日收盘价 * (1 + 送股比例) + 每股派息) / 昨日收盘价 - 1
    #     numerator = close_raw * (1 + stk_div_matrix) + cash_div_matrix
    #     true_pct_chg = numerator / pre_close_raw - 1
    # 
    #     logger.info(f"  > 最终结果形状: {true_pct_chg.shape}")
    # 
    #     final_pct_chg = true_pct_chg.where(close_raw.notna())
    #     return final_pct_chg

    # # 涨跌幅能对的上
    # def _calculate_close_hfq(self) -> pd.DataFrame:
    #     """
    #     【return 后复权 close】
    #     使用真实的“总回报率”和“不复权收盘价”来计算后复权价格序列。
    #     """
    #     # 1. 获取最关键的两个输入数据
    #     true_pct_chg = self.factor_manager.get_raw_factor('pct_chg')  # 我们之前计算的真实总回报率 (涨跌幅)
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')  # 当天真实价格 (不复权)
    #
    #     # 2. 处理边界情况：如果输入为空，则返回空DataFrame
    #     if close_raw.empty:
    #         raise  ValueError('价格data为空')
    #
    #     # 3. 计算每日的增长因子 (1 + 收益率)
    #     # 第一天的pct_chg是NaN，因为没有前一日的数据
    #     growth_factor = 1 + true_pct_chg
    #
    #     # 4. 使用.cumprod()计算自第一天以来的累积收益因子
    #     # cumprod() 会自动忽略开头的NaN值，从第一个有效数字开始累乘
    #     cumulative_growth_factor = growth_factor.cumprod()
    #
    #     # 5. 获取计算的基准价格 (即第一天的真实收盘价)
    #     base_price = close_raw.iloc[0]
    #
    #     # 6. 后复权价 = 基准价格 * 累积收益因子
    #     close_hfq = base_price * cumulative_growth_factor
    #
    #     # 7. 【关键修正】第一天的累积收益因子是NaN，导致第一天的后复权价也是NaN。
    #     # 我们必须将其修正为基准价格本身。
    #     close_hfq.iloc[0] = base_price
    #
    #     return close_hfq

    # def _calculate_close_hfq(self) -> pd.DataFrame:
    #     """
    #     【return 后复权 close
    #     """
    #     true_pct_chg = self.factor_manager.get_raw_factor('pct_chg')#我们刚才讨论的总回报率 涨跌幅
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')#当天真实价格

    def _calculate_hfq_adj_factor(self) -> pd.DataFrame:
        close_raw  = self.factor_manager.get_raw_factor('close_raw').copy(deep=True)
        close_hfq  = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        ret = close_hfq/close_raw
        return ret
    def _calculate_vol_hfq(self) -> pd.DataFrame:
        ##
        # 复权成交量 = 原始成交量 / 复权因子
        #
        # # 其中复权因子 = 复权价格 / 原始价格
        # 复权因子 = close_adj / close_raw
        # vol_adj = vol_raw / 复权因子#
        """【V3.0 - 统一版】根据通用复权乘数计算【反向】复权成交量"""
        vol_raw = self.factor_manager.get_raw_factor('vol_raw')
        hfq_adj_factor = self.factor_manager.get_raw_factor('hfq_adj_factor')
         ##
        # 核心原则： 我们到底为什么要对成交量进行复权？
        # 答案是：为了消除因“送转股”或“拆股”等股本变化事件，导致的成交量剧烈跳变，从而让历史上的成交量和现在的成交量具有可比性。#
        ret =  vol_raw/hfq_adj_factor
        return ret

    #ok
    # def _calculatesss_adj_factor(self) -> pd.DataFrame:
    #     """1
    #     【V2.0 - 最终生产版】
    #     根据不复权价格和分红送股事件，从头开始计算一个严格“随时点”的
    #     累积复权因子。此版本已修复停牌对前收盘价获取的bug。
    #     """
    #     logger.info("  > 正在基于第一性原理，计算权威的 adj_factor (V2.0)...")
    #
    #     # --- 1. 获取原材料 ---
    #     # 最后一个有效价格填补停牌期间的NaN ，明确需要 必须无脑ffill，那怕是十年前的收盘价格！
    #
    #     close_raw_filled = self.factor_manager.get_raw_factor('close_raw_ffill')
    #     # 然后，在这个“无空洞”的价格序列上，获取前一天的价格
    #     pre_close_raw_robust = close_raw_filled.shift(1)
    #     dividend_events = load_dividend_events_long()
    #
    #     # --- 2. 计算【每日】的调整比例 ---
    #     daily_adj_ratio = pd.DataFrame(1.0, index=close_raw_filled.index, columns=close_raw_filled.columns)
    #
    #     for _, event in dividend_events.iterrows():
    #         event_date, evet_stock_code = event['ex_date'], event['ts_code']
    #         if event_date in daily_adj_ratio.index and evet_stock_code in daily_adj_ratio.columns:
    #             cash_div = event.get('cash_div_tax', 0)
    #             stk_div = event.get('stk_div', 0)
    #
    #             # 【修正】使用我们新计算的、更稳健的前收盘价
    #             prev_close = pre_close_raw_robust.at[event_date, evet_stock_code]
    #
    #             if pd.isna(prev_close) or prev_close <= 0 or (cash_div == 0 and stk_div == 0):
    #                 continue
    #
    #             numerator = prev_close - cash_div
    #             denominator = prev_close * (1 + stk_div)
    #
    #             if denominator > 0:
    #                 daily_adj_ratio.at[event_date, evet_stock_code] = numerator / denominator
    #
    #     # --- 3. 计算【累积】复权因子 ---
    #     daily_adj_ratio.replace(0, np.nan, inplace=True) # 因为 numerator = prev_close - cash_div极端情况会==0.这里对修复
    #     daily_adj_ratio.fillna(1.0, inplace=True) #对上面那行产生的nan 填充成1 ，1是干净的，对后续计算没有影响的 （因为我们下面是是累计乘法！
    #     adj_factor_df = daily_adj_ratio.cumprod(axis=0)
    #
    #     logger.info("  > ✓ 权威的 adj_factor (V2.0) 计算完成。")
    #     return adj_factor_df


    #填充好 ，供于重复使用！ （目前场景 计算cci 要求必须是连续的价格数据！且是后复权


    def _calculate_close_hfq_filled(self,limit: int) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_open_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('open_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_high_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('high_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_low_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('low_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    ##基础换算！
    def _calculate_circ_mv(self):
        circ_mv = self.factor_manager.data_manager.raw_dfs['circ_mv'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        circ_mv = circ_mv * 10000
        return circ_mv
    def _calculate_total_mv(self):
        total_mv = self.factor_manager.data_manager.raw_dfs['total_mv'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        total_mv = total_mv * 10000
        return total_mv
    def _calculate_amount(self):
        amount = self.factor_manager.data_manager.raw_dfs['amount'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        amount = amount * 1000
        return amount
    def _calculate_turnover_rate(self):
        turnover_rate = self.factor_manager.data_manager.raw_dfs['turnover_rate'].copy(deep=True)
        turnover_rate = turnover_rate / 100
        return turnover_rate
    ###标准内部件

    def _calculate_vol_raw(self):
        return self.factor_manager.data_manager.raw_dfs['vol_raw'].copy(deep=True) * 100 # 成交量 vol 的单位是 手 (1手 = 100股)，需要乘以 100 换算成 股。

    ##
    #  目前用于 计算adj_factor 必须是ffill#
    def _calculate_close_raw_ffill(self):
        ret = self.factor_manager.get_raw_factor('close_raw').copy(deep=True)
        return ret.ffill()

    ##
    # 估值与市值类 (每日更新)
    # 字段: ps_ttm, total_mv, circ_mv, pb, pe_ttm
    #
    # 金融含义: NaN代表停牌，或指标本身无效（如PE为负）。这些是**“状态类”**数据。
    #
    # 下游需求: 因子计算和中性化时，我们希望尽可能有多的有效数据点。
    #
    # 填充策略: FILL_STRATEGY_FFILL_LIMIT_65 (有限前向填充)。
    #
    # 理由:
    #
    # 经济学假设：一个公司在停牌期间，其估值水平和市值，最合理的估计就是它停牌前的状态。ffill完美地符合这个假设。
    #
    # 风险控制：我们不希望这个假设无限期地延续。如果一只股票停牌超过一个季度（约65个交易日），我们就认为它停牌前的信息已经“陈腐”，不再具有代表性。limit=65正是为了控制这个风险。#
    # def _calculate_ps_ttm_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('ps_ttm').ffill(limit=65)
    def _calculate_total_mv_fill_limit65(self):
        return self.factor_manager.get_raw_factor('total_mv').copy(deep=True).ffill(limit=65)

    def _calculate_circ_mv_fill_limit65(self):
        return self.factor_manager.get_raw_factor('circ_mv').copy(deep=True).ffill(limit=65)

    # def _calculate_pb_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('pb').ffill(limit=65)
    # def _calculate_pe_ttm_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('pe_ttm').ffill(limit=65)

    ##
    # 交易行为类
    # 字段: turnover_rate, amount
    #
    # 金融含义: NaN代表停牌，当天没有发生任何交易行为。
    #
    # 下游需求: 滚动计算流动性因子时，需要处理这些NaN。
    #
    # 填充策略: FILL_STRATEGY_ZERO (填充为0)。
    #
    # 理由:
    #
    # 经济学假设：这是最符合事实的假设。停牌 = 0成交量 = 0成交额 = 0换手率。
    #
    # ffill的危害：如果你对turnover_rate进行ffill，就等于错误地假设“停牌日的热度=停牌前一天”，这与事实完全相反。#
    ##        # 金融逻辑：停牌日的真实换手率就是0


    ##
    #  静态信息类
    # 字段: list_date, delist_date
    #
    # 金融含义: NaN代表股票还未上市或尚未退市。
    #
    # 下游需求: 确定股票的生命周期。
    #
    # 填充策略: FILL_STRATEGY_FFILL (无限制前向填充)。
    #
    # 理由: 一只股票的上市日期是一个永恒不变的事实。一旦这个信息出现，它就对该股票的整个生命周期有效。因此，使用无限制的ffill将这个事实广播到所有后续的日期，是完全正确的。#
    def _calculate_delist_date_raw_ffill(self):
        return self.factor_manager.get_raw_factor('delist_date').copy(deep=True).ffill()
    def _calculate_list_date_ffill(self):
        return self.factor_manager.get_raw_factor('list_date').copy(deep=True).ffill()

    # === 质量/成长类 (Quality/Growth) 新增 ===
    #ok
    def _calculate_roe_change_q(self) -> pd.DataFrame:
        """
        计算单季度ROE的环比变化 (ROE Quarter-over-Quarter Change)。
        金融逻辑:
        捕捉公司盈利能力的“边际变化”或“加速度”。一个公司的ROE很高固然好，
        但如果其ROE正在持续改善（环比为正），这通常是更强的未来超额收益信号。
        这是一个结合了质量与动量思想的因子。
        """
        # 直接调用通用的季度因子计算引擎
        # 我们只需要提供：1.数据加载函数 2.源数据列名 3.具体的计算逻辑
        return self._create_general_quarterly_factor_engine(
            factor_name='roe_change_q',
            data_loader_func=load_fina_indicator_df,
            source_column='q_roe',  # Tushare财务指标接口中的“单季度净资产收益率”
            calculation_logic_func=self._qoq_change_logic  # 使用下方定义的通用环比计算逻辑
        )

    # === 动量类 (Momentum) 新增 ===

    #ok
    def _calculate_sharpe_momentum_60d(self) -> pd.DataFrame:
        """
        计算60日夏普动量 (Volatility-Adjusted Momentum)。

        金融逻辑:
        传统的动量因子只关心收益率，但高收益可能伴随着高风险。夏普动量通过用波动率
        来调整历史收益率，旨在寻找那些“稳定上涨”的股票，剔除纯粹因高波动而产生
        高收益的股票，信号质量通常更高。
        """
        logger.info("    > 正在计算因子: sharpe_momentum_pct_60d...")

        # 1. 获取日收益率
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg')

        # 2. 计算滚动均值和滚动标准差
        rolling_mean = pct_chg_df.rolling(window=60, min_periods=40).mean()
        rolling_std = pct_chg_df.rolling(window=60, min_periods=40).std()

        # 3. 风险控制：波动率（标准差）必须为正
        rolling_std_safe = rolling_std.where(rolling_std > 1e-6)  # 避免除以极小的数

        # 4. 计算夏普动量
        sharpe_momentum_df = rolling_mean / rolling_std_safe

        # 5. 后处理，清理无穷大值
        return sharpe_momentum_df.replace([np.inf, -np.inf], np.nan)
    # === 市场微观结构类 (market_microstructure) 新增 ===

    def _calculate_vwap_deviation_20d(self) -> pd.DataFrame:
        """
        计算20日VWAP偏离度  使用后复权VWAP)。
        金融逻辑:
        衡量后复权收盘价(close_hfq)相对于后复权全天交易均价(vwap_hfq)的偏离程度，
        并对该偏离度取20日平均，以获得更平滑的信号。
        """
        # 1. 获取基础数据 (现在口径完全统一)
        close_df = self.factor_manager.get_raw_factor('close_hfq')
        # 调用我们新建的、计算正确的后复权VWAP函数
        vwap_df = self.factor_manager.get_raw_factor('vwap_hfq')

        # 2. 对齐数据
        close_aligned, vwap_aligned = close_df.align(vwap_df, join='inner', axis=None)

        # 3. 风险控制：VWAP必须为正
        vwap_aligned_safe = vwap_aligned.where(vwap_aligned > 0)

        # 4. 计算每日偏离度
        daily_deviation = (close_aligned - vwap_aligned) / vwap_aligned_safe

        # 5. 计算20日滚动平均值，使信号更平滑
        return daily_deviation.rolling(window=20, min_periods=15).mean()


    # === 为通用引擎新增的、可复用的“计算逻辑”函数 ===

    def _qoq_change_logic(self, df: pd.DataFrame, col_name: str, factor_name: str) -> pd.DataFrame:
        """
        【可复用逻辑 - V3.0 重构版】计算单季度数据的环比变化值。
        通过调用通用脚手架工具，确保diff()总是在连续的季度之间进行。
        """
        logger.info(f"    > [V3.0 逻辑] 正在为 '{col_name}' 计算 QoQ 环比变化...")

        # 步骤一：【调用新工具】创建脚手架，让缺失的季度显式化为NaN
        merged_df = self._create_scaffold_and_merge_quarterly_data(df, 'end_date')

        # 步骤二：在时间连续的DataFrame上安全地计算环比
        # .diff() 会自动处理NaN，如果前一季度是NaN，则结果也是NaN
        merged_df[factor_name] = merged_df.groupby('ts_code')[col_name].diff(1)

        # 步骤三：只返回那些真实存在公告日的、且成功计算出环比值的行
        final_df = merged_df.dropna(subset=['ann_date', factor_name])

        return final_df

def _broadcast_ann_date_to_daily(
                                 sparse_wide_df: pd.DataFrame,
                                 trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    【核心通用工具】将一个基于稀疏公告日(ann_date)的宽表，
    安全地广播并填充到一个密集的交易日历上。

    这是解决所有财报类因子“期初NaN”问题的最终解决方案。

    Args:
        sparse_wide_df (pd.DataFrame): 以ann_date为索引的稀疏宽表。
        trading_dates (pd.DatetimeIndex): 目标交易日历。

    Returns:
        pd.DataFrame: 以交易日为索引的、被正确填充的稠密宽表。
    """
    # 1. 将稀疏的“公告日”索引与密集的“交易日”索引合并
    combined_index = sparse_wide_df.index.union(trading_dates)

    # 2. 扩展到“超级索引”上，然后进行决定性的前向填充
    filled_df = sparse_wide_df.reindex(combined_index).copy(deep=True).ffill()

    # 3. 最后，只裁剪出我们需要的交易日，并返回
    daily_df = filled_df.loc[trading_dates]

    return daily_df
# --- 如何在你的主流程中使用 (用法完全不变！) ---
# from data_manager import DataManager

# # 1. 初始化数据仓库
# dm = DataManager(...)
# dm.prepare_all_data()

# # 2. 初始化因子引擎 (内部已自动创建计算器)
# fm = FactorManager(data_manager=dm)

# # 3. 获取你需要的因子
# # 第一次获取 bm_ratio 时，FactorManager 会委托 Calculator 去计算
# bm_factor = fm.get_raw_factor('bm_ratio')

# # 第二次获取时，它会直接从 FactorManager 的缓存加载，速度极快
# bm_factor_again = fm.get_raw_factor('bm_ratio')

# print("\n最终得到的 bm_ratio 因子:")
# print(bm_factor.head())

# 【已删除】standardize_cross_sectionally 函数
# 原因：因子计算阶段不应该进行标准化，应该在预处理阶段统一处理
# 如果需要截面标准化，请使用 FactorProcessor._standardize_robust 方法


def calculate_rolling_beta_pure(
        stock_returns: pd.DataFrame,
        market_returns: pd.Series,
        window: int = 60,
        min_periods: int = 20
) -> pd.DataFrame:
    """
    【】根据输入的个股和市场收益率，计算滚动Beta。
    这是一个独立的计算引擎，不涉及任何数据加载或预处理。

    Args:
        stock_returns (pd.DataFrame): 个股收益率矩阵 (index=date, columns=stock)。
        market_returns (pd.Series): 市场收益率序列 (index=date)。
        window (int): 滚动窗口大小（天数）。
        min_periods (int): 窗口内计算所需的最小观测数。

    Returns:
        pd.DataFrame: 滚动Beta矩阵 (index=date, columns=stock)，未做任何日期截取或移位。
    """
    logger.info(f"  > 开始执行滚动Beta计算 (窗口: {window}天)...")

    # --- 1. 数据对齐 ---
    # 使用 'left' join 确保所有股票的日期和市场收益率对齐
    # 这是计算逻辑的核心部分，必须保留
    combined_df = stock_returns.join(market_returns.rename('market_return'), how='left')
    market_returns_aligned = combined_df.pop('market_return')

    # --- 2. 滚动计算 ---
    # Beta = Cov(R_stock, R_market) / Var(R_market)

    # a) 计算滚动协方差
    # 在对齐后，combined_df 就是我们要的 stock_returns
    rolling_cov = combined_df.rolling(window=window, min_periods=min_periods).cov(market_returns_aligned)

    # b) 计算市场收益率的滚动方差
    rolling_var = market_returns_aligned.rolling(window=window, min_periods=min_periods).var()

    # c) 计算滚动Beta
    beta_df = rolling_cov.div(rolling_var, axis=0)

    return beta_df


#工具类函数！
    #工具类函数
def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """对DataFrame进行截面标准化(Z-Score)"""
    return (df - df.mean(axis=1, skipna=True).values.reshape(-1, 1)) / df.std(axis=1, skipna=True).values.reshape(
        -1, 1)

if __name__ == '__main__':
    x = [1,1,1,2,2,2,3,3,3]
    x.rolling(window=20).std()