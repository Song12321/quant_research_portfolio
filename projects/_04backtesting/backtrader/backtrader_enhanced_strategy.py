"""
Backtrader增强策略 - 完整迁移vectorBT复杂逻辑

关键改进：
1. 完整迁移_generate_improved_signals的复杂状态管理
2. 自动处理停牌、重试、超期等所有边缘情况
3. 使用Backtrader事件驱动模型替代复杂for循环
4. 保持原有策略的所有核心逻辑和参数
"""
import math # 确保导入
import matplotlib
import matplotlib.pyplot as plt
import quantstats as qs

matplotlib.use('Agg')  # 非交互模式，不阻塞
import warnings
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple

import backtrader as bt
import numpy as np
import pandas as pd

from data.local_data_load import load_trading_lists, get_tomorrow_b_day
from quant_lib.utils.dataFrame_utils import align_dfs

warnings.filterwarnings('ignore')
from collections import defaultdict

from quant_lib.config.logger_config import setup_logger, log_success
from quant_lib.rebalance_utils import generate_rebalance_dates

logger = setup_logger(__name__)


class OrderState(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class EnhancedFactorStrategy(bt.Strategy):
    """
    增强因子策略 - 完整替代vectorBT复杂逻辑
    
    完整迁移原有的：
    - actual_holdings状态追踪
    - pending_exits_tracker（待卖清单）
    - pending_buys_tracker（待买清单）
    - 重试逻辑和有效期管理
    - 强制卖出超期持仓
    """

    params = (
        # 策略核心参数
        ('factor_data', None),  # 因子数据DataFrame
        ('rebalance_dates', []),  # 调仓日期列表
        ('top_quantile', []),  #
        ('max_positions', 10),  # 最大持仓数量
        ('max_holding_days', 60),  # 最大持仓天数（强制卖出）
        ('retry_buy_days', 3),  # 买入重试天数

        # 风控参数
        ('max_weight_per_stock', 0.15),  # 单股最大权重
        ('min_weight_threshold', 0.01),  # 最小权重阈值
        ('emergency_exit_threshold', 0.9),  # 紧急止损阈值

        # 调试参数
        ('debug_mode', True),  # 调试模式
        ('log_detailed', True),  # 详细日志
        ('enable_retry', True),  # 启用重试机制
        ('trading_days', None),  # 交易日列表
        ('_buy_success_num', {}),  #每日买入数量
        ('_sell_success_num', {}),  #每日卖出数量
        ('real_wide_close_price', None),  # 保存真实价格
        ('buy_after_sell_cooldown', None),  #
    )

    def __init__(self):
        """策略初始化 - 替代vectorBT的状态变量初始化"""
        logger.info("初始化EnhancedFactorStrategy...")

        # === 核心状态变量 - 完整替代vectorBT中的状态追踪 ===

        # 1. 调仓日期处理
        self.rebalance_dates_set = set(pd.to_datetime(self.p.rebalance_dates).date)

        # 2. 持仓状态追踪（替代actual_holdings）
        self.actual_positions = {}  # {stock_name: data_obj}
        self.holding_start_dates = {}  # {stock_name: entry_date}
        self.holding_days_counter = {}  # {stock_name: days}

        # 3. 待处理队列（替代pending_exits_tracker和pending_buys_tracker）
        self.pending_sells = {}  # {stock_name: (retry_count, target_date, reason)}
        self.pending_buys = {}  # {stock_name: (retry_count, target_date, target_weight)}

        # 4. 交易重试管理（完全替代vectorBT中的复杂重试逻辑）
        self.buy_retry_log = {}  # {stock_name: [失败日期列表]}
        self.sell_retry_log = {}  # {stock_name: [失败日期列表]}
        #交易冷却
        self.recently_sold={} #记录卖出前一天的第k根位置

        # 1. 加载未经shift的原始因子计算出的每日排名
        #    ranks.loc[T] 存储的是基于T日收盘信息计算出的排名
        #反正就要今日数据!.用于参考出rank排名,明天好买入!
        self.ranks = self.load_t_ranks_df()

        # 2. “明日目标持仓”状态变量， (每次调仓日才决定写入!!
        self.tomorrow_target_positions = set()

        # 5. 性能统计和调试
        self.daily_stats = []  # 每日统计信息
        self.rebalance_count = 0  # 调仓次数！
        self.submit_buy_orders = 0  #
        self.submit_sell_orders = 0
        self.success_buy_orders = 0
        self.success_sell_orders = 0
        self.failed_orders = 0  # （别担心，反正有重试

        # 6. 风险控制
        self.emergency_exits = 0  # 紧急止损次数
        self.forced_exits = 0  # 强制超期卖出次数
        self.real_wide_close_price = self.p.real_wide_close_price  # 保存真实价格


        ##权重
        # ...
        # 1. 组合目标总敞口 (例如，最多用90%的资金买股票)
        self.p.target_portfolio_exposure = 0.9

        # 容忍带参数
        self.p.tolerance_ratio = 0.10  # 相对比例部分：10%
        self.p.tolerance_abs = 0.02  # 绝对阈值部分：2%

        # 3. 股票池最大持仓数
        # self.p.max_positions =3  # 示例值

        # 4. 选股分位数
        # self.top_quantile = 0.1  # 示例值

        #订单 意图
        self.order_intents = {}
        # 辅助信息

        logger.info(f"策略初始化完成:")
        logger.info(f"  调仓日期: {len(self.rebalance_dates_set)}个")
        logger.info(f"  最大持仓: {self.p.max_positions}只")
        logger.info(f"  最大持有期: {self.p.max_holding_days}天")
        logger.info(f"  重试期限: {self.p.retry_buy_days}天")

    # 在你的策略类中，增加这个辅助方法
    def _is_weight_out_of_band(self, current_weight: float, ideal_weight: float) -> bool:
        """
        【混合模式】检查当前权重是否超出了动态计算的容忍带。
        返回: True (超出) / False (在带内)
        """
        # 1. 计算容忍带的宽度 (取相对值和绝对值中的较大者)
        tolerance = max(ideal_weight * self.p.tolerance_ratio, self.p.tolerance_abs)

        # 2. 计算上下轨
        upper_bound = ideal_weight + tolerance
        lower_bound = ideal_weight - tolerance

        # 3. 判断是否出界
        return not (lower_bound <= current_weight <= upper_bound)
    def load_t_ranks_df(self):
          origin_t1= self.p.factor_data.copy()#因为我们传入的t-1
          factor_df =origin_t1.shift(-1)
          return factor_df.rank(axis=1, pct=True, method='average', na_option='keep')

    # next机制：函数内部：决定好明天买什么！，明天9点半准时开盘价买入 （所以我给的信号，也是说明明天要买什么的信号！
    def expect_t_buy_by_signals(self):
        # """
        # 预期明天买入的股票 -
        # """
        # current_date = self.datetime.date(0)
        # target_holdings_signal = self.p.holding_signals.loc[pd.to_datetime(current_date)]
        # t_want_hold_stocks = target_holdings_signal[target_holdings_signal].index.tolist()
        #
        # return t_want_hold_stocks
        return self.tomorrow_target_positions
    def is_rebalance_day(self):
        current_date = self.datetime.date(0)
        return pd.to_datetime(current_date) in  self.p.rebalance_dates

    def init_data_for_target_positions_if_rebalance_day(self):
        current_date = self.datetime.date(0)
        #在调仓日，更新“战略目标  ###不然每天都任由每天的rank情况进行自由选股,那样手续费罩不住!
        if self.is_rebalance_day():
            try:
                # a. 获取今天的排名 (基于今天收盘的数据)
                daily_ranks = self.ranks.loc[pd.to_datetime(current_date)].dropna()

                if not daily_ranks.empty:
                    # b. 根据排名和规则选股
                    num_to_select = int(np.ceil(len(daily_ranks) * self.p.top_quantile))
                    if self.p.max_positions:
                        num_to_select = min(num_to_select, self.p.max_positions)

                    # c. 【核心】更新目标持仓状态变量
                    new_targets = set(daily_ranks.nlargest(num_to_select).index)
                    if self.tomorrow_target_positions != new_targets:
                        logger.info(f"【调仓日】明日待买目标数量：{len(new_targets)} 具体: {new_targets}")
                        self.tomorrow_target_positions = new_targets
            except KeyError:
                raise ValueError(f"在 {current_date} 无法找到因子排名数据。")

    def next(self):
        """
        统一的sell-buy分离架构 - 优先卖出释放现金，再统一买入
        """
        if len(self) == self.data.buflen():
            logger.warning("到达回测终点，不再为明天做决策。")
            return

        self.show_cur_day_first() #先打印!

        # === 第1步：日常状态更新 ===
        self._daily_state_update()
        self.init_data_for_target_positions_if_rebalance_day()
        # logger.info(f"每天打印明天需要买的今日需要挂单的---.>:{self.tomorrow_target_positions}")

        # 1. 【只收集，不执行】收集战术卖出意图
        tactical_sells = self._get_tactical_sell_intentions()

        # 2. 【分情况决策】
        if self.is_rebalance_day():
            # 调仓日：由“大脑”统一处理所有事情
            self._rebalance_portfolio(tactical_sells)
        else:
            # 非调仓日：只执行战术卖出和处理待买
            self._execute_all_sells(tactical_sells)
            self._execute_pending_buys_V2(set(self.pending_buys.keys()))
    #         self.buy_for_cash_reamain_to_many()
    # # def buy_for_cash_reamain_to_many(self):
    # #     cash = self.broker.get_cash()
    # #     value = self.broker.getvalue()
    # #     ratio = cash/value
    # #     if ratio > 0.20:

    # 在你的策略类中
    def _execute_pending_buys(self, stocks_to_try: set):
        """
        执行待买清单中的任务。
        它会使用记录在 pending_buys 中的目标权重。
        """
        if not stocks_to_try:
            return

        logger.info(f"检查待买清单中的 {len(stocks_to_try)} 个遗留任务...")

        # 使用 list(...) 创建副本进行安全迭代，因为 self.pending_buys 可能会在循环中被修改
        for stock_name in list(stocks_to_try):

            # a. 从“任务清单”中获取完整的指令（股票 + 权重）
            old_retrys,target_weight, _,_= self.pending_buys.get(stock_name)
            if target_weight is None:
                continue  # 如果因为某些原因任务已被移除，则跳过

            # b. 检查“执行可行性”（今天是否可交易）
            data_obj = self.getdatabyname(stock_name)
            if self._is_tradable(data_obj):
                logger.info(f"待买任务 '{stock_name}' 今日已复牌，明日开仓尝试以目标权重 {target_weight:.2%} 买入。")

                # c. 提交订单
                # 注意：我们在这里调用 _submit_order_with_pending
                # 如果提交成功，订单进入Broker。如果再次失败（比如现金还是不足），
                # _submit_order_with_pending 会确保它被重新放回 pending_buys，形成一个闭环。
                self._submit_order_with_pending(
                    stock_name=stock_name,
                    target_weight=target_weight,
                    reason="待买清单执行"
                )
            else:
                # 今日依然停牌，不做任何操作，任务保留在清单中，等待明天
                if self.p.debug_mode:
                    logger.debug(f"待买任务 '{stock_name}' 今日仍不可交易，继续等待。")

        # 在你的策略类中

    def _execute_pending_buys_V2(self, stocks_to_try: set):
        """
        执行待买清单中的任务。
        它会使用记录在 pending_buys 中的目标权重。

        更改：直接不买了！
        直接按当天rank 排序 重新选！
        """
        # if not stocks_to_try:
        #     return

        #持仓多少只
        #need_buy_num
        #权重计算
        #确定买的股票！
        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        need_buy_num =  self.p.max_positions-current_holdings_count
        logger.info(f"进入执行 cur:{current_holdings_count} ")

        if need_buy_num == 0:
            return
        logger.info(f"为了满仓 需要准备购入{need_buy_num}只股票")

        current_date = self.datetime.date(0)
        daily_ranks = self.ranks.loc[pd.to_datetime(current_date)].dropna()

        rank_stock_names = set(daily_ranks.nlargest(self.p.max_positions).index)
        positions  = self.getpositions()
        # 遍历持仓，提取股票名称
        current_names = set()
        for data, pos in positions.items():
            if pos.size != 0:  # 确保确实有持仓
                current_names.add(data._name)  # 或者 data._dataname，看你数据源怎么传入的

        diff_stocks = rank_stock_names - current_names
        selected_stock_name = sorted(diff_stocks, reverse=True)[:need_buy_num]

        #取最大的need_buy_num个
        logger.info(f"为了满仓运行 买入：{len(selected_stock_name)}个 : {selected_stock_name}")

        ideal_weight = self.p.target_portfolio_exposure /  self.p.max_positions
        # ideal_weight = self.p.target_portfolio_exposure / len(selected_stock_name) * 现金比例！
        for stock_name in selected_stock_name:
            self._submit_order_with_pending(
                stock_name=stock_name,
                target_weight=ideal_weight,
                reason="为了满仓运行 买入"
            )

    def tomorrow_is_rebalance_day(self):
        execution_date = get_tomorrow_b_day(self.p.trading_days, pd.Timestamp(self.datetime.date(0)))
        return execution_date in self.rebalance_dates_set

    def show_cur_day_first(self):
        current_date = self.datetime.date(0)
        log_success(f'{current_date}今天准备明天事宜')
        # === 当天开端统计信息 ===
        if self.p.log_detailed:
            self._log_daily_status(current_date)

    def _get_all_buy_intentions(self):
        """
          收集所有潜在的买入候选者（新的+待办的）。
          """
        # 1. 获取今天“理想计划”中的新买入目标
        target_signal = self.expect_t_buy_by_signals()

        current_positions = set([d._name for d in self.datas if self.getposition(d).size > 0])
        new_buy_candidates = target_signal - current_positions

        # 2. 获取“待买清单”中的旧目标
        pending_buy_candidates = set(self.pending_buys)
        logger.info(f'待买--->{pending_buy_candidates}')

        # 3. 返回合并后的总候选池
        buys = new_buy_candidates.union(pending_buy_candidates)
        buys = self._filter_cooldown_stocks(buys) #todo 最后排查完问题!开启
        return buys, set(self.pending_buys.keys()).__len__() == 0

    def _execute_all_buys_prioritized(self, buy_candidates: set, no_pending_buys: bool = None):
        """
        统一执行所有买入
        - 核心：对所有候选者，根据最新因子值进行重排，择优录取。
        """
        if not buy_candidates:
            return
        if no_pending_buys:#没有待买的股票 需要重新rank -->那就直接买!.
            self._execute_all_buys(buy_candidates)
            return
        current_date = self.datetime.date(0)

        # 1. 【择优】获取所有候选者“今天”的最新因子排名
        #    注意：我们索引的是 factor_data 的【今天】，它包含的是 T-1 的因子值
        try:  # todo思考 会不会有问题！，这里基于昨天收盘的后排序！ 然后明天开盘买入！ 显然不合理
            latest_ranks = self.p.factor_data.loc[pd.to_datetime(current_date), list(buy_candidates)].dropna()
        except KeyError:
            raise ValueError(f"[{current_date}] 无法获取部分候选者的最新因子值。")

        slots_available = len(buy_candidates)

        # 3. 从排名最高的候选者中，选出最终要买入的股票
        final_stocks_to_buy = latest_ranks.nlargest(slots_available).index.tolist()

        if not final_stocks_to_buy:
            # 这活生生的把新选出来的股票丢了？那不能！ (场景:当天满持仓,确实买不进去了,那么进入pending
            for buy_candidate in buy_candidates:
                self.push_to_pending_buys(buy_candidate)
            return

        # 4. 【分配权重】为最终入选者计算等权权重
        target_weight = self._calculate_dynamic_weight(
            need_buy_count=len(final_stocks_to_buy)
        )

        # 5. 【执行】为最终入选者下单
        for stock_name in final_stocks_to_buy:
            data_obj = self.getdatabyname(stock_name)

            if self._is_tradable(data_obj):
                self._submit_order_with_pending(
                    stock_name=stock_name,
                    target_weight=target_weight
                )
            else:
                # 停牌，加入/更新待买清单
                self.push_to_pending_buys(stock_name, "")

    def _get_tactical_sell_intentions(self) -> dict:
        """
        【升级版】只收集“战术性”（非调仓）的卖出意图。
        """
        """
                收集所有卖出意图（无副作用的只读函数）。
                返回: dict { stock_name: "原因1；原因2" }
                参数:
                    target_stocks_for_rebalance: 可选，调仓日的目标持仓列表（避免在本函数中调用 expect_t_buy_by_signals）
                """
        reasons = defaultdict(list)  # stock_name -> list of reason strings

        # 1. 待卖清单中的股票
        # 使用 list(...) 避免迭代时外部被修改导致问题
        for stock_name in list(self.pending_sells.keys()):
            reasons[stock_name].append("待卖清单")

        # 2. 强制到期的股票（max_holding_days）
        if self.p.max_holding_days is not None:
            for stock_name, days in list(self.holding_days_counter.items()):
                data_obj = self.getdatabyname(stock_name)
                # 只在确实持仓的情况下才考虑强制卖出
                if self.getposition(data_obj).size > 0 and days >= self.p.max_holding_days:
                    reasons[stock_name].append(f"强制到期:{days}天")

        # 3. 停牌股票（持仓期间发现停牌）
        for data_obj in self.datas:
            stock_name = data_obj._name
            if self.getposition(data_obj).size > 0 and not self._is_tradable(data_obj):
                # self.tomorrow_target_positions.discard(stock_name) #太暴力

                reasons[stock_name].append("停牌退出")

        # 最终格式化为字符串输出
        all_sells = {}
        for stock_name, reason_list in reasons.items():
            # 去重并保持顺序（如果需要）
            seen = set()
            deduped = []
            for r in reason_list:
                if r not in seen:
                    seen.add(r)
                    deduped.append(r)
            all_sells[stock_name] = "；".join(deduped)

        return all_sells

    def _get_all_sell_intentions(self, target_stocks_for_rebalance=None):
        """
        收集所有卖出意图（无副作用的只读函数）。
        返回: dict { stock_name: "原因1；原因2" }
        参数:
            target_stocks_for_rebalance: 可选，调仓日的目标持仓列表（避免在本函数中调用 expect_t_buy_by_signals）
        """
        reasons = defaultdict(list)  # stock_name -> list of reason strings

        # 1. 待卖清单中的股票
        # 使用 list(...) 避免迭代时外部被修改导致问题
        for stock_name in list(self.pending_sells.keys()):
            reasons[stock_name].append("待卖清单")

        # 2. 强制到期的股票（max_holding_days）
        if self.p.max_holding_days is not None:
            for stock_name, days in list(self.holding_days_counter.items()):
                data_obj = self.getdatabyname(stock_name)
                # 只在确实持仓的情况下才考虑强制卖出
                if self.getposition(data_obj).size > 0 and days >= self.p.max_holding_days:
                    reasons[stock_name].append(f"强制到期:{days}天")

        # 3. 停牌股票（持仓期间发现停牌）
        for data_obj in self.datas:
            stock_name = data_obj._name
            if self.getposition(data_obj).size > 0 and not self._is_tradable(data_obj):
                self.tomorrow_target_positions.discard(stock_name)

                reasons[stock_name].append("停牌退出")

        # 4. 调仓日不再持有的股票（避免在这里直接改变状态或调用副作用函数）
        if self.is_rebalance_day():
            # 推荐：外部先计算好 target_stocks 并传入；若为空则调用一次 expect_t_buy_by_signals（注意：确保此函数无副作用）
            if target_stocks_for_rebalance is None:
                # 如果你确定 expect_t_buy_by_signals 是纯的可重入函数，才这样调用
                target_stocks = self.expect_t_buy_by_signals()
            else:
                target_stocks = target_stocks_for_rebalance

            for data_obj in self.datas:
                stock_name = data_obj._name
                if self.getposition(data_obj).size > 0 and stock_name not in set(target_stocks):
                    reasons[stock_name].append("调仓卖出")

        # 最终格式化为字符串输出
        all_sells = {}
        for stock_name, reason_list in reasons.items():
            # 去重并保持顺序（如果需要）
            seen = set()
            deduped = []
            for r in reason_list:
                if r not in seen:
                    seen.add(r)
                    deduped.append(r)
            all_sells[stock_name] = "；".join(deduped)

        return all_sells

    # 在你的策略类中，增加这个辅助方法
    def _get_practical_target_weight(self, data_obj: bt.DataBase, ideal_weight: float) -> float:
        """
        根据理想权重，计算出最接近的、可用整手股数实现的“实用”目标权重。
        这是方案A的最终实现。
        """
        portfolio_value = self.broker.getvalue()
        price = data_obj.close[0]

        # 如果价格无效或总价值为0，则无法计算，返回0
        if price <= 0 or portfolio_value == 0:
            return 0.0

        # 1. 根据理想权重，计算出理想的、未取整的股数
        ideal_size = (portfolio_value * ideal_weight) / price

        # 2. 对理想股数进行向下取整（100的倍数）
        rounded_size = (ideal_size // 100) * 100

        # 3. 如果取整后股数为0（买不起一手），则目标权重也为0
        if rounded_size == 0:
            return 0.0

        # 4. 将取整后的股数，重新计算成一个“实用”的权重
        practical_weight = (rounded_size * price) / portfolio_value

        return practical_weight
    def _rebalance_portfolio(self, tactical_sells: dict):
        """
        原子性调仓】
        在调仓日当天，完成所有计算、意图生成和下单。
        融合了动态容忍带、最低持仓过滤、战术/战略合并、冷却风控等所有高级功能。
        """
        current_date = self.datetime.date(0)
        portfolio_value = self.broker.getvalue()
        current_positions_map = {d._name: self.getposition(d) for d in self.datas if self.getposition(d).size > 0}
        current_positions_set = set(current_positions_map.keys())

        # ==========================================================
        # 模块 1: 确定“理想中”的战略目标
        # ==========================================================
        try:
            daily_ranks = self.ranks.loc[pd.to_datetime(current_date)].dropna()
            #todo 测试！
            #快速颠倒一下rank
            # daily_ranks = -daily_ranks

            if daily_ranks.empty:
                logger.warning(f"调仓日 {current_date} 无有效因子排名，目标设定为空仓。")
                target_positions_ideal = set()
            else:
                num_to_select = int(np.ceil(len(daily_ranks) * self.p.top_quantile))
                if self.p.max_positions:
                    num_to_select = min(num_to_select, self.p.max_positions)
                target_positions_ideal = set(daily_ranks.nlargest(num_to_select).index)
        except KeyError:
            raise ValueError(f"在 {current_date} 无法找到排名数据，调仓失败。")
            return  # 关键数据缺失，直接中止调仓

        # ==========================================================
        # 模块 2: 【风控】分离决策，保护现有持仓，过滤新候选
        # ==========================================================
        positions_to_hold = current_positions_set & target_positions_ideal
        new_buy_candidates = target_positions_ideal - current_positions_set

        affordable_new_buys = set()
        if new_buy_candidates:
            # 预算是基于“理想”目标持仓数来平均计算的
            final_potential_count = len(target_positions_ideal)
            if final_potential_count > 0:
                ideal_weight_for_budgeting = self.p.target_portfolio_exposure / final_potential_count
                value_per_stock_budget = portfolio_value * ideal_weight_for_budgeting

                for stock_name in new_buy_candidates:
                    price = self.getdatabyname(stock_name).close[0]
                    if price * 100 <= value_per_stock_budget:
                        affordable_new_buys.add(stock_name)
                    else:
                        logger.info(
                            f"新候选 '{stock_name}' (价格:{price:.2f}) 因最低持仓金额不足(预算:{value_per_stock_budget:.2f})被剔除。")

        # 【得出最终目标】= 计划继续持有的 + 审查后买得起的新目标
        final_target_positions = positions_to_hold | affordable_new_buys
        logger.info(f"本日最终战略目标确定，共 {len(final_target_positions)} 只股票: {final_target_positions}")

        # ==========================================================
        # 模块 3: 计算最终理想权重
        # ==========================================================
        if not final_target_positions:
            ideal_weight = 0
        else:
            ideal_weight = self.p.target_portfolio_exposure / len(final_target_positions)

        # ==========================================================
        # 模块 4: 【完整版】统一决策与执行
        # ==========================================================

        # --- 决策 4.1: 处理清仓 ---
        # 卖出原因：1. 战术性卖出 2. 战略性卖出（不在最终目标池中）
        strategic_sells = {stock: "调仓卖出" for stock in (current_positions_set - final_target_positions)}
        final_sells = strategic_sells
        final_sells.update(tactical_sells)  # 用战术原因覆盖战略原因

        for stock_name, reason in final_sells.items():
            if stock_name in current_positions_map:
                self._submit_order_with_pending(stock_name, 0.0, reason)

        # --- 决策 4.2: 处理持仓调整 (应用容忍带) ---
        # 只对那些应该继续持有的股票，进行权重微调
        positions_to_adjust = current_positions_set & final_target_positions
        for stock_name in positions_to_adjust:
            # 双重保险：如果一只股票既要被持有又要被战术性卖出，听战术的，跳过调整
            if stock_name in final_sells:
                continue

            pos = current_positions_map[stock_name]
            current_weight = pos.size * pos.price / portfolio_value

            # 调用我们之前设计的混合模式容忍带函数
            if self._is_weight_out_of_band(current_weight, ideal_weight):
                self._submit_order_with_pending(stock_name, ideal_weight, "调仓-权重调整")

        # --- 决策 4.3: 处理新建仓 ---
        # 新建仓的来源，是之前审查过的“买得起”的股票池
        final_new_buy_candidates = final_target_positions - current_positions_set

        # 应用最终的行为层风控（冷却机制）
        final_buys = self._filter_cooldown_stocks(final_new_buy_candidates)

        for stock_name in final_buys:
            # 双重保险：确保不会买回在同一次决策中要被卖出的股票
            if stock_name in final_sells:
                continue

            data_obj = self.getdatabyname(stock_name)

            if self._is_tradable(data_obj):
                # 不直接使用 ideal_weight，而是先将其转换为 practical_weight
                practical_weight = self._get_practical_target_weight(data_obj, ideal_weight)

                if practical_weight > 0:
                    self._submit_order_with_pending(stock_name, practical_weight, "调仓-新建仓")
                else:
                    logger.info(f"'{stock_name}' 目标权重 {ideal_weight:.2%} 过小，无法买入一手，本次跳过。")
            else:
                self.push_to_pending_buys(stock_name, ideal_weight, "调仓新建时停牌")

        logger.info(f"====== 【调仓日】决策与only挂单执行完毕 ======")


    # 在你的策略类中
    def _filter_cooldown_stocks(self, buy_candidates: set) -> set:
        """从买入候选中，剔除正在冷却期的股票（基于K线索引计算）"""
        if not self.recently_sold:
            return buy_candidates

        current_bar_index = len(self)
        stocks_in_cooldown = set()

        for stock, sold_bar_index in list(self.recently_sold.items()):
            # 极其简单、高效且精确的整数减法
            bars_since_sold = current_bar_index - sold_bar_index

            if bars_since_sold < self.p.buy_after_sell_cooldown:
                stocks_in_cooldown.add(stock)
            else:
                # 冷却期已过，从字典中移除
                del self.recently_sold[stock]
                # logger.info(f"【冷却解除】: {stock} 已度过 {bars_since_sold} 根K线，解除冷却。")

        return buy_candidates - stocks_in_cooldown
    def _daily_state_update(self):#ok
        """
        每日状态更新 -
        """
        # 更新持仓天数
        for stock_name in list(self.holding_days_counter.keys()):
            data_obj = self.getdatabyname(stock_name)
            if self.getposition(data_obj).size > 0:
                self.holding_days_counter[stock_name] += 1
            else:
                # 清理已平仓的记录
                self._cleanup_position_records(stock_name)

        # 更新待买清单的"年龄"（替代pending_buys_age逻辑）
        current_date = self.datetime.date(0)
        for stock_name in list(self.pending_buys.keys()):
            retry_count, target_weight,target_date,_ = self.pending_buys[stock_name]
            trading_days_elapsed = self._get_trading_days_between(target_date, current_date)
            if trading_days_elapsed > self.p.retry_buy_days:
                # 超期，放弃买入
                self.del_pengding_buys_safe(stock_name)
                if self.p.debug_mode:
                    logger.info(f"买入任务超期放弃: {stock_name} 原定{target_date}买入，到现在{current_date}还没买入")

    def _get_trading_days_between(self, start_date, end_date):
        """计算两个日期间的交易日数量"""
        if not hasattr(self, '_trading_calendar'):
            # 缓存交易日历
            self._trading_calendar = set(self.p.trading_days)

        trading_days = 0
        current = start_date
        while current < end_date:
            current = get_tomorrow_b_day(self.p.trading_days, pd.to_datetime(current))
            if pd.Timestamp(current) in self._trading_calendar:
                trading_days += 1
        return trading_days

    def del_pengding_buys_safe(self, stock_name):
        if stock_name in self.pending_buys:
            del self.pending_buys[stock_name]

    def del_pengding_sells_safe(self, stock_name):
        if stock_name in self.pending_sells:
            del self.pending_sells[stock_name]

    def _get_rebalance_buy_intentions(self):
        """调仓日的买入意图"""
        target_stocks = self.expect_t_buy_by_signals()
        buy_list = {}

        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        target_stocks_num = min(len(target_stocks), self.p.max_positions)

        if target_stocks_num <= 0:
            return buy_list

        # 权重计算
        if current_holdings_count > 0:
            target_weight = 0.8 / target_stocks_num
        else:
            target_weight = 0.9 / target_stocks_num

        for stock_name in target_stocks:
            data_obj = self.getdatabyname(stock_name)
            if self.getposition(data_obj).size == 0:  # 只买入未持有的
                buy_list[stock_name] = target_weight

        return buy_list

    def _get_pending_buy_intentions(self):
        """非调仓日：处理待买清单"""
        buy_list = {}
        current_date = self.datetime.date(0)

        for stock_name in list(self.pending_buys.keys()):
            retry_count, target_date, target_weight = self.pending_buys[stock_name]
            trading_days_elapsed = self._get_trading_days_between(target_date, current_date)

            if trading_days_elapsed > self.p.retry_buy_days:
                self.del_pengding_buys_safe(stock_name)
            else:
                buy_list[stock_name] = target_weight

        return buy_list

    def _execute_all_sells(self, stocks_to_sell):
        """统一执行所有卖出"""
        for stock_name, reason in stocks_to_sell.items():
            data_obj = self.getdatabyname(stock_name)

            if self.getposition(data_obj).size > 0:
                self._submit_order_with_pending(
                    stock_name=stock_name,
                    target_weight=0,
                    reason=reason
                )

    def _execute_all_buys(self, stocks_to_buy):
        """统一执行所有买入"""
        target_weight = self._calculate_dynamic_weight(len(stocks_to_buy))
        for stock_name in stocks_to_buy:
            data_obj = self.getdatabyname(stock_name)

            if self._is_tradable(data_obj):
                #有必要判断,明天该买的股票 今天居然是停牌状态!,所以有必要让其加入pending,次日就不买! 次次日 发现pending有数据,会重新rank排序,如果发现排序依然在前面,才买!这样更安全
                self._submit_order_with_pending(
                    stock_name=stock_name,
                    target_weight=target_weight
                )
            else:
                # 停牌，保持在待买清单
                self.push_to_pending_buys(stock_name)

    def _check_position_limits(self):
        """检查持仓限制"""
        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        return current_holdings_count < self.p.max_positions

    # 调用函数之前，必须提前判断价格是否存在！
    # 买：
    # --调仓 信号买
    # --停牌-信号为F--复牌--信号--T 买 （需要考虑，我觉得也可也不用买了！万一本来没啥时效了！ （无需关注这个！，我们正常是调仓日才进行买！ 最多看一下待买清单，确保待买清单 只能是买入失败的！才能进入！
    # --上述结尾：提到的：待买清单买
    # 卖
    # --强制超期
    # --停牌危机
    # --调仓日发现信号没
    def _submit_order_with_pending(self, stock_name: str, target_weight: float, reason: str = 'normal') -> bool:
        """
        【升级版】调用统一的订单函数，并处理失败后的pending逻辑
        """
        # 调用新的、统一的核心函数
        is_submitted, action_type = self._submit_target_order(stock_name, target_weight, reason)

        if not is_submitted:
            # 提交失败，推入待办清单
            # 注意：这里的逻辑可以更精细，比如区分是买入待办还是卖出待办
            # 一个简单的判断：如果目标权重>0, 意图是买；如果目标=0, 意图是卖
            if target_weight > 0:
                self.push_to_pending_buys(stock_name, target_weight,f"挂单买入失败 (原因: {reason})")
            else:
                self.push_to_pending_sells(stock_name, f"挂单卖出失败 (原因: {reason})")
            return False
        return True

        return True

    def _submit_target_order(self, stock_name: str, target_weight: float, reason: str = 'normal') -> tuple[bool, str]:
        """
        【最终融合版】提交“目标权重”订单，并集成了你旧版函数中精细化的失败日志。
        """
        try:
            # 步骤 1: 预先收集所有用于决策和日志记录的数据
            data_obj = self.getdatabyname(stock_name)
            if not len(data_obj):  # 确保data_obj里有数据
                logger.warning(f"数据不足，无法为 {stock_name} 下单。")
                return False, "失败"

            # 步骤 1: 手动计算理论目标股数
            portfolio_value = self.broker.getvalue()
            target_value = portfolio_value * target_weight
            price = data_obj.close[0]
            if price <= 0 or np.isnan(price):
                return False, "没有获取到当日价格"

            ideal_size = target_value / price

            # 步骤 2: 【关键】执行A股的“整手”规则，自己掌控取整
            rounded_target_size = math.floor(ideal_size / 100.0) * 100.0
            #提交核心订单
            ##
            #     核心：放弃 order_target_percent，改用 order_target_size。
            #     因为我之前专门算好整100 size 转换成权重，然调用！，但是底层会自动画蛇添足 只卖n-1股！ （可能是因为手续费的原因！）
            #     #
            # order = self.order_target_percent(data=data_obj, target=target_weight)

            # 步骤 3: 使用 order_target_size 下达精确指令
            order = self.order_target_size(data=data_obj, target=rounded_target_size)

            current_pos = self.getposition(data_obj)
            current_size = current_pos.size
            current_price = data_obj.close[0]
            portfolio_value = self.broker.getvalue()
            current_weight = (current_size * current_price / portfolio_value) if portfolio_value != 0 else 0
            #当前资产
            value,_ = self.get_current_value_approximate()

            current_cash = self.broker.get_cash()
            if '新建仓' in reason:
                self.submit_buy_orders += 1

            action_type='UNKNOW'
            if '权重调整' in reason:
                action_type = '权重调整'


            # 步骤 3: 处理订单提交结果
            if order:
                # 【关键】订单已创建
                # order.created.size 会告诉你 backtrader 内部计算出的、未成交的理论股数
                logger.info(f"【Backtrader 内部计算】标的: {stock_name}, 理论目标股数: {order.created.size}")
                if action_type == 'UNKNOW' and order.isbuy():
                    action_type = "新建仓"
                elif action_type == 'UNKNOW' and order.issell():
                    action_type = "平仓"

                self.order_intents[order.ref] =  {'weight': target_weight, 'reason': reason}
                # 订单已成功提交到Broker - 逻辑不变，非常清晰

                logger.info(
                    f"\t\t\t\t{self.datetime.date(0)}-{action_type}-预备订单提交: {stock_name}, "
                    f"目标权重: {target_weight:.2%}, 原因: {reason}")
                return True, action_type
            else:
                # 订单未被创建 - 启动精细化失败原因分析

                # 用一个小的阈值来处理浮点数精度问题
                if target_weight == current_weight :
                    logger.info("权重已到位，无需交易")
                    return True, "无操作"  # 权重已到位，无需交易
                #size 一致无需交易
                if rounded_target_size == current_size:
                    logger.info("股数已到位，无需交易")
                    return True, "无操作"


                # b. 【完美移植】沿用你原有的、优秀的失败原因分析逻辑
                failure_reason = "未知原因"
                if reason in ['调仓-权重调整', '调仓卖出', '停牌退出', '强制到期']:
                    action_type = '平仓'
                    if current_size <= 0:
                        failure_reason = "无持仓可卖"
                    elif np.isnan(current_price) or current_price <= 0:
                        failure_reason = "停牌/价格异常无法卖出"
                    # 你可以继续加入更多卖出失败的判断...

                elif    reason in ['调仓-新建仓', '待买清单执行', '挂单买入失败 (原因: normal)']:
                    action_type = '新建仓'
                    required_cash = abs(portfolio_value * (target_weight - current_weight))
                    if current_cash < required_cash and current_cash < current_price * 100:
                        failure_reason = "现金不足"
                    elif np.isnan(current_price) or current_price <= 0:
                        failure_reason = "价格异常/停牌"
                    elif target_weight < 0.001:  # 假设目标权重为0是清仓意图
                        failure_reason = "目标权重过小"
                    else:  # 如果以上都不是，很可能是Broker的保证金检查失败
                        failure_reason = f"保证金不足或未知原因(目标权重{target_weight:.2%})"

                # c. 打印你想要的、详细的警告日志
                logger.warning(f"{self.datetime.date(0)}-{action_type}订单提交失败: {stock_name} (原因: {failure_reason})")
                logger.warning(
                    f"  详细状态: 现金={current_cash:.2f}, 总价值:{value},当前此股票持仓={current_size}, "
                    f"股票价格={current_price:.2f}, 当前权重={current_weight:.2%}, 目标权重={target_weight:.2%},"
                    f"权重差额={value*(target_weight - current_weight)}"
                )

                return False, "失败"

        except Exception as e:
            logger.error(f"{self.datetime.date(0)}-执行订单时异常 for {stock_name}: {e}")
            raise ValueError(e)

    def _submit_order(self, stock_name: str, data_obj, target_weight: float, action: str,
                      reason: str = 'narmal') -> bool:
        """
        提交订单 -
        Args:
            stock_name: 股票名称 data_obj: 数据对象 target_weight: 目标权重
        Returns:
            bool: 是否创单成功
        """
        if action == 'buy':
            self.submit_buy_orders += 1
        else:
            self.submit_sell_orders += 1
        try:
            current_position, current_cash, current_price = self.debug_data_for_submit(stock_name, action,
                                                                                       target_weight)
            # 对于卖出订单 （无脑强制卖）（为了照顾：如果是COC当日买卖限制  （第一个c是昨日收盘价！我们人为理解是昨天的买入！。但是此框架今天买入！。导致今天无法卖出！，所以出此下策！强制卖！
            if action == 'sell':
                order = self.order_target_size(data=data_obj, target=0)
                if order:
                    logger.info(
                        f"\t\t\t\t{self.datetime.date(0)}-{action}-预备订单提交(强制): {stock_name} reason:{reason}")
                    return True

            # 买入
            if action == 'buy':
                order = self.order_target_percent(data=data_obj, target=target_weight)
                if order:
                    logger.info(
                        f"\t\t\t\t{self.datetime.date(0)}-{action}预备订单提交: {stock_name}, 目标权重: {target_weight} reason:{reason}")
                    return True

            # 都是失败
            # 详细的失败原因分析
            failure_reason = "未知原因"
            if action == 'sell':
                if current_position <= 0:
                    failure_reason = "无持仓可卖"
                elif np.isnan(current_price) or current_price <= 0:
                    failure_reason = "停牌无法卖出"
                else:
                    # 检查是否是COC导致的当日买卖限制
                    if stock_name in self.holding_start_dates:
                        holding_start = self.holding_start_dates[stock_name]
                        if holding_start == self.datetime.date(0):
                            failure_reason = "COC当日买卖限制"
                        else:
                            failure_reason = "卖出订单被拒绝"
                    else:
                        failure_reason = "卖出订单被拒绝"
            elif action == 'buy':
                if current_cash < current_price * 100:  # 至少能买100股
                    failure_reason = "现金不足"
                elif np.isnan(current_price) or current_price <= 0:
                    failure_reason = "价格异常/停牌"
                elif target_weight < 0.001:
                    failure_reason = "目标权重过小"
                else:
                    failure_reason = f"未知原因-目标权重{target_weight}"

            logger.warning(f"{self.datetime.date(0)}-{action}预备订单提交失败: {stock_name} (原因: {failure_reason})")
            logger.warning(
                f"  现金: {current_cash:.2f}, 持仓: {current_position}, 价格: {current_price:.2f}, 目标权重: {target_weight}")
            return False

        except Exception as e:
            logger.error(f"{self.datetime.date(0)}-Error executing {action} order for {stock_name}: {e}")
            logger.error(f"  异常详情: 现金={self.broker.get_cash():.2f}, 价格={data_obj.close[0]:.2f}")
            raise ValueError(e)

    def _is_tradable(self, data_obj) -> bool:
        """
        检查股票是否可交易 -
        Args:  data_obj: 数据对象
        Returns:bool: 是否可交易
        """
        current_date = self.datetime.date(0)  # 可能需要+Bday ：最终解释：千万不要，那样偷看了未来！
        stock_name = data_obj._name

        price = self.p.real_wide_close_price.loc[pd.to_datetime(current_date), stock_name]
        #停牌只能查看真实原始数据
        if not np.isfinite(price):
            return False
        if data_obj.high[0] == data_obj.low[0]:
            # 这通常意味着股票全天一字板，无法买入也无法卖出
            logger.info(f"'{data_obj._name}' 在 {self.datetime.date(0)} 出现一字板，认定为不可交易。")
            return False

        return True

    def _cleanup_position_records(self, stock_name: str):
        """
        清理已平仓股票的所有记录
        Args:
            stock_name: 股票名称
        """
        records_to_clean = [
            self.holding_start_dates,
            self.holding_days_counter,
            self.actual_positions
        ]

        for record_dict in records_to_clean:
            if stock_name in record_dict:
                del record_dict[stock_name]

    def debug_data_for_submit(self, stock_name, action, target_weight):
        data_obj = self.getdatabyname(stock_name)
        current_position = self.getposition(data_obj).size
        # check
        if action == 'sell':
            if target_weight != 0:
                raise ValueError("卖出订单的目标权重必须为0")
            if current_position <= 0:
                raise ValueError("卖出订单 但是居然没有发现持仓（大概率是之前买入失败！严重错误 或者是这卖出信号不准！")

        # 增加调试信息：检查订单提交前的状态
        current_cash = self.broker.get_cash()

        current_price = data_obj.close[0]

        current_value, _ = self.get_current_value_approximate()

        if self.p.debug_mode:
            logger.debug(f"\t\t\t挂单前状态 - 现金: {current_cash:.2f}, 总价值: {current_value:.2f}, "
                         f"此次{action}目标: {stock_name}  价格: {current_price:.2f}, 当前已持仓: {current_position}")
        return current_position, current_cash, current_price

    def get_current_value_approximate(self):
        """
       解决Backtrader在某些情况下get_value返回NaN的问题
        Returns:
            float: 估算的总资产价值
        """
        current_value = self.broker.get_value()
        if not np.isnan(current_value):
            return current_value, False

        current_cash = self.broker.get_cash()

        # 估算总价值
        sum_value = 0
        for d, pos in self.positions.items():
            if pos.size != 0:
                valid_price = self.find_last_notNa_price(d)
                if not np.isnan(valid_price):
                    sum_value += pos.size * valid_price

        return current_cash + sum_value, True

    def find_last_notNa_price(self, data):
        """
           从当前bar往前找，返回最近一个非NaN的收盘价
           Args:
               data: Backtrader数据feed
           Returns:
               float: 最近的有效价格（找不到则返回 np.nan）
           """
        # 当前索引位置
        i = 0
        # 从当前开始往前查找
        while True:
            try:
                price = data.close[-i]
            except IndexError:
                # 超出历史范围，返回NaN
                return np.nan
            if not np.isnan(price):
                return price
            i += 1

    def refresh_for_success_buy(self, stock_name: str, pending_buys_snap):
        """
        成功买入后的记录刷新
        Args:
            stock_name: 股票名称
        """
        self.success_buy_orders += 1
        # 初始化持仓记录
        self.holding_start_dates[stock_name] = self.datetime.date(0)
        self.holding_days_counter[stock_name] = 1
        self.actual_positions[stock_name] = self.getdatabyname(stock_name)
        # 移除，反正我今天是买到了
        if stock_name in pending_buys_snap:
            self.pending_buys.pop(stock_name, None)

    def refresh_for_success_sell(self, stock_name: str, pending_sells_snap):
        """
        成功卖出后的记录刷新
        Args:
            stock_name: 股票名称
        """
        self.success_sell_orders += 1
        # 清理已平仓的记录
        self._cleanup_position_records(stock_name)
        # 移除，反正我今天是卖出去了
        self.pending_sells.pop(stock_name, None)

    def push_to_pending_sells(self, stock_name, descrip=""):
        old_retrys = 0
        if stock_name in self.pending_sells:
            old_retrys = self.pending_sells[stock_name][0]  # 获取重试次数
        trade_day = get_tomorrow_b_day(self.p.trading_days, pd.to_datetime(self.datetime.date(0)))

        self.pending_sells[stock_name] = (old_retrys + 1, trade_day, descrip)

    def push_to_pending_buys(self, stock_name,ideal_weight, descrip=None):
        if ideal_weight <= 0:#
            raise ValueError("权重计算严重有误!")
        old_retrys = 0
        if stock_name in self.pending_buys:
            old_retrys = self.pending_buys[stock_name][0]  # 获取重试次数

        trade_day = get_tomorrow_b_day(self.p.trading_days, pd.to_datetime(self.datetime.date(0)))
        self.pending_buys[stock_name] = (old_retrys + 1,ideal_weight, trade_day, descrip)

    def notify_order(self, order):
        """
        订单状态通知 - 增强的交易状态处理
        """
        #简单点！ 权重调整的订单 不进行任何处理

        stock_name = order.data._name
        current_date = self.datetime.date(0)
        pending_buys_snap = self.pending_buys
        pending_sells_snap = self.pending_sells
        if self.order_intents[order.ref]['reason'] is not None and '权重调整' in self.order_intents[order.ref][
            'reason']:
            action = '权重调整'
        else:
            action = "新建仓" if order.isbuy() else "平仓"

        # 订单成功执行
        if order.status == order.Completed:

            actionTimeType = "属于延迟日级别重试" if (
                    (stock_name in pending_sells_snap) or (stock_name in pending_buys_snap)) else "调仓"

            if order.isbuy():
                # 初始化持仓记录
                self.refresh_for_success_buy(stock_name, pending_buys_snap)
            if order.issell():
                # 卖出成功，清理记录
                self.refresh_for_success_sell(stock_name, pending_sells_snap)
                # 更新统计计数器（基于原始待卖记录中的原因）
                if stock_name in pending_sells_snap:
                    _, _, reason = pending_sells_snap[stock_name]
                    if "强制到期" in reason:
                        self.forced_exits += 1

            if self.p.log_detailed:
                execution_date = get_tomorrow_b_day(self.p.trading_days, pd.Timestamp(self.datetime.date(0)))
                logger.info(f"\t\t\t{execution_date}--{actionTimeType}-{action}-成功: {stock_name}, "
                            f"股数: {order.executed.size:.0f}, "
                            f"价格: {order.executed.price:.2f},"
                            f"乘积: {order.executed.price * order.executed.size}")
            # 存入map
            self.push_to_cur_day_num(execution_date, order)
            # 无论如何，事已至此  清理订单意图记录，
            self.order_intents.pop(order.ref, None)
            return
        # 订单失败处理
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:


            self.failed_orders += 1
            original_target_weight = self.order_intents.get(order.ref, {}).get('weight', None)

            # 记录失败原因
            failure_record = {
                'date': current_date,
                'stock': stock_name,
                'action': action,
                'status': order.getstatusname(),
                'price': order.data.close[0],
                'cash': self.broker.get_cash(),
                'value': self.broker.get_value(),
                'original_target_weight': original_target_weight
            }

            # 根据失败类型决定重试策略
            if order.isbuy() and self.p.enable_retry:
                # 买入失败，加入待买清单
                self.push_to_pending_buys(
                    stock_name=stock_name,
                    ideal_weight=original_target_weight,
                    descrip=f"{order.getstatusname()}"
                )
            elif order.issell():
                # 卖出失败，加入待卖清单
                self.push_to_pending_sells(stock_name, f"{failure_record['status']}")

            # 无论如何，事已至此  清理订单意图记录，
            self.order_intents.pop(order.ref, None)

            if self.p.debug_mode:
                logger.warning(f"{action}失败，加入重试: {stock_name}, 原因: {failure_record}")

    def notify_trade(self, trade):
        """当一笔交易关闭（平仓）时，记录该股票和当时的K线索引"""
        if trade.isclosed:
            stock_name = trade.data._name
            # len(self) 提供了当前K线的索引+1，是完美的“时间戳”
            bar_index_when_sold = len(self)
            self.recently_sold[stock_name] = bar_index_when_sold  # 记录的是整数索引
            # logger.info(f"【冷却记录】: {stock_name} 在第 {bar_index_when_sold} 根K线被卖出，进入冷却期。")
    def push_to_cur_day_num(self, execution_date, order):
        origin_num = 0
        if order.isbuy():
            if execution_date in self.p._buy_success_num:
                origin_num = self.p._buy_success_num[execution_date]
            self.p._buy_success_num[execution_date] = origin_num + 1
            return
        if order.issell():
            if execution_date in self.p._sell_success_num:
                origin_num = self.p._sell_success_num[execution_date]
            self.p._sell_success_num[execution_date] = origin_num + 1

    # 注意场景！
    def _calculate_dynamic_weight(self, need_buy_count, ) -> float:  # todo 需要测试 回测
        """
        动态计算目标权重 - 根据当前现金和持仓情况
        Returns:
            float: 动态计算的目标权重
        """
        # 计算当前实际持仓数量
        current_positions = len([d for d in self.datas if self.getposition(d).size > 0])

        # 总目标持仓数
        total_target = min(self.p.max_positions, current_positions + need_buy_count)

        # 动态权重分配
        if current_positions <= 0:  # 么有持仓
            return 0.9 / total_target
        else:
            return 0.9 / self.p.max_positions

    def _log_daily_status(self, current_date):
        """
        每日状态 - 用于调试和监控

        Args:
            current_date: 当前日期
        """
        # 统计当前状态
        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        pending_sells_count = len(self.pending_sells)
        pending_buys_count = len(self.pending_buys)
        total_value, _ = self.get_current_value_approximate()
        cash = self.broker.get_cash()

        daily_stat = {
            'date': current_date,
            'holdings': current_holdings_count,
            'pending_sells': pending_sells_count,
            'pending_buys': pending_buys_count,
            'total_value': total_value,
            'cash_ratio': cash / total_value,
        }

        self.daily_stats.append(daily_stat)

        if self.p.debug_mode:
            logger.info(f"\t\t{current_date}：今日开盘前 实际持仓{current_holdings_count}只, "
                        f"pending_sells_count {pending_sells_count}只, pending_buys_count {pending_buys_count}只, "
                        f"现金{cash:.1f}--总价值{total_value}")

    def stop(self):
        """策略结束处理 - 详细统计和分析"""
        logger.info("=" * 80)
        logger.info("策略执行完成 - 详细统计报告")
        logger.info("=" * 80)

        # 基本统计
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100

        logger.info(f"资金统计:")
        logger.info(f"  初始资金: {self.broker.startingcash:,.2f}")
        logger.info(f"  最终资金: {self.broker.getvalue():,.2f}")
        logger.info(f"  总收益率: { (final_value / self.broker.startingcash - 1) * 100:.2f}%")

        # 交易统计
        success_rate = self.success_buy_orders / max(self.submit_buy_orders, 1) * 100
        # 统计基准！！
        logger.info(f"交易统计:")
        logger.info(f"  总提交订单数（提交买入订单数）: {self.submit_buy_orders}")
        logger.info(f"  失败订单（别担心，反正有重试: {self.failed_orders}")
        logger.info(f"  买入成功率: {self.success_buy_orders / max(self.submit_buy_orders, 1) * 100:.1f}%")

        # 调仓统计
        logger.info(f"调仓统计:")
        logger.info(f"  调仓次数: {len(self.rebalance_dates_set)}")
        logger.info(f"  强制到期: {self.forced_exits}次")
        logger.info(f"  紧急止损: {self.emergency_exits}次")

        # 待处理队列统计
        if self.pending_buys or self.pending_sells:
            logger.info(f"未完成任务:")
            logger.info(f"  待买清单: {len(self.pending_buys)}只")
            logger.info(f"  待卖清单: {len(self.pending_sells)}只")

            if self.pending_buys:
                logger.info("  待买股票: %s", list(self.pending_buys.keys()))
            if self.pending_sells:
                logger.info("  待卖股票: %s", list(self.pending_sells.keys()))

        # 持仓分析
        self._analyze_holding_patterns()

    def _analyze_holding_patterns(self):
        """
        分析持仓模式 - 替代vectorBT中的_debug_holding_days逻辑
        """
        if not self.daily_stats:
            return

        logger.info("持仓模式分析:")

        # 转换为DataFrame进行分析
        stats_df = pd.DataFrame(self.daily_stats)

        avg_holdings = stats_df['holdings'].mean()
        max_holdings = stats_df['holdings'].max()
        min_holdings = stats_df['holdings'].min()

        avg_cash_ratio = stats_df['cash_ratio'].mean()

        logger.info(f"  平均持仓: {avg_holdings:.1f}只")
        logger.info(f"  最大持仓: {max_holdings}只")
        logger.info(f"  最小持仓: {min_holdings}只")
        logger.info(f"  平均现金比例: {avg_cash_ratio:.1%}")

        # 分析待处理队列的变化
        avg_pending_buys = stats_df['pending_buys'].mean()
        avg_pending_sells = stats_df['pending_sells'].mean()

        if avg_pending_buys > 0.5:
            logger.warning(f"⚠️ 买入执行困难，平均待买: {avg_pending_buys:.1f}只")
        if avg_pending_sells > 0.5:
            logger.warning(f"⚠️ 卖出执行困难，平均待卖: {avg_pending_sells:.1f}只")



class AShareCommission(bt.CommInfoBase):
    params = (
        ('stamp_duty', 0.001),  # 印花税，千分之一
        ('commission', 0.0001),  # 手续费，万分之一点五
        ('stocklike', True),
        ('percabs', True),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        这个函数只负责计算【佣金】和【税费】，滑点由框架自动处理
        """
        if size > 0:  # 买入
            return abs(size) * price * self.p.commission
        elif size < 0:  # 卖出
            return abs(size) * price * (self.p.commission + self.p.stamp_duty)
        else:
            return 0
class BacktraderMigrationEngine:
    """
    Backtrader迁移引擎 - 一键式从vectorBT迁移的完整解决方案
    """

    def __init__(self, original_config=None):
        """
        初始化迁移引擎
        
        Args:
            original_config: 原有的vectorBT配置对象
        """
        self.original_config = original_config
        self.bt_config = self._convert_config(original_config)
        self.results = {}

        logger.info("BacktraderMigrationEngine初始化完成")

    def _convert_config(self, vectorbt_config) -> Dict:
        """
        配置转换 - 从vectorBT配置转换为Backtrader配置
        
        Args:
            vectorbt_config: 原有配置对象
            
        Returns:
            Dict: Backtrader配置字典
        """
        if vectorbt_config is None:
            return self._default_config()

        return {
            'top_quantile': getattr(vectorbt_config, 'top_quantile', 0.2),
            'rebalancing_freq': getattr(vectorbt_config, 'rebalancing_freq', 'M'),
            'commission_rate': getattr(vectorbt_config, 'commission_rate', 0.0003),
            'slippage_rate': getattr(vectorbt_config, 'slippage_rate', 0.001),
            'stamp_duty': getattr(vectorbt_config, 'stamp_duty', 0.001),
            'initial_cash': getattr(vectorbt_config, 'initial_cash', 1000000.0),
            'max_positions': getattr(vectorbt_config, 'max_positions', 10),
            'max_holding_days': getattr(vectorbt_config, 'max_holding_days', 60),
            'retry_buy_days': 3,
            'max_weight_per_stock': getattr(vectorbt_config, 'max_weight_per_stock', 0.15),
            'min_weight_threshold': getattr(vectorbt_config, 'min_weight_threshold', 0.01),
            'buy_after_sell_cooldown': getattr(vectorbt_config, 'buy_after_sell_cooldown', 0.01),
        }

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'top_quantile': 0.2,
            'rebalancing_freq': 'M',
            'commission_rate': 0.0003,
            'slippage_rate': 0.001,
            'stamp_duty': 0.001,
            'initial_cash': 1000000.0,
            'max_positions': 10,
            'max_holding_days': 60,
            'retry_buy_days': 3,
            'max_weight_per_stock': 0.15,
            'min_weight_threshold': 0.01
        }

    def migrate_and_run(self, price_dfs: Dict[str,pd.DataFrame], factor_dict: Dict[str, pd.DataFrame],
                        ) -> Dict:
        """
        一键迁移并运行 - 完整替代原有的run_backtest函数
        
        Args:
            close_df: 价格数据
            factor_dict: 因子数据字典
            comparison_with_vectorbt: 是否与vectorBT结果对比
        Returns:
            Dict: 迁移结果
        """
        migration_results = {}

        for factor_name, factor_data in factor_dict.items():

            try:
                cerebro = bt.Cerebro(quicknotify=True)
                # cerebro.broker.set_coo(True)

                factor_data = self.reshape_del_always_tottom(factor_data)
                all_dfs =  align_dfs(**price_dfs, **{'factor_data':factor_data})
                # ===：创建两份价格数据 ===

                # 添加数据源
                self.add_wide_df_to_cerebro(cerebro, all_dfs)
                # for d in cerebro.datas:
                #     for i in range(len(d)):
                #         print(d.datetime.date(i), d.open[i])
                # 生成调仓日期
                rebalance_dates = generate_rebalance_dates(
                    all_dfs['factor_data'].index,
                    self.bt_config['rebalancing_freq']
                )

                # 添加策略
                cerebro.addstrategy(
                    EnhancedFactorStrategy,
                    factor_data=all_dfs['factor_data'],
                    rebalance_dates=rebalance_dates,
                    max_positions=self.bt_config['max_positions'],
                    top_quantile=self.bt_config['top_quantile'],
                    max_holding_days=self.bt_config['max_holding_days'],
                    retry_buy_days=self.bt_config['retry_buy_days'],
                    buy_after_sell_cooldown=self.bt_config['buy_after_sell_cooldown'],
                    debug_mode=True,
                    trading_days=load_trading_lists(all_dfs['factor_data'].index[0], all_dfs['factor_data'].index[-1]),
                    real_wide_close_price=all_dfs['close'],
                    log_detailed=True,
                )

                # === 4. 配置交易环境 ===
                cerebro.broker.setcash(self.bt_config['initial_cash'])

                # cerebro.broker.setcommission(commission=comprehensive_fee)
                commission_scheme = AShareCommission(stamp_duty=self.bt_config['stamp_duty'], commission=self.bt_config['commission_rate'])


                # 3. 将这位配置齐全的“全能会计师”指派给Broker
                cerebro.broker.addcommissioninfo(commission_scheme)

                cerebro.broker.  set_slippage_perc(self.bt_config['slippage_rate'], slip_open=True, slip_limit=True, slip_match=True, slip_out=False)

                # === 5. 添加分析器 ===
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

                # === 6. 执行回测 ===
                logger.info(f"开始执行{factor_name}回测...")
                start_time = datetime.now()

                strategy_results = cerebro.run()

                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                strategy = strategy_results[0]

                # 3. 【核心】从分析器中，提取出每日收益率序列
                #    get_analysis() 返回的是一个字典，key是日期，value是当天的收益率
                #    我们把它转换成一个 pandas.Series，这是 QuantStats 最喜欢的格式
                daily_returns_dict = strategy.analyzers.time_return.get_analysis()
                daily_returns_series = pd.Series(daily_returns_dict)

                # 确保索引是 DatetimeIndex
                daily_returns_series.index = pd.to_datetime(daily_returns_series.index)
                # 4. 生成一份完整的、漂亮的、可交互的HTML报告
                #    output 参数会让它保存成一个文件，而不是试图弹出窗口
                report_path = 'my_strategy_report.html'
                qs.reports.html(daily_returns_series,
                                output=report_path,
                                title='My First Factor Strategy Report')

                logger.info(f"专业的策略表现报告已生成: {report_path}")

                # === 7. 提取结果 ===
                strategy = strategy_results[0]
                final_value = cerebro.broker.getvalue()

                migration_results[factor_name] = {
                    'strategy': strategy,
                    'final_value': final_value,
                    'execution_time': execution_time,
                    'analyzers': strategy.analyzers,
                    'config_used': self.bt_config.copy()
                }


            except Exception as e:
                raise ValueError("失败") from e

        self.results = migration_results
        self.show_per_day_trade_details()

        return migration_results

    import backtrader as bt
    import pandas as pd

    def add_wide_df_to_cerebro(self, cerebro: bt.Cerebro,all_dfs) -> None:

        #对必须要ffill                ( 一份用于“欺骗”Backtrader底层引擎，确保getvalue()正常

        open_df = all_dfs['open'].fillna(method='ffill').fillna(method='bfill')
        high_df = all_dfs['high'].fillna(method='ffill').fillna(method='bfill')
        low_df = all_dfs['low'].fillna(method='ffill').fillna(method='bfill')
        close_df = all_dfs['close'].fillna(method='ffill').fillna(method='bfill')
        factor_wide_df = all_dfs['factor_data']

        logger.info(f"add_wide_df_to_cerebro.")
        for stock_symbol in close_df.columns:
            # 1. 创建一个不带索引的DataFrame
            df_single_stock = pd.DataFrame()

            # 2. 【核心】将日期从索引变成一个名为'datetime'的普通列
            df_single_stock['datetime'] = pd.to_datetime(close_df.index)

            #    使用 .values 可以避免pandas版本差异带来的索引对齐问题
            df_single_stock['open'] = open_df[stock_symbol].values
            df_single_stock['high'] = high_df[stock_symbol].values
            df_single_stock['low'] = low_df[stock_symbol].values
            df_single_stock['close'] = close_df[stock_symbol].values
            df_single_stock['volume'] = 0
            df_single_stock['openinterest'] = 0
            # if stock_symbol == 'STOCK_B':
            #     #所有价格*1.5
            #     df_single_stock[['open','close','volume','low','high']]=df_single_stock[['open','close','volume','low','high']]*1.5

            # #mock数据
            # if stock_symbol == 'STOCK_B':
            #     df_single_stock.loc[df_single_stock.index[2],['open','close','volume','low','high','openinterest']]=np.nan
            # 4. 【核心】调用PandasData，并明确告知每一列的位置 (mapping)
            data_feed = bt.feeds.PandasData(
                dataname=df_single_stock,

                # --- 手动指定“表格”的每一列 ---
                datetime=0,  # 第0列是日期时间
                open=1,  # 第1列是开盘价
                high=2,  # 第2列是最高价
                low=3,  # 第3列是最低价
                close=4,  # 第4列是收盘价
                volume=5,  # 第5列是成交量
                openinterest=6  # 第6列是持仓量
            )

            cerebro.adddata(data_feed, name=stock_symbol)
        logger.info(f"\n成功为 {len(cerebro.datas)} 只股票添加了独立的数据源。")

    def _align_data(self, price_df: pd.DataFrame, factor_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        数据对齐 - 兼容原有的vectorBT对齐逻辑
        
        Args:
            price_df: 价格数据
            factor_df: 因子数据
        Returns:
            Tuple: 对齐后的(价格数据, 因子数据)
        """
        # 时间对齐
        common_dates = price_df.index.intersection(factor_df.index)

        # 股票对齐  
        common_stocks = price_df.columns.intersection(factor_df.columns)

        aligned_price = price_df.loc[common_dates, common_stocks]
        aligned_factor = factor_df.loc[common_dates, common_stocks]

        return aligned_price, aligned_factor

    # def _generate_holding_signals_for_next(self, factor_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    #     holding_signals = self._generate_holding_signals(factor_df, price_df)
    #     # 将 T 日的交易计划，移动到 T-1 日的行上，为策略的 next() 方法做好准备
    #     final_signals_for_strategy = holding_signals.shift(-1)
    #     return final_signals_for_strategy
    # def _generate_holding_signals(self, factor_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     生成持仓信号 -：factor_df t行数据 是t-1的数据！。也就说：t row rank排名：本质上是昨日数据rank的结果！
    #     那么next走入t行，拿到这些昨日rank的结果，然后t+1开盘下单，显然有延迟！
    #     解决办法：利用_generate_holding_signals_for_next ：调用包装这个_generate_holding_signals ，主要shift(-1)
    #     Args:
    #         factor_df: 对齐后的因子数据
    #         price_df: 对齐后的价格数据
    #
    #     Returns:
    #         pd.DataFrame: 持仓信号矩阵
    #     """
    #     # 安全起见，  factor_df跟price_df 对照一下，如果价格为nan，那么把factor也置为nan，然后再去rank
    #     is_tradable_mask = price_df.notna() & (price_df > 0)
    #     factor_df = factor_df.where(is_tradable_mask)
    #     # 计算每日排名百分位
    #     ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')
    #
    #     # 生成调仓日期
    #     rebalance_dates = generate_rebalance_dates(factor_df.index, self.bt_config['rebalancing_freq'])
    #
    #     # 初始化持仓信号矩阵
    #     holding_signals = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)
    #
    #     # 当前持仓组合（调仓间隔期间保持不变）
    #     current_positions = None
    #
    #     for date in factor_df.index:
    #         is_rebalance_day = date in rebalance_dates
    #
    #         if is_rebalance_day:
    #             # 调仓日：重新选择股票
    #             daily_valid_ranks = ranks.loc[date].dropna()
    #
    #             if len(daily_valid_ranks) > 0:
    #                 # 计算目标持仓数
    #                 num_to_select = int(np.ceil(len(daily_valid_ranks) * self.bt_config['top_quantile']))
    #                 if self.bt_config['max_positions']:
    #                     num_to_select = min(num_to_select, self.bt_config['max_positions'])
    #
    #                 # 选择排名最高的股票
    #                 chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index
    #                 current_positions = chosen_stocks
    #
    #         # 重复前一天
    #         if current_positions is not None:
    #             holding_signals.loc[date, current_positions] = True
    #
    #     # 兜底保证，停牌日的持仓信号为False 放弃！
    #     # holding_signals[price_df.isna() | (price_df <= 0)] = False#下游策略“信息缺失”，无法区分“主动调仓”与“被动停牌”
    #     # 思考：还是有必要置为false ，因为backTrade用的fill数据价格，无法感知停牌！next（我们手动判断：也只能满足：每天看看持有的：哪些停牌了，得卖出。而无法满足：拦截买入时判断停牌状态！
    #     # 假设两个仓之间：A因为停牌，卖出，但是信号一直为True，每天都在尝试买入! 假设后面可买入，准备明天开盘买，但是明天又停牌了!导致买入失败！ （但是好像也不大
    #     # holding_signals[price_df.isna() | (price_df <= 0)] = False
    #
    #     holding_signals = holding_signals.loc[:, holding_signals.any(axis=0)].astype(bool)
    #     return holding_signals.astype(bool)


    def show_per_day_trade_details(self):
        #根据 _buy_success_num _sell_success_num 统计每天情况
        if not self.results:
            return
        for factor_name, result in self.results.items():
            strategy = result.get('strategy', None)
            if strategy is None:
                continue
            buy_nums = strategy.p._buy_success_num
            sell_nums = strategy.p._sell_success_num
            if not buy_nums and not sell_nums:
                continue
            all_dates = set(buy_nums.keys()).union(set(sell_nums.keys()))
            all_dates = sorted(all_dates)[-5:] # 只看最近5天
            logger.info(f"每日交易详情 - 因子: {factor_name}")
            for date in all_dates:
                buys = buy_nums.get(date, 0)
                sells = sell_nums.get(date, 0)
                logger.info(f"  {date}: 买入成功 {buys} 笔, 卖出成功 {sells} 笔")
            logger.info("-" * 40)

        pass

    def reshape_del_always_tottom(self, factor_df):
        # 计算每日排名百分位
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')
        # 生成调仓日期
        rebalance_dates = rebalance_dates = generate_rebalance_dates(factor_df.index, self.bt_config['rebalancing_freq'])
        # 初始化持仓信号矩阵
        holding_signals = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)

        # 当前持仓组合（调仓间隔期间保持不变）
        current_positions = None

        for date in factor_df.index:
            is_rebalance_day = date in rebalance_dates

            if is_rebalance_day:
                # 调仓日：重新选择股票
                daily_valid_ranks = ranks.loc[date].dropna()

                if len(daily_valid_ranks) > 0:
                    # 计算目标持仓数
                    num_to_select = int(np.ceil(len(daily_valid_ranks) * (self.bt_config['top_quantile']+0.1))) #加0.1 为了安全
                    if self.bt_config['max_positions']:
                        num_to_select = int(min(num_to_select, self.bt_config['max_positions']*1.1)) #因为我没考虑停牌的,!这里多加入

                    # 选择排名最高的股票
                    chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index
                    current_positions = chosen_stocks

            # 重复前一天
            if current_positions is not None:
                holding_signals.loc[date, current_positions] = True

        holding_signals = holding_signals.loc[:, holding_signals.any(axis=0)].astype(bool)
        #裁剪
        factor_df = factor_df[holding_signals.columns]
        return factor_df


# === 便捷迁移函数 ===

def one_click_migration(price_dfs:Dict[str, pd.DataFrame], factor_dict: Dict[str, pd.DataFrame],
                        original_vectorbt_config=None) -> Dict:
    """
    Args:
        price_df: 价格数据
        factor_dict: 因子数据字典  
            original_vectorbt_config: 原有的vectorBT配置
    Returns:
        Tuple: (Backtrader回测结果, 对比表)
    """
    # 创建迁移引擎
    migration_engine = BacktraderMigrationEngine(original_vectorbt_config)

    # 执行迁移和回测
    results = migration_engine.migrate_and_run(price_dfs, factor_dict)
    return results


# 运行测试
class A_ShareRoundLotSizer(bt.sizers.PercentSizer):
    """
    A股整手（100股）Sizer
    - 继承自 PercentSizer，完美配合 order_target_percent
    - 只重写 _getsizing 方法，加入“向下取整到100的倍数”的逻辑
    - 保持了 Sizer 的无状态性和职责单一性
    """
    params = (('lot', 100),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        # 1. 调用父类（PercentSizer）的_getsizing方法，获取理论上的、未取整的理想股数
        #    我们不重复造轮子，让 backtrader 的原生逻辑先计算
        size = super()._getsizing(comminfo, cash, data, isbuy)
        # print(f"DEBUGGGGG: {data._name} size={size} isbuy={isbuy}")

        # 如果计算出的size是0，直接返回0
        if size == 0:
            return 0

        # 2. 对理想股数，执行我们唯一的、核心的逻辑：向下取整到 lot 的倍数
        #    math.floor(size / self.p.lot) * self.p.lot 也是可以的
        rounded_size = (size // 100) * 100

        # 3. 返回处理后的最终股数
        return rounded_size
if __name__ == "__main__":
    print()







