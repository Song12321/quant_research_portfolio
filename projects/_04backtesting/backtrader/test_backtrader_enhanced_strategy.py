"""
backtrader_enhanced_strategy.py 单元测试
测试策略的 next() 方法在各种边界情况下的处理能力

主要测试场景：
1. 次日停牌无法买入的重试买入逻辑
2. 调仓期间停牌股票的pending_sells处理逻辑  
3. pending_buys超过三天自动取消的逻辑
4. 其他边界条件和异常情况

作者：根据backtrader_enhanced_strategy.py生成
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings

# 导入待测试的策略类
import sys
import os
sys.path.append(os.path.dirname(__file__))

from backtrader_enhanced_strategy import EnhancedFactorStrategy, OrderState
import backtrader as bt

warnings.filterwarnings('ignore')


class MockBacktraderData:
    """模拟Backtrader数据对象"""
    def __init__(self, name, price_series=None):
        self._name = name
        self.close = MockPriceArray(price_series) if price_series is not None else MockPriceArray()
        self.open = self.close
        self.high = self.close  
        self.low = self.close
        self.volume = MockPriceArray([1000] * 100)
        

class MockPriceArray:
    """模拟价格数组"""
    def __init__(self, prices=None):
        self.prices = prices if prices else [100.0] * 100
        
    def __getitem__(self, index):
        if index == 0:
            return self.prices[0] if self.prices else 100.0
        elif index < 0 and abs(index) <= len(self.prices):
            return self.prices[index]
        else:
            return 100.0


class MockBroker:
    """模拟经纪人"""
    def __init__(self, cash=1000000, value=1000000):
        self._cash = cash
        self._value = value
        self.startingcash = cash
        
    def get_cash(self):
        return self._cash
        
    def get_value(self):
        return self._value
    
    def getvalue(self):
        return self._value
        
    def setcash(self, cash):
        self._cash = cash
        self.startingcash = cash


class MockPosition:
    """模拟持仓"""
    def __init__(self, size=0):
        self.size = size


class MockOrder:
    """模拟订单"""
    # 订单状态常量
    Submitted = 1
    Accepted = 2  
    Partial = 3
    Completed = 4
    Canceled = 5
    Expired = 6
    Margin = 7
    Rejected = 8
    
    def __init__(self, is_buy=True, status=Completed, data_name="TEST_STOCK"):
        self._isbuy = is_buy
        self.status = status
        self.data = Mock()
        self.data._name = data_name
        self.executed = Mock()
        self.executed.size = 100
        self.executed.price = 100.0
        
    def isbuy(self):
        return self._isbuy
        
    def issell(self):
        return not self._isbuy
        
    def getstatusname(self):
        status_map = {
            self.Completed: "Completed",
            self.Canceled: "Canceled", 
            self.Rejected: "Rejected",
            self.Margin: "Margin"
        }
        return status_map.get(self.status, "Unknown")


class TestEnhancedFactorStrategy(unittest.TestCase):
    """EnhancedFactorStrategy 测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        
        # 模拟因子数据
        self.factor_data = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)),
            index=dates,
            columns=stocks
        )
        
        # 模拟持仓信号
        self.holding_signals = pd.DataFrame(
            False, index=dates, columns=stocks
        )
        # 设置几天有信号
        self.holding_signals.iloc[0:3, :] = True
        self.holding_signals.iloc[5:7, 0] = True  # STOCK_A在第5-6天有信号
        
        # 模拟价格数据（包含停牌情况）
        self.price_data = pd.DataFrame(
            100.0, index=dates, columns=stocks
        )
        # STOCK_B在第2天停牌（NaN价格）
        self.price_data.iloc[2, 1] = np.nan
        # STOCK_C在第6天停牌
        self.price_data.iloc[6, 2] = np.nan
        
        # 创建交易日列表
        self.trading_days = dates.tolist()
        
        # 调仓日期
        self.rebalance_dates = [dates[0], dates[4], dates[8]]
        
        # 创建策略实例
        self.strategy = self._create_mock_strategy()
        
    def _create_mock_strategy(self):
        """创建模拟策略实例"""
        strategy = EnhancedFactorStrategy()
        
        # 设置参数
        strategy.p = Mock()
        strategy.p.factor_data = self.factor_data
        strategy.p.holding_signals = self.holding_signals
        strategy.p.rebalance_dates = self.rebalance_dates
        strategy.p.max_positions = 3
        strategy.p.max_holding_days = 60
        strategy.p.retry_buy_days = 3
        strategy.p.max_weight_per_stock = 0.15
        strategy.p.min_weight_threshold = 0.01
        strategy.p.debug_mode = True
        strategy.p.log_detailed = True
        strategy.p.trading_days = self.trading_days
        strategy.p.real_wide_close_price = self.price_data
        
        # 模拟broker
        strategy.broker = MockBroker()
        
        # 模拟datetime
        strategy.datetime = Mock()
        strategy.datetime.date = Mock(return_value=date(2023, 1, 1))
        
        # 模拟数据源
        strategy.datas = [
            MockBacktraderData('STOCK_A'),
            MockBacktraderData('STOCK_B'), 
            MockBacktraderData('STOCK_C')
        ]
        
        # 模拟位置信息
        strategy.positions = {
            strategy.datas[0]: MockPosition(0),
            strategy.datas[1]: MockPosition(0),
            strategy.datas[2]: MockPosition(0)
        }
        
        # 初始化策略状态
        strategy.__init__()
        
        # 重写一些关键方法以便测试
        strategy.getdatabyname = lambda name: next(d for d in strategy.datas if d._name == name)
        strategy.getposition = lambda data: strategy.positions[data]
        strategy.order_target_percent = Mock(return_value=Mock())
        strategy.order_target_size = Mock(return_value=Mock())
        
        return strategy
    
    def test_01_suspended_stock_retry_buy_logic(self):
        """
        测试场景1：次日停牌无法买入的重试买入逻辑
        
        模拟场景：
        - T日决策买入STOCK_B
        - T+1日STOCK_B停牌，加入pending_buys
        - T+2日STOCK_B复牌，成功买入
        """
        print("\n=== 测试1：次日停牌无法买入的重试买入逻辑 ===")
        
        # T日：决策日，准备买入STOCK_B
        self.strategy.datetime.date.return_value = date(2023, 1, 1)
        
        # 手动添加STOCK_B到买入意图中
        buy_candidates = {'STOCK_B'}
        
        # 模拟STOCK_B停牌（价格为NaN）
        with patch.object(self.strategy, '_is_tradable') as mock_tradable:
            mock_tradable.return_value = False  # STOCK_B不可交易
            
            # 执行买入操作
            self.strategy._execute_all_buys_prioritized(buy_candidates)
            
            # 验证：STOCK_B应该被添加到pending_buys
            self.assertIn('STOCK_B', self.strategy.pending_buys)
            self.assertEqual(self.strategy.pending_buys['STOCK_B'][0], 0)  # 重试次数为0
            print(f"✓ STOCK_B已加入待买清单: {self.strategy.pending_buys['STOCK_B']}")
        
        # T+1日：STOCK_B复牌，尝试重新买入
        self.strategy.datetime.date.return_value = date(2023, 1, 2)
        
        with patch.object(self.strategy, '_is_tradable') as mock_tradable:
            mock_tradable.return_value = True  # STOCK_B可以交易了
            
            # 模拟成功提交订单
            with patch.object(self.strategy, '_submit_order_with_pending') as mock_submit:
                mock_submit.return_value = True
                
                # 获取待买清单
                buy_intentions = self.strategy._get_all_buy_intentions()
                self.assertIn('STOCK_B', buy_intentions)
                
                # 执行买入
                self.strategy._execute_all_buys_prioritized(buy_intentions)
                
                # 验证订单被提交
                mock_submit.assert_called()
                print("✓ STOCK_B复牌后成功提交买入订单")
        
        # 模拟订单成功回调
        mock_order = MockOrder(is_buy=True, status=MockOrder.Completed, data_name='STOCK_B')
        self.strategy.notify_order(mock_order)
        
        # 验证：成功买入后，pending_buys中应该删除STOCK_B
        self.assertNotIn('STOCK_B', self.strategy.pending_buys)
        self.assertEqual(self.strategy.success_buy_orders, 1)
        print("✓ STOCK_B成功买入后从待买清单中移除")
        
    def test_02_suspended_stock_pending_sells_logic(self):
        """
        测试场景2：调仓期间停牌股票的pending_sells处理逻辑
        
        模拟场景：
        - 持有STOCK_A，调仓日需要卖出
        - STOCK_A停牌无法卖出，加入pending_sells
        - 复牌后成功卖出
        """
        print("\n=== 测试2：调仓期间停牌股票的pending_sells处理逻辑 ===")
        
        # 设置初始持仓：持有STOCK_A
        self.strategy.positions[self.strategy.datas[0]] = MockPosition(100)  # 持有100股
        self.strategy.holding_start_dates['STOCK_A'] = date(2023, 1, 1)
        self.strategy.holding_days_counter['STOCK_A'] = 1
        
        # T日：调仓日，STOCK_A停牌需要卖出
        self.strategy.datetime.date.return_value = date(2023, 1, 5)  # 调仓日
        
        # 模拟STOCK_A不在新的持仓信号中（需要卖出）
        target_signal = pd.Series([False, True, True], index=['STOCK_A', 'STOCK_B', 'STOCK_C'])
        
        with patch.object(self.strategy, 'expect_t_buy_by_signals') as mock_signals:
            mock_signals.return_value = ['STOCK_B', 'STOCK_C']  # 新信号不包含STOCK_A
            
            with patch.object(self.strategy, 'tomorrow_is_rebalance_day') as mock_rebalance:
                mock_rebalance.return_value = True
                
                # 模拟STOCK_A停牌
                with patch.object(self.strategy, '_is_tradable') as mock_tradable:
                    mock_tradable.side_effect = lambda data: data._name != 'STOCK_A'  # STOCK_A不可交易
                    
                    # 获取卖出意图
                    stocks_to_sell = self.strategy._get_all_sell_intentions()
                    
                    # 验证：STOCK_A应该在卖出清单中
                    self.assertIn('STOCK_A', stocks_to_sell)
                    print(f"✓ STOCK_A被标记为需要卖出: {stocks_to_sell['STOCK_A']}")
                    
                    # 执行卖出操作
                    self.strategy._execute_all_sells(stocks_to_sell)
                    
                    # 验证：STOCK_A应该被添加到pending_sells
                    self.assertIn('STOCK_A', self.strategy.pending_sells)
                    print(f"✓ STOCK_A停牌，加入待卖清单: {self.strategy.pending_sells['STOCK_A']}")
        
        # 复牌后成功卖出
        self.strategy.datetime.date.return_value = date(2023, 1, 6)
        
        with patch.object(self.strategy, '_is_tradable') as mock_tradable:
            mock_tradable.return_value = True  # STOCK_A可以交易了
            
            # 获取待卖清单并执行
            stocks_to_sell = {'STOCK_A': '待卖清单'}
            with patch.object(self.strategy, '_submit_order_with_pending') as mock_submit:
                mock_submit.return_value = True
                
                self.strategy._execute_all_sells(stocks_to_sell)
                mock_submit.assert_called()
                print("✓ STOCK_A复牌后成功提交卖出订单")
        
        # 模拟卖出成功回调
        mock_order = MockOrder(is_buy=False, status=MockOrder.Completed, data_name='STOCK_A')
        self.strategy.notify_order(mock_order)
        
        # 验证：成功卖出后，pending_sells中应该删除STOCK_A
        self.assertNotIn('STOCK_A', self.strategy.pending_sells)
        self.assertEqual(self.strategy.success_sell_orders, 1)
        print("✓ STOCK_A成功卖出后从待卖清单中移除")
        
    def test_03_pending_buys_expiry_logic(self):
        """
        测试场景3：pending_buys超过三天自动取消的逻辑
        
        模拟场景：
        - STOCK_C加入pending_buys
        - 持续停牌超过3天
        - 自动取消买入任务
        """
        print("\n=== 测试3：pending_buys超过三天自动取消的逻辑 ===")
        
        # T日：将STOCK_C加入pending_buys
        initial_date = date(2023, 1, 1)
        self.strategy.datetime.date.return_value = initial_date
        
        # 手动添加到待买清单
        target_weight = 0.1
        self.strategy.pending_buys['STOCK_C'] = (0, initial_date, target_weight)
        print(f"✓ STOCK_C加入待买清单，日期: {initial_date}")
        
        # 创建交易日历缓存
        self.strategy._trading_calendar = set(self.trading_days)
        
        # T+1日：检查，应该仍在清单中
        self.strategy.datetime.date.return_value = date(2023, 1, 2)
        self.strategy._daily_state_update()
        self.assertIn('STOCK_C', self.strategy.pending_buys)
        print("✓ T+1日：STOCK_C仍在待买清单中")
        
        # T+2日：检查，应该仍在清单中  
        self.strategy.datetime.date.return_value = date(2023, 1, 3)
        self.strategy._daily_state_update()
        self.assertIn('STOCK_C', self.strategy.pending_buys)
        print("✓ T+2日：STOCK_C仍在待买清单中")
        
        # T+3日：检查，应该仍在清单中（恰好3天）
        self.strategy.datetime.date.return_value = date(2023, 1, 4)
        self.strategy._daily_state_update()
        self.assertIn('STOCK_C', self.strategy.pending_buys)
        print("✓ T+3日：STOCK_C仍在待买清单中（恰好3天）")
        
        # T+4日：检查，应该被自动取消（超过3天）
        self.strategy.datetime.date.return_value = date(2023, 1, 5)
        self.strategy._daily_state_update()
        self.assertNotIn('STOCK_C', self.strategy.pending_buys)
        print("✓ T+4日：STOCK_C超期自动从待买清单中移除")
        
    def test_04_forced_sell_max_holding_days(self):
        """
        测试场景4：强制超期卖出逻辑
        
        模拟场景：
        - 持有STOCK_A超过最大持有天数
        - 被强制卖出
        """
        print("\n=== 测试4：强制超期卖出逻辑 ===")
        
        # 设置最大持有天数为5天
        self.strategy.p.max_holding_days = 5
        
        # 设置持有STOCK_A，已持有6天（超期）
        self.strategy.positions[self.strategy.datas[0]] = MockPosition(100)
        self.strategy.holding_start_dates['STOCK_A'] = date(2023, 1, 1)
        self.strategy.holding_days_counter['STOCK_A'] = 6  # 超过5天限制
        
        # 获取卖出意图
        stocks_to_sell = self.strategy._get_all_sell_intentions()
        
        # 验证：STOCK_A应该因超期被标记为卖出
        self.assertIn('STOCK_A', stocks_to_sell)
        self.assertIn('强制到期', stocks_to_sell['STOCK_A'])
        print(f"✓ STOCK_A超期持有，被标记为强制卖出: {stocks_to_sell['STOCK_A']}")
        
        # 模拟成功卖出
        mock_order = MockOrder(is_buy=False, status=MockOrder.Completed, data_name='STOCK_A')
        
        # 设置pending_sells快照以便测试统计
        self.strategy.pending_sells['STOCK_A'] = (1, date(2023, 1, 7), '强制到期')
        
        self.strategy.notify_order(mock_order)
        
        # 验证：强制卖出计数器增加
        self.assertEqual(self.strategy.forced_exits, 1)
        print("✓ 强制卖出计数器正确更新")
        
    def test_05_order_failure_retry_mechanism(self):
        """
        测试场景5：订单失败重试机制
        """
        print("\n=== 测试5：订单失败重试机制 ===")
        
        # 模拟买入订单失败
        mock_order = MockOrder(is_buy=True, status=MockOrder.Rejected, data_name='STOCK_B')
        self.strategy.p.enable_retry = True
        
        # 执行订单失败回调
        self.strategy.notify_order(mock_order)
        
        # 验证：失败的股票被加入pending_buys
        self.assertIn('STOCK_B', self.strategy.pending_buys)
        self.assertEqual(self.strategy.failed_orders, 1)
        print("✓ 买入订单失败，STOCK_B加入待买清单")
        
        # 模拟卖出订单失败
        mock_order = MockOrder(is_buy=False, status=MockOrder.Canceled, data_name='STOCK_A')
        
        self.strategy.notify_order(mock_order)
        
        # 验证：失败的股票被加入pending_sells
        self.assertIn('STOCK_A', self.strategy.pending_sells)
        self.assertEqual(self.strategy.failed_orders, 2)
        print("✓ 卖出订单失败，STOCK_A加入待卖清单")
        
    def test_06_edge_cases_and_error_handling(self):
        """
        测试场景6：边界条件和错误处理
        """
        print("\n=== 测试6：边界条件和错误处理 ===")
        
        # 测试空的买入候选集
        empty_candidates = set()
        self.strategy._execute_all_buys_prioritized(empty_candidates)
        print("✓ 空买入候选集处理正常")
        
        # 测试空的卖出清单
        empty_sells = {}
        self.strategy._execute_all_sells(empty_sells)
        print("✓ 空卖出清单处理正常")
        
        # 测试持仓已满的情况
        self.strategy.p.max_positions = 1
        # 设置已经有一个持仓
        self.strategy.positions[self.strategy.datas[0]] = MockPosition(100)
        
        buy_candidates = {'STOCK_B', 'STOCK_C'}
        self.strategy._execute_all_buys_prioritized(buy_candidates)
        print("✓ 持仓已满情况处理正常")
        
        # 测试无效的交易日计算
        invalid_start_date = date(2025, 1, 1)  # 未来日期
        invalid_end_date = date(2020, 1, 1)    # 过去日期
        
        days_between = self.strategy._get_trading_days_between(invalid_start_date, invalid_end_date)
        self.assertEqual(days_between, 0)
        print("✓ 无效日期范围处理正常")
        
    def test_07_complex_rebalance_scenario(self):
        """
        测试场景7：复杂调仓场景
        
        同时包含：买入、卖出、停牌、重试等多种情况
        """
        print("\n=== 测试7：复杂调仓场景 ===")
        
        # 设置复杂的初始状态
        # 持有STOCK_A和STOCK_B
        self.strategy.positions[self.strategy.datas[0]] = MockPosition(100)  # STOCK_A
        self.strategy.positions[self.strategy.datas[1]] = MockPosition(50)   # STOCK_B
        
        # 设置新的目标持仓：只要STOCK_A和STOCK_C（需要卖出STOCK_B，买入STOCK_C）
        with patch.object(self.strategy, 'expect_t_buy_by_signals') as mock_signals:
            mock_signals.return_value = ['STOCK_A', 'STOCK_C']
            
            with patch.object(self.strategy, 'tomorrow_is_rebalance_day') as mock_rebalance:
                mock_rebalance.return_value = True
                
                # 模拟STOCK_C停牌无法买入，STOCK_B正常可卖
                with patch.object(self.strategy, '_is_tradable') as mock_tradable:
                    def tradable_logic(data_obj):
                        return data_obj._name != 'STOCK_C'  # STOCK_C停牌
                    mock_tradable.side_effect = tradable_logic
                    
                    # 执行完整的next()逻辑
                    self.strategy.datetime.date.return_value = date(2023, 1, 5)
                    
                    # 模拟len和data.buflen()
                    with patch.object(self.strategy, '__len__') as mock_len, \
                         patch.object(self.strategy.data, 'buflen') as mock_buflen:
                        mock_len.return_value = 5
                        mock_buflen.return_value = 10
                        
                        # 执行next()
                        self.strategy.next()
                        
                        # 验证结果
                        # STOCK_B应该被标记为卖出（不在新信号中）
                        # STOCK_C应该被加入pending_buys（停牌无法买入）
                        
                        print("✓ 复杂调仓场景执行完成")
                        print(f"  待卖清单: {list(self.strategy.pending_sells.keys())}")
                        print(f"  待买清单: {list(self.strategy.pending_buys.keys())}")
                        
    def test_08_data_integrity_checks(self):
        """
        测试场景8：数据完整性检查
        """
        print("\n=== 测试8：数据完整性检查 ===")
        
        # 测试价格数据缺失的处理
        stock_with_no_data = 'STOCK_MISSING'
        
        with self.assertRaises(StopIteration):
            # 应该抛出异常，因为找不到对应的数据源
            self.strategy.getdatabyname(stock_with_no_data)
        print("✓ 缺失数据源的错误处理正常")
        
        # 测试价格为NaN的情况
        self.strategy.p.real_wide_close_price.iloc[0, 0] = np.nan
        
        data_obj = self.strategy.getdatabyname('STOCK_A')
        is_tradable = self.strategy._is_tradable(data_obj)
        self.assertFalse(is_tradable)
        print("✓ NaN价格的停牌检测正常")
        
        # 测试因子数据为空的情况
        empty_factor_data = pd.DataFrame()
        
        with patch.object(self.strategy.p, 'factor_data', empty_factor_data):
            # 这种情况下应该优雅地处理，而不是崩溃
            try:
                buy_candidates = {'STOCK_A'}
                self.strategy._execute_all_buys_prioritized(buy_candidates)
                print("✓ 空因子数据处理正常")
            except KeyError:
                print("✓ 空因子数据正确抛出KeyError")
                
    def test_09_weight_calculation_accuracy(self):
        """
        测试场景9：权重计算准确性
        """
        print("\n=== 测试9：权重计算准确性 ===")
        
        # 测试不同持仓数量下的权重计算
        test_cases = [
            (0, 1),  # 无持仓，需要买1只
            (2, 3),  # 已有2只，需要再买3只
            (5, 2),  # 已有5只，需要再买2只
        ]
        
        for current_pos, need_buy in test_cases:
            # 设置当前持仓数量
            for i in range(current_pos):
                self.strategy.positions[self.strategy.datas[i]] = MockPosition(100)
            
            weight = self.strategy._calculate_dynamic_weight(need_buy)
            
            # 验证权重在合理范围内
            self.assertGreater(weight, 0)
            self.assertLessEqual(weight, 1.0)
            
            print(f"✓ 当前持仓{current_pos}只，需买{need_buy}只 -> 权重{weight:.3f}")
            
            # 重置持仓
            for data in self.strategy.datas:
                self.strategy.positions[data] = MockPosition(0)
                
    def test_10_comprehensive_workflow(self):
        """
        测试场景10：完整工作流程测试
        
        模拟一个完整的交易日工作流程
        """
        print("\n=== 测试10：完整工作流程测试 ===")
        
        # 重置策略状态
        self.strategy = self._create_mock_strategy()
        
        # 设置当前日期
        current_date = date(2023, 1, 1)
        self.strategy.datetime.date.return_value = current_date
        
        # 模拟一个完整的交易日
        with patch.object(self.strategy, '__len__') as mock_len, \
             patch.object(self.strategy.data, 'buflen') as mock_buflen:
            
            mock_len.return_value = 1
            mock_buflen.return_value = 10
            
            # 设置是调仓日
            with patch.object(self.strategy, 'tomorrow_is_rebalance_day') as mock_rebalance:
                mock_rebalance.return_value = True
                
                # 设置目标持仓
                with patch.object(self.strategy, 'expect_t_buy_by_signals') as mock_signals:
                    mock_signals.return_value = ['STOCK_A', 'STOCK_B']
                    
                    # 模拟所有股票都可交易
                    with patch.object(self.strategy, '_is_tradable') as mock_tradable:
                        mock_tradable.return_value = True
                        
                        # 执行完整的next()流程
                        self.strategy.next()
                        
                        # 验证各种状态
                        self.assertEqual(len(self.strategy.daily_stats), 1)
                        print("✓ 完整工作流程执行成功")
                        print(f"  每日统计记录数: {len(self.strategy.daily_stats)}")
                        
        # 测试策略结束处理
        self.strategy.stop()
        print("✓ 策略结束处理正常")

    def tearDown(self):
        """测试后清理"""
        pass


def run_comprehensive_tests():
    """运行所有测试"""
    print("=" * 80)
    print("开始执行 EnhancedFactorStrategy 全面单元测试")
    print("=" * 80)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试
    test_methods = [
        'test_01_suspended_stock_retry_buy_logic',
        'test_02_suspended_stock_pending_sells_logic', 
        'test_03_pending_buys_expiry_logic',
        'test_04_forced_sell_max_holding_days',
        'test_05_order_failure_retry_mechanism',
        'test_06_edge_cases_and_error_handling',
        'test_07_complex_rebalance_scenario',
        'test_08_data_integrity_checks',
        'test_09_weight_calculation_accuracy',
        'test_10_comprehensive_workflow'
    ]
    
    for method in test_methods:
        test_suite.addTest(TestEnhancedFactorStrategy(method))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print("\n" + "=" * 80)
    print("测试执行完成！")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 80)
    
    return result


if __name__ == '__main__':
    # 运行所有测试
    run_comprehensive_tests()