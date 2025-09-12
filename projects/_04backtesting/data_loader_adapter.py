
"""
数据加载适配器

将现有的ResultLoadManager数据适配到QuantBacktester需要的格式
提供标准化的数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


class BacktestDataLoader:
    """回测数据加载器"""
    
    def __init__(self, 
                 calcu_return_type: str = 'o2o',
                 version: str = '20190328_20231231',
                 is_raw_factor: bool = False):
        """
        初始化数据加载器
        
        Args:
            calcu_return_type: 收益计算类型
            version: 数据版本
            is_raw_factor: 是否为原始因子
        """
        self.result_manager = ResultLoadManager(
            calcu_return_type=calcu_return_type,
            version=version,
            is_raw_factor=is_raw_factor
        )
        
        logger.info(f"BacktestDataLoader初始化完成: {calcu_return_type}, {version}")
    
    def load_price_data(self,
                       stock_pool_index: str,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
        """
        加载价格数据
        
        Args:
            stock_pool_index: 股票池索引
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 价格数据，index为日期，columns为股票代码
        """
        logger.info(f"开始加载价格数据: {stock_pool_index}, {start_date} -> {end_date}")
        
        # 这里需要根据你的数据结构调整
        # 目前先生成模拟数据，实际使用时请替换为真实价格数据加载逻辑
        try:
            # 尝试获取一个因子数据来获取时间和股票维度信息
            sample_factor = self.result_manager.get_factor_data(
                'volatility_40d', stock_pool_index, start_date, end_date
            )
            
            if sample_factor is None:
                raise ValueError("无法获取样本因子数据来推断价格数据维度")
            
            dates = sample_factor.index
            stocks = sample_factor.columns
            
            logger.warning("⚠️ 使用模拟价格数据 - 实际使用时需要提供真实价格数据!")
            
            # 生成模拟价格数据
            np.random.seed(42)  # 保证可重现
            
            # 生成日收益率
            n_days, n_stocks = len(dates), len(stocks)
            daily_returns = np.random.normal(0.0003, 0.018, (n_days, n_stocks))
            
            # 加入一些市场相关性
            market_factor = np.random.normal(0.0005, 0.015, n_days)
            for i in range(n_stocks):
                beta = np.random.uniform(0.7, 1.3)  # 随机beta
                daily_returns[:, i] += beta * market_factor * 0.3
            
            # 转换为DataFrame
            returns_df = pd.DataFrame(daily_returns, index=dates, columns=stocks)
            
            # 计算价格序列（累积收益）
            initial_prices = np.random.uniform(8, 150, n_stocks)  # 随机初始价格
            price_df = pd.DataFrame(index=dates, columns=stocks)
            
            price_df.iloc[0] = initial_prices
            for i in range(1, len(dates)):
                price_df.iloc[i] = price_df.iloc[i-1] * (1 + returns_df.iloc[i])
            
            logger.info(f"价格数据生成完成: {price_df.shape}")
            return price_df.astype(float)
            
        except Exception as e:
            logger.error(f"价格数据加载失败: {e}")
            raise
    
    def load_factor_data(self,
                        factor_name: str,
                        stock_pool_index: str,
                        start_date: str,
                        end_date: str) -> Optional[pd.DataFrame]:
        """
        加载单个因子数据
        
        Args:
            factor_name: 因子名称
            stock_pool_index: 股票池索引
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame or None: 因子数据
        """
        logger.info(f"加载因子数据: {factor_name}")
        
        try:
            factor_data = self.result_manager.get_factor_data(
                factor_name, stock_pool_index, start_date, end_date
            )
            
            if factor_data is not None:
                logger.info(f"因子 {factor_name} 数据加载成功: {factor_data.shape}")
                return factor_data
            else:
                logger.warning(f"因子 {factor_name} 数据加载失败")
                return None
                
        except Exception as e:
            logger.error(f"因子 {factor_name} 数据加载异常: {e}")
            return None
    
    def load_multiple_factors(self,
                            factor_names: List[str],
                            stock_pool_index: str,
                            start_date: str,
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量加载多个因子数据
        
        Args:
            factor_names: 因子名称列表
            stock_pool_index: 股票池索引
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 因子数据字典
        """
        logger.info(f"批量加载 {len(factor_names)} 个因子数据")
        
        factor_dict = {}
        success_count = 0
        
        for factor_name in factor_names:
            factor_data = self.load_factor_data(
                factor_name, stock_pool_index, start_date, end_date
            )
            
            if factor_data is not None:
                factor_dict[factor_name] = factor_data
                success_count += 1
        
        logger.info(f"批量加载完成: 成功 {success_count}/{len(factor_names)} 个因子")
        return factor_dict
    
    def load_backtest_dataset(self,
                            factor_names: List[str],
                            stock_pool_index: str = '000906',
                            start_date: str = '2020-01-01',
                            end_date: str = '2023-12-31') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        加载完整的回测数据集
        
        Args:
            factor_names: 因子名称列表
            stock_pool_index: 股票池索引
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Tuple: (价格数据, 因子数据字典)
        """
        logger.info(f"加载完整回测数据集")
        logger.info(f"  股票池: {stock_pool_index}")
        logger.info(f"  时间范围: {start_date} -> {end_date}")
        logger.info(f"  因子数量: {len(factor_names)}")
        
        # 1. 加载因子数据
        factor_dict = self.load_multiple_factors(
            factor_names, stock_pool_index, start_date, end_date
        )
        
        if not factor_dict:
            raise ValueError("未能加载任何有效的因子数据")
        
        # 2. 加载价格数据
        price_df = self.load_price_data(stock_pool_index, start_date, end_date)
        
        logger.info(f"完整数据集加载完成:")
        logger.info(f"  价格数据: {price_df.shape}")
        logger.info(f"  有效因子: {len(factor_dict)}")
        
        return price_df, factor_dict


class PredefinedDatasets:
    """预定义数据集"""
    
    @staticmethod
    def get_champion_vs_composite(data_loader: BacktestDataLoader) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        获取冠军因子 vs 合成因子对比数据集
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            Tuple: (价格数据, 因子数据字典)
        """
        factor_names = [
            'volatility_40d',      # 冠军因子
            'lqs_orthogonal_v1'    # 你的合成因子
        ]
        
        return data_loader.load_backtest_dataset(
            factor_names=factor_names,
            stock_pool_index='000906',
            start_date='2020-01-01', 
            end_date='2023-12-31'
        )
    
    @staticmethod
    def get_top_factors_comparison(data_loader: BacktestDataLoader, 
                                 top_n: int = 5) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        获取头部因子对比数据集
        
        Args:
            data_loader: 数据加载器
            top_n: 选择前N个因子
            
        Returns:
            Tuple: (价格数据, 因子数据字典)
        """
        # 这里可以根据之前的IC分析结果选择top因子
        top_factors = [
            'volatility_40d',
            'momentum_60d', 
            'amihud_liquidity',
            'reversal_1d',
            'momentum_120d'
        ][:top_n]
        
        return data_loader.load_backtest_dataset(
            factor_names=top_factors,
            stock_pool_index='000906',
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
    
    @staticmethod
    def get_factor_category_comparison(data_loader: BacktestDataLoader) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        获取不同类别因子对比数据集
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            Tuple: (价格数据, 因子数据字典)
        """
        category_factors = {
            'volatility_40d': '波动率因子',
            'momentum_60d': '动量因子', 
            'ep_ratio': '价值因子',
            'roe_ttm': '质量因子',
            'amihud_liquidity': '流动性因子'
        }
        
        price_df, factor_dict = data_loader.load_backtest_dataset(
            factor_names=list(category_factors.keys()),
            stock_pool_index='000906',
            start_date='2020-01-01',
            end_date='2023-12-31'
        )
        
        # 重命名因子以包含类别信息
        renamed_factor_dict = {
            f"{name} ({category_factors[name]})": data
            for name, data in factor_dict.items()
        }
        
        return price_df, renamed_factor_dict


if __name__ == "__main__":
    # 示例用法
    logger.info("BacktestDataLoader 示例运行")
    
    # 创建数据加载器
    data_loader = BacktestDataLoader(
        calcu_return_type='o2o',
        version='20190328_20231231',
        is_raw_factor=False
    )
    
    try:
        # 示例1: 加载冠军因子 vs 合成因子数据
        logger.info("=== 示例1: 冠军因子 vs 合成因子 ===")
        price_df, factor_dict = PredefinedDatasets.get_champion_vs_composite(data_loader)
        
        logger.info("数据加载结果:")
        logger.info(f"  价格数据: {price_df.shape}")
        for name, data in factor_dict.items():
            logger.info(f"  {name}: {data.shape}")
        
        # 示例2: 加载头部因子对比数据
        logger.info("\n=== 示例2: 头部因子对比 ===")
        price_df2, factor_dict2 = PredefinedDatasets.get_top_factors_comparison(data_loader, top_n=3)
        
        logger.info("头部因子数据:")
        logger.info(f"  价格数据: {price_df2.shape}")
        for name, data in factor_dict2.items():
            logger.info(f"  {name}: {data.shape}")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
    
    logger.info("BacktestDataLoader 示例完成")