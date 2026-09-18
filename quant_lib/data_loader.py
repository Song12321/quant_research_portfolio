"""
数据加载模块

该模块提供了数据加载、处理和对齐的功能。
支持从本地文件、数据库和API加载数据。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from quant_lib import setup_logger
from quant_lib.config.constant_config import MARKET_DATA_ROOT, get_market_data_path

# 获取模块级别的logger
logger = setup_logger(__name__)


class DataLoader:
    """
    数据加载器类
    
    负责从各种数据源加载数据，并进行预处理、对齐等操作。
    支持本地Parquet文件、数据库和API数据源。
    
    Attributes:
        data_root (Path): 市场数据根目录
    """

    # ok
    def __init__(self, data_root: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_root: 市场数据根目录，如果为None则使用默认路径
        """
        self.data_root = MARKET_DATA_ROOT if data_root is None else Path(data_root)
        self.stock_data_root = self.data_root / 'stock'
        self.trade_cal = None

    def check_local_date_period_completeness(self, file_to_fields, start_date, end_date):
        for logical_name, columns_to_need_load in file_to_fields.items():
            logger.info(f"开始检查{logical_name} 时间段完整")
            file_path = get_market_data_path(logical_name, self.data_root)

            df = pd.read_parquet(file_path)
            if logical_name in ['index_daily.parquet', 'daily_basic', 'daily_basic', 'index_weights',
                                'daily', 'stk_limit', 'margin_detail']:
                self.check_local_date_period_completeness_for_trade(logical_name, df, start_date, end_date)
            if 'trade_cal.parquet' == logical_name:
                self.check_local_date_period_completeness_col(logical_name, df, 'cal_date', start_date, end_date)
            if 'namechange.parquet' == logical_name:
                self.check_local_date_period_completeness_col(logical_name, df, 'ann_date', start_date, end_date)
            if 'stock_basic.parquet' == logical_name:
                self.check_local_date_period_completeness_col(logical_name, df, 'list_date', start_date, end_date)
            if 'fina_indicator.parquet' == logical_name:
                self.check_local_date_period_completeness_col(logical_name, df, 'ann_date', start_date, end_date)

    def _load_trade_cal(self) -> pd.DataFrame:
        """加载交易日历"""
        try:
            trade_cal_df = pd.read_parquet(get_market_data_path('trade_cal.parquet', self.data_root))
            trade_cal_df['cal_date'] = pd.to_datetime(trade_cal_df['cal_date'])
            trade_cal_df=trade_cal_df.sort_values('cal_date', inplace=False)
            return trade_cal_df
        except Exception as e:
            logger.error(f"加载交易日历失败: {e}")
            raise

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """根据起止日期，从交易日历中获取交易日序列。"""
        mask = (self.trade_cal['cal_date'] >= start_date) & \
               (self.trade_cal['cal_date'] <= end_date) & \
               (self.trade_cal['is_open'] == 1)
        dates = pd.to_datetime(self.trade_cal.loc[mask, 'cal_date'].unique())
        return pd.DatetimeIndex(sorted(dates))  # 显式排序，确保有序

    def read_field(self, field, start_date, end_date, ts_codes=None):
        """从明确的数据集读取字段，不扫描文件推断来源。"""
        if field in ("open_raw", "close_raw", "high_raw", "low_raw", "vol_raw"):
            dataset, column = "daily", field.removesuffix("_raw")
        elif field == "amount":
            dataset, column = "daily", field
        elif field in ("circ_mv", "total_mv", "turnover_rate", "dv_ttm"):
            dataset, column = "daily_basic", field
        elif field == "adj_factor":
            dataset, column = "adj_factor", field
        elif field in ("list_date", "delist_date"):
            dataset, column = "stock_basic.parquet", field
        else:
            raise ValueError(f"未定义字段读取方式: {field}")
        return self._read_panel(dataset, column, start_date, end_date, ts_codes)

    def _read_panel(self, dataset, column, start_date, end_date, ts_codes):
        # 股票基本信息是静态数据，其余已支持的数据集使用交易日主键。
        keys = ["ts_code"] if dataset == "stock_basic.parquet" else ["ts_code", "trade_date"]
        frame = pd.read_parquet(
            get_market_data_path(dataset, self.data_root), columns=keys + [column]
        )
        frame = self.extract_during_period(frame, dataset, start_date, end_date)
        if ts_codes is not None:
            frame = frame[frame["ts_code"].isin(ts_codes)]
        trading_dates = self.get_trading_dates(start_date, end_date)
        if "trade_date" in keys:
            # 沿用既有时段筛选和同日记录取最后一条的规则。
            frame = frame[frame["trade_date"].isin(trading_dates)]
            frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
            return frame.pivot(index="trade_date", columns="ts_code", values=column)
        series = frame.drop_duplicates("ts_code").set_index("ts_code")[column]
        return pd.DataFrame(
            np.tile(series.values, (len(trading_dates), 1)),
            index=trading_dates, columns=series.index,
        )

    # ok
    def get_raw_dfs_by_require_fields(self,
                                      fields: List[str],
                                      buffer_start_date: str,
                                      end_date: str,
                                      ts_codes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载数据
        
        Args:
            fields: 需要加载的字段列表
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表，如果为None则加载所有股票
            
        Returns:
            字段到DataFrame的映射字典
        """
        logger.info(f"开始加载数据: 字段={fields}, 时间范围={buffer_start_date}至{end_date}")

        if not fields:
            raise ValueError("加载原始数据时 fields 不得为空")
        panels = {
            field: self.read_field(field, buffer_start_date, end_date, ts_codes)
            for field in sorted(set(fields))
        }
        return self._align_dataframes(panels)

    def _align_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:  # ok
        """
        【修复版】对齐多个DataFrame - 以主要数据表为基准，避免过度数据丢失

        Args:
            dfs: 字段到DataFrame的映射字典

        Returns:
            对齐后的DataFrame字典
        """
        if not dfs:
            raise ValueError("居然所传需对齐数据是空的")

        # 【修复】选择基准表 - 优先选择价格数据，其次选择覆盖度最高的表
        primary_candidates = ['close_raw', 'open_raw', 'high_raw', 'low_raw']
        base_key = None
        base_df = None

        # 首先尝试找到价格数据作为基准
        for candidate in primary_candidates:
            if candidate in dfs:
                base_key = candidate
                base_df = dfs[candidate]
                break

        # 如果没有价格数据，选择覆盖度最高的表
        if base_df is None:
            max_coverage = 0
            for name, df in dfs.items():
                coverage = df.notna().sum().sum()
                if coverage > max_coverage:
                    max_coverage = coverage
                    base_key = name
                    base_df = df

        logger.info(f"📊 数据对齐: 使用 '{base_key}' 作为基准表 {base_df.shape}")

        target_dates = base_df.index
        target_stocks = base_df.columns

        # 【修复】以基准表为准对齐所有数据，而不是取交集
        aligned_data = {}
        for name, df in dfs.items():
            aligned_df = df.reindex(index=target_dates, columns=target_stocks)
            aligned_df = aligned_df.sort_index()

            # 统计对齐后的覆盖度
            total_cells = aligned_df.size
            valid_cells = aligned_df.notna().sum().sum()
            coverage = valid_cells / total_cells if total_cells > 0 else 0
            logger.info(f"  {name}: 对齐后形状 {aligned_df.shape}, 覆盖度 {coverage:.1%}")

            # 不进行填充，保持原始缺失值，上层DataManager配合universe决定填充策略
            aligned_data[name] = aligned_df

        logger.info(f"数据对齐完成: {len(target_dates)}个交易日, {len(target_stocks)}只股票")
        return aligned_data

    def clear_cache(self):
        """清除缓存"""
        self.cache = {}
        logger.info("数据缓存已清除")

    def extract_during_period(self, long_df, logical_name, start_date, end_date):
        """
        根据时间范围筛选数据

        Args:
            long_df: 输入的DataFrame
            logical_name: 数据文件的逻辑名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            筛选后的DataFrame
        """
        if 'trade_date' in long_df.columns:
            long_df['trade_date'] = pd.to_datetime(long_df['trade_date'])
            long_df = long_df[
                (long_df['trade_date'] >= pd.Timestamp(start_date)) &
                (long_df['trade_date'] <= pd.Timestamp(end_date))
                ]
            return long_df
        # elif logical_name == 'stock_basic.parquet':
        #     # 对于股票基本信息，筛选上市日期早于开始日期的股票
        #     long_df['list_date'] = pd.to_datetime(long_df['list_date'])
        #     long_df = long_df[long_df['list_date'] < pd.Timestamp(start_date)]
        #
        #     # 添加交易日期列，便于数据统一处理
        #     trading_dates = get_trading_dates(start_date, end_date)#  待确认到底是 需要start_date end_date期间的交易日 ，还是连续的每日 确实需要这样！
        #     # 为每个股票创建所有交易日的记录
        #     stocks = long_df['ts_code'].unique()
        #     dates_df = pd.DataFrame(
        #         [(date, code) for date in trading_dates for code in stocks],
        #         columns=['trade_date', 'ts_code']
        #     )
        #     dates_df['trade_date'] = pd.to_datetime(dates_df['trade_date'])
        #
        #     # 合并基本信息到所有交易日
        #     result_df = pd.merge(dates_df, long_df, on='ts_code', how='left')
        #     return result_df

        return long_df  # 如果没有日期列，返回原始数据 反正后面有 对齐！

    def check_local_date_period_completeness_col(self, logical_name, df, col, start_date, end_date):
        df[col] = pd.to_datetime(df[col])
        min_date = df[col].min()
        max_date = df[col].max()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if min_date > start_date:
            raise ValueError(
                f"[{logical_name}] 最早 trade_date = {min_date.date()} 晚于 start_date = {start_date.date()} ❌")
        if max_date < end_date:
            raise ValueError(
                f"[{logical_name}] 最晚 trade_date = {max_date.date()} 早于 end_date = {end_date.date()} ❌")
        print(f"[{logical_name}] 日期覆盖完整 ✅")
        pass

    def check_local_date_period_completeness_for_namechange(self, logical_name, df, start_date, end_date):

        pass

    #
    # def rename_for_safe(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    #     """
    #     对加载的数据字典进行安全的重命名，将通用价格字段统一加上 _raw 后缀。
    #     确保下游模块接收到的是含义明确的数据。
    #     """
    #     # 创建一个新的字典来存储结果， 避免修改原始传入的对象
    #     renamed_data = aligned_data.copy()
    #
    #     # 定义需要被重命名的目标列
    #     cols_to_rename = ['close', 'open', 'high', 'low']
    #
    #     for old_name in cols_to_rename:
    #         # 检查旧的名称是否存在于字典中
    #         if old_name in renamed_data:
    #             new_name = f"{old_name}_hfq"
    #             # 使用 .pop() 方法，将旧键的值赋给新键，并从字典中移除旧键
    #             renamed_data[new_name] = renamed_data.pop(old_name)
    #
    #
    #     ##
    #     # 为什么 amount (成交额) 要用 raw 的？
    #     # 一句话概括：因为amount（成交额）是一个名义价值（Nominal Value）指标，它衡量的是“今天有多少钱在交易”，而这个问题的答案与历史上的分红送股无关。#
    #     if 'amount' in renamed_data:
    #         renamed_data['amount'] = renamed_data.pop('amount')
    #
    #     return renamed_data
    def fix_name_for_origin(self, field, logical_name):
        if field.endswith('_raw') & (logical_name == 'daily'):
            return field.replace('_raw', '')
        return field

    def fix_names_for_origin(self, columns_to_need_load, logical_name):
       return  [self.fix_name_for_origin(column,logical_name) for column in columns_to_need_load]

