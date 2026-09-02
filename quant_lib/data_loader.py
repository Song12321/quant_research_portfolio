"""
数据加载模块

该模块提供了数据加载、处理和对齐的功能。
支持从本地文件、数据库和API加载数据。
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict

from odbc import noError

from quant_lib import setup_logger
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR

# 获取模块级别的logger
logger = setup_logger(__name__)


class DataLoader:
    """
    数据加载器类
    
    负责从各种数据源加载数据，并进行预处理、对齐等操作。
    支持本地Parquet文件、数据库和API数据源。
    
    Attributes:
        data_path (Path): 数据存储路径
        cache (Dict): 数据缓存
        field_map (Dict): 字段到数据源的映射
    """

    # ok
    def __init__(self, data_path: Optional[Path] = None, use_cache: bool = True):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据存储路径，如果为None则使用默认路径
            use_cache: 是否使用内存缓存
        """
        self.data_path = data_path or LOCAL_PARQUET_DATA_DIR

        if not self.data_path.exists():
            os.makedirs(self.data_path, exist_ok=True)
            logger.info(f"数据路径不存在,现已创建数据路径: {self.data_path}")

        self.available_columns_by_file = defaultdict(set)
        self.field_map = self._build_field_map_to_file_name()
        logger.info(f"字段->所在文件Name--映射构建完毕，共发现 {len(self.field_map)} 个字段")
        # 在初始化时就加载交易日历，因为它是后续操作的基础(此处还没区分是否open，是全部
        self.trade_cal = self._load_trade_cal()

    def check_local_date_period_completeness(self, file_to_fields, start_date, end_date):
        for logical_name, columns_to_need_load in file_to_fields.items():
            logger.info(f"开始检查{logical_name} 时间段完整")
            file_path = self.data_path / logical_name

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
            trade_cal_df = pd.read_parquet(self.data_path / 'trade_cal.parquet')
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

    def _build_field_map_to_file_name(self) -> Dict[str, str]:
        """
        构建字段到数据源的映射
        
        Returns:
            字段到数据源的映射字典
        """
        field_to_files_map = {}

        # 递归查找所有parquet文件
        for file_path in self.data_path.rglob('*.parquet'):
            try:
                # 只读取schema以获取列名
                columns = pq.read_schema(file_path).names

                # data.xxx 就是逻辑数据集名称（即：按年份分区的数据
                if file_path.stem == 'data':
                    # 分区数据
                    logical_name = file_path.parent.parent.name
                else:
                    # 单文件
                    logical_name = file_path.stem + '.parquet'
                self.available_columns_by_file[logical_name].update(columns)

                # 构建字段映射
                for col in columns:
                    if (col in ['total_mv', 'circ_mv', 'turnover_rate']) & (
                            logical_name != 'daily_basic'):
                        continue
                    if (col in ['list_date', 'delist_date']) & (
                            logical_name != 'stock_basic.parquet'):
                        continue
                    if (col in ['close', 'open', 'high', 'low']) & (  # 实测 amount 和vol 在daily和 在daily_hfq数值一模一样！
                            logical_name == 'daily_hfq'):  # ，我们需要daily_hfq(后复权的数据)表里面的数据 #最新修改 手动计算，不依赖不纯洁的hfq
                        field_to_files_map[col + '_hfq'] = logical_name
                        continue
                    if (col in ['close', 'vol']) & (
                            logical_name == 'daily'):
                        field_to_files_map[col + '_raw'] = logical_name
                        continue
                    if (col in ['amount']) & (
                            logical_name == 'daily'):
                        field_to_files_map[col] = logical_name
                        continue
                        # 'turnover_rate', 'circ_mv', 'total_mv'  这些是“纯净原材料”，它们是每日更新的、不依赖于财报发布时间的随时点（Point-in-Time）数据
                    not_allow_load_fieds_for_not_fq = ['adj_factor', 'pe_ttm', 'pb', 'ps_ttm']
                    not_allow_load_fieds_for_不知道内部逻辑_万一人家数据有前世偏差呢 = [ 'pe_ttm', 'pb', 'ps_ttm']
                    if (col in not_allow_load_fieds_for_not_fq or (col in not_allow_load_fieds_for_不知道内部逻辑_万一人家数据有前世偏差呢)):  #
                        continue  # (f'严谨加载依赖报告日发布的数据{col} 非daily数据 ,请将config 用adj_factor的 from_daily配置 置为false，这样就不会此阶段加载了')
                    if col not in field_to_files_map:
                        field_to_files_map[col] = logical_name
            except Exception as e:
                logger.error(f"读取文件 {file_path} 的元数据失败: {e}")

        return field_to_files_map

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

        # 确定需要加载的数据集和字段
        file_to_fields = defaultdict(list)
        base_fields = ['ts_code', 'trade_date']

        if not fields:
            raise ValueError("加载原始数据时 fields 不得为空")

        for field in sorted(set(fields)):
            logical_name = self.field_map.get(field)
            if logical_name is None:
                raise ValueError(f"未找到字段 {field} 的数据源")

            file_to_fields[logical_name].append(field)
        # self.check_local_date_period_completeness(file_to_fields, start_date, end_date) todo 后面实盘开启
        # 加载和处理数据
        raw_wide_dfs = {}  # 装 宽化的df
        raw_long_dfs = {}  # 原生的 从本地拿到的 key :文件，value：df（所有列！）
        for logical_name, columns_to_need_load in file_to_fields.items():
            try:
                file_path = self.data_path / logical_name

                # 检查文件中实际存在的字段
                columns_to_need_load = self.fix_names_for_origin(columns_to_need_load, logical_name)
                available_columns = self.available_columns_by_file.get(logical_name)
                if not available_columns:
                    raise ValueError(f"数据源元数据缺失: logical_name={logical_name}")
                missing_columns = sorted(set(columns_to_need_load) - available_columns)
                if missing_columns:
                    raise ValueError(
                        f"数据源缺少必需字段: logical_name={logical_name}, "
                        f"missing={missing_columns}, available={sorted(available_columns)}"
                    )
                if 'ts_code' not in available_columns:
                    raise ValueError(f"数据源缺少面板主键: logical_name={logical_name}, field=ts_code")
                columns_can_read = sorted(
                    set(columns_to_need_load) |
                    (set(base_fields) & available_columns)
                )

                # 加载数据
                long_df = pd.read_parquet(
                    file_path,
                    columns=columns_can_read
                )

                # 时间筛选
                long_df = self.extract_during_period(long_df, logical_name, buffer_start_date, end_date)

                # 股票池筛选
                if ts_codes is not None and 'ts_code' in long_df.columns:
                    long_df = long_df[long_df['ts_code'].isin(ts_codes)]

                raw_long_dfs[logical_name] = long_df

            except Exception as e:
                logger.error(f"处理数据集 {logical_name} 失败: {e}")
                raise ValueError(f"处理数据集 {logical_name} 失败: {e}")

        # --- 3. 将所有数据处理成统一的面板宽表格式 ---
        trading_dates = self.get_trading_dates(buffer_start_date, end_date)
        for field in sorted(fields):
            logical_name = self.field_map.get(field)
            if not logical_name or logical_name not in raw_long_dfs:
                raise ValueError(f"未找到或加载失败: 字段 '{field}' 的数据源 '{logical_name}'")
            source_df = raw_long_dfs[logical_name]
            if 'trade_date' in source_df.columns:
                # a) 对于本身就是每日更新的面板数据
                df = source_df.copy()
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].isin(trading_dates)]

                # 明确地定义重复的键
                duplicate_keys = ['trade_date', 'ts_code']

                # 在转换前，先使用 drop_duplicates 进行清洗
                # keep='last' 是一个重要的选择：我们假定文件末尾的记录是最新的、最准确的
                unique_long_df = df.drop_duplicates(subset=duplicate_keys, keep='last')

                # 确认没有重复项后，可以安全地进行转换
                #  此时可以直接使用 pivot()，它比 pivot_table() 略快，且能再次验证唯一性
                field_for_origin =   self.fix_name_for_origin(field, logical_name)
                wide_df = unique_long_df.pivot(index='trade_date', columns='ts_code', values=field_for_origin)
            else:
                # b) 对于需要“广播”到每日的静态属性数据 (如name, industry)
                logger.info(f"  正在将静态字段 '{field}' 广播到每日面板...")
                static_series = source_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')[field]

                # #  方式（1）：直接广播
                # for ts_code in wide_df.columns:
                #     # 构造空 DataFrame，行是日期，列是股票代码
                #     wide_df = pd.DataFrame(index=pd.DatetimeIndex(trading_dates), columns=static_series.index)
                #     wide_df[ts_code] = static_series[ts_code]
                # 方式2 更高效 类似铺砖
                ##
                # np.tile(A, (M, 1)) = 把一行数组 A，重复 M 行，不重复列」
                #
                # 也就是说：
                #
                # M 控制的是“你有多少行”（行方向“铺砖”）
                #
                # 1 表示“列不要扩展”（只保留原来的股票维度）#
                wide_df = pd.DataFrame(
                    data=np.tile(static_series.values, (len(trading_dates), 1)),  # 使用numpy.tile高效复制数据
                    index=trading_dates,
                    columns=static_series.index
                )
            raw_wide_dfs[field] = wide_df

        # 对齐数据
        aligned_data = self._align_dataframes(raw_wide_dfs)

        # aligned_data = self.rename_for_safe(aligned_data)
        return aligned_data #close ——raw 已经 hfq 通过聚宽 比对 数据完全对上

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
        primary_candidates = ['close_hfq', 'open_hfq',
                              'low_hfq']  # primary_candidates = ['close_raw', 'close', 'open_raw', 'open', 'high_raw', 'low_raw']
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
        if field.endswith('_hfq') & logical_name.endswith('_hfq'):
            return field.replace('_hfq', '')
        if field.endswith('_raw') & (logical_name == 'daily'):
            return field.replace('_raw', '')
        return field

    def fix_names_for_origin(self, columns_to_need_load, logical_name):
       return  [self.fix_name_for_origin(column,logical_name) for column in columns_to_need_load]

