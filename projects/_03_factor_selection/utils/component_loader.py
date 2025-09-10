# --- file: utils/component_loader.py ---
from typing import Union

import pandas as pd


class IndexComponentLoader:
    """
    一个专门用于加载、处理和提供指数历史成分股数据的工具类。

    它会一次性加载所有配置的Excel数据，并在内存中构建一个每日成分股的“快照”缓存，
    以极高的效率为回测提供每日查询服务。
    """

    def __init__(self, excel_files_config: Union[dict,None]=None):
        """
        初始化加载器。
        :param excel_files_config: 一个字典，key是指数代码(str)，value是Excel文件路径(str)。
                                 例如: {'000300': 'path/to/hs300.xlsx', '000905': 'path/to/csi500.xlsx'}
        """
        print("--- 初始化 IndexComponentLoader ---")
        if excel_files_config is None:
            excel_files_config = {
                '000905':"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\my_file\index_files\指数成分(中证500_000905).xlsx",
                '000300':"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\my_file\index_files\指数成分(沪深300_000300).xlsx"
            }
        self.config = excel_files_config
        self._raw_components_df = self._load_and_prepare_all_data()
        self._daily_members_cache = {}  # 缓存每日成分股，避免重复计算
        print("--- IndexComponentLoader 初始化完成 ---")

    def _format_stock_code(self, row: pd.Series) -> str:
        """根据交易市场，将纯数字的股票代码转换为您系统所需的带后缀的格式。"""
        code = str(row['成分券代码']).zfill(6)
        market = row['交易市场']
        if market == 'XSHE':  # 深圳证券交易所
            return f"{code}.SZ"
        if market == 'XSHG':  # 上海证券交易所
            return f"{code}.SH"
        return code

    def _load_and_prepare_all_data(self) -> pd.DataFrame:
        """加载所有配置文件中的Excel，合并并处理成标准格式。"""
        all_dfs = []
        for index_code, file_path in self.config.items():
            try:
                print(f"  > 正在加载指数 '{index_code}' 的成分股数据: {file_path}")
                df = pd.read_excel(file_path)
                df['stock_code'] = df.apply(self._format_stock_code, axis=1)
                df['in_date'] = pd.to_datetime(df['纳入日期'], errors='coerce')
                df['out_date'] = pd.to_datetime(df['剔除日期'], errors='coerce')
                df['index_code'] = index_code  # 标记每条记录来源于哪个指数
                df.dropna(subset=['in_date'], inplace=True)
                all_dfs.append(df[['stock_code', 'in_date', 'out_date', 'index_code']])
            except Exception as e:
                print(f"【错误】加载文件 {file_path} 失败: {e}")

        if not all_dfs:
            raise ValueError("未能成功加载任何成分股数据！请检查配置文件路径。")

        return pd.concat(all_dfs, ignore_index=True)

    def get_members_on_date(self, target_date: pd.Timestamp, index_codes: list) -> set:
        """
        【核心查询函数】获取一个或多个指数在指定日期的合并成分股列表。

        :param target_date: 目标查询日期 (pandas.Timestamp)
        :param index_codes: 一个包含指数代码字符串的列表 (e.g., ['000300', '000905'])
        :return: 一个包含所有成分股代码的集合 (set)，查询效率高。
        """
        # 使用日期和指数列表元组作为缓存的键
        cache_key = (target_date, tuple(sorted(index_codes)))
        if cache_key in self._daily_members_cache:
            return self._daily_members_cache[cache_key]

        # 筛选出与目标指数相关的所有历史记录
        target_df = self._raw_components_df[self._raw_components_df['index_code'].isin(index_codes)]

        # 核心筛选逻辑: 找到在 target_date 当天有效的所有成分股
        mask = (target_df['in_date'] <= target_date) & \
               (target_df['out_date'].isnull() | (target_df['out_date'] > target_date))

        members_set = set(target_df[mask]['stock_code'])

        # 将结果存入缓存
        self._daily_members_cache[cache_key] = members_set
        return members_set