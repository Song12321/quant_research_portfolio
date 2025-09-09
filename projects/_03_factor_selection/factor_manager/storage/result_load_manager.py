import json
from pathlib import Path

import  pandas as pd

from quant_lib.evaluation.evaluation import calculate_forward_returns_tradable_o2o


#ic_series_processed_60d.parquet
# def build_targer_name(core_eveluation_type,is_raw_factor, forward_period):
#
#     if core_eveluation_type == 'ic':
#         if is_raw_factor:
#             return f'ic_raw_{forward_period}d'
#         else:
#             return f'ic_processed_{forward_period}d'
#     elif core_eveluation_type == 'tmb':
#         if is_raw_factor:
#             return f'tmb_raw_{forward_period}d'
#         else:
#             return f'tmb_processed_{forward_period}d'
#     elif core_eveluation_type == 'monotonicity':
#     pass


def load_summary_stats(param):
    d1 = json.load(open(param, 'r', encoding='utf-8'))
    return d1
def load_ic_stats(json:json=None,is_raw_factor:bool=False):
    subfix='_processed'
    if is_raw_factor:
        subfix='_raw'
    ic_stas =json.get(f'ic_analysis{subfix}')
    return ic_stas
class ResultLoadManager:
    def __init__(self, calcu_return_type='o2o',pool_index:str=None, s:str=None,e:str=None,version:str=None, is_raw_factor: bool=False):
        if version is None :
            raise ValueError('请指定版本')
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result")
        self.calcu_type = calcu_return_type
        self.version = version
        self.pool_index = pool_index
        self.start_date = s
        self.end_date = e

        self.is_raw_factor = is_raw_factor
    #严禁使用！这是整体的ic。整个周期的（严重未来寒函数
    # def get_ic_stats_from_local(self, stock_pool_index, factor_name):
    #     path = self.get_factor_self_path(stock_pool_index, factor_name)
    #     ret = load_summary_stats(path / "summary_stats.json")
    #     ic_stas = load_ic_stats(ret, self.is_raw_factor)
    #     return ic_stas

    def get_factor_data(self, factor_name, pool_index=None,start_date=None, end_date=None):
        if self.is_raw_factor:
            raise ValueError('暂不支持raw因子数据')
        pool_index =  self.pool_index if pool_index is None else pool_index
        factor_self_path = self.get_factor_self_path(pool_index, factor_name)
        df = pd.read_parquet(factor_self_path / "processed_factor.parquet")
        df.index = pd.to_datetime(df.index)
        if start_date is None and end_date is None :
            return df
        return  df.loc[start_date:end_date]

    def get_o2o_return_data(self, stock_pool_index, start_date, end_date, period_days):
        path = self.main_work_path / stock_pool_index /'open_hfq'/ self.version / 'open_hfq.parquet'
        df =pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        returns = calculate_forward_returns_tradable_o2o(period=period_days, open_df=df)
        #过滤时间"
        returns = returns.loc[start_date:end_date]

        return returns

    def get_close_hfq_data(self, stock_pool_index, start_date, end_date):
        path = self.main_work_path / stock_pool_index / 'close_hfq' / self.version / 'close_hfq.parquet'
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # 过滤时间"
        returns = df.loc[start_date:end_date]

        return returns
    #抽象一下
    def get_price_data_by_type(self, stock_pool_index, start_date, end_date, price_type=None):
        if price_type is None:
            raise ValueError('请指定价格类型')
        path = self.main_work_path / stock_pool_index / price_type / self.version / f'{price_type}.parquet'
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # 过滤时间"
        returns = df.loc[start_date:end_date]
        return returns


    def get_factor_self_path(self, stock_pool_index, factor_name):
        return self.main_work_path / stock_pool_index / factor_name / self.calcu_type / self.version

    def get_ic_series_by_period(self,stock_pool_index, factor_name, period_days:int):
        factor_path = self.get_factor_self_path(stock_pool_index, factor_name)
        df = pd.read_parquet(factor_path/f"ic_series_processed_{period_days}d.parquet")
        #转成series
        return df.squeeze()
    def get_summary_stats(self, factor_name):
        factor_path = self.get_factor_self_path(self.pool_index, factor_name)
        json = load_summary_stats(factor_path/'summary_stats.json')
        return json


if __name__ == '__main__':
    manager = ResultLoadManager(version='20230601_20240710')
    # manager.get_factor_data('volatility_90d','000906', '20190328', '20231231')
    manager.get_ic_series_by_period('000300', 'vwap_deviation_20d', 5)
