import alphalens as al
import pandas as pd

from data.local_data_load import load_income_df, load_price_hfq
from alphalens.utils import  get_clean_factor_and_forward_returns
def alphalens_ic(factor_wide_df, price_df) -> None:
    alphalens_factor = prepare_alphalens_factor_data(factor_wide_df)
    ret = get_clean_factor_and_forward_returns(factor=alphalens_factor,prices=price_df)

    ic_cs = al.performance.factor_information_coefficient(ret)
    print(1)

    # ... 后续可以基于 factor_data 进行更多分析 ...
def prepare_alphalens_factor_data( factor_wide_df: pd.DataFrame) -> pd.Series:
    """
    将“宽表”因子数据 (index=日期, columns=股票)
    转换为 Alphalens 所需的“长表”格式 (MultiIndex Series)。
    """
    # 确保日期索引是 datetime 类型
    factor_wide_df = factor_wide_df.copy()
    factor_wide_df.index = pd.to_datetime(factor_wide_df.index)

    # 转成长表 Series
    factor_long = factor_wide_df.stack()
    factor_long.index.names = ['date', 'asset']

    return factor_long

if __name__ == '__main__':
    price_df = load_price_hfq('open','2023-01-01','2023-10-01')

    factor_wide_df = load_price_hfq('close','2023-01-01','2023-10-01')

    alphalens_ic(factor_wide_df, price_df)