import numpy as np
import pandas as pd

from quant_lib.config.constant_config import get_market_data_path
from quant_lib.tushare.data.downloader import download_suspend_d
from quant_lib.tushare.tushare_client import TushareClient


# daily_hfq 有问题
def get_fields_map():
    result = []
    paths = [ 'daily', 'daily_basic', 'daily_hfq',  'index_weights', 'margin_detail',
             'stk_limit',

             'cashflow.parquet',
             'income.parquet',
             'index_daily.parquet',
             'namechange.parquet',
             'stock_basic.parquet',
             'fina_indicator.parquet',
             'suspend_d.parquet',
             'trade_cal.parquet',
             'balancesheet.parquet',
             ]

    for path in paths:
        orin_df = pd.read_parquet(get_market_data_path(path))
        print(f'logic_name:{path}')
        if 'trade_date' in orin_df.columns:
            orin_df['trade_date'] = pd.to_datetime(orin_df['trade_date'])
            df = orin_df.copy(deep=True)
            df = df[(df['trade_date'] >= pd.to_datetime('20250611')) & (df['ts_code'] == '000012.SZ')]
            df['trade_date_and_tscode_重复次数'] = df.groupby(['trade_date','ts_code'])['trade_date'].transform('count')
            dup_rows = df[df['trade_date_and_tscode_重复次数'] > 1].sort_values(by='ts_code')
            print("结果如下----")
            print(dup_rows)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result


# daily_hfq 有问题
def dup_check():
    result = []
    # call_pro_tushare_api('index_daily', ts_code='000300.SH', start_date='20100101', end_date='20250711')
    # call_ts_tushare_api('pro_bar',**{'adj': 'hfq', 'asset': 'E','ts_code':'000002.SZ'})
    # api_df  = call_ts_tushare_api("pro_bar", ts_code='000012.SZ', start_date='20180601', end_date='20250711')
    # call_ts_tushare_api('pro_bar', adj =  'hfq', asset='E',ts_code='000002.SZ')

    paths = [
        'cashflow.parquet',
        'income.parquet'
    ]
    for path in paths:
        orin_df = pd.read_parquet(get_market_data_path(path))
        print(f'logic_name:{path}')
        if 'trade_date' in orin_df.columns:
            orin_df['trade_date'] = pd.to_datetime(orin_df['trade_date'])
            df = orin_df.copy(deep=True)
            df = df[(df['trade_date'] >= pd.to_datetime('20250611')) & (df['ts_code'] == '000012.SZ')]
            df['trade_date_and_tscode_重复次数'] = df.groupby(['trade_date','ts_code'])['trade_date'].transform('count')
            dup_rows = df[df['trade_date_and_tscode_重复次数'] > 1].sort_values(by='ts_code')
            print("结果如下----")
            print(dup_rows)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result



# daily_hfq 有问题
def dup_check_report_type():
    result = []
    # call_pro_tushare_api('index_daily', ts_code='000300.SH', start_date='20100101', end_date='20250711')
    # call_ts_tushare_api('pro_bar',**{'adj': 'hfq', 'asset': 'E','ts_code':'000002.SZ'})
    # api_df  = call_ts_tushare_api("pro_bar", ts_code='000012.SZ', start_date='20180601', end_date='20250711')
    # call_ts_tushare_api('pro_bar', adj =  'hfq', asset='E',ts_code='000002.SZ')

    paths = [
        'income.parquet',
        'index_daily.parquet',
    ]
    for path in paths:
        orin_df = pd.read_parquet(get_market_data_path(path))
        print(f'logic_name:{path}')
        if 'end_date' in orin_df.columns:
            orin_df['end_date'] = pd.to_datetime(orin_df['end_date'])
            df = orin_df.copy(deep=True)
            df = df[df['report_type'] !='1' ]
            print("结果如下----")
            print(df)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result

def fq():
    import pandas as pd

    # ▼▼▼▼▼ 请在这里修改为你自己的配置 ▼▼▼▼▼
    # 你的“不复权”日线行情文件的真实路径
    RAW_DAILY_FILE_PATH = get_market_data_path('daily')

    # 你选择的股票和它的一个历史除权日
    STOCK_TO_CHECK = '600519.SH'  # 以贵州茅台为例
    EX_DATE = '2023-06-30'  # 茅台2022年度分红的除权日
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # --- 执行检验 ---
    print("--- 正在对最底层‘不复权’数据源进行终极审查 ---")
    try:
        df = pd.read_parquet(RAW_DAILY_FILE_PATH)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        stock_df = df[df['ts_code'] == STOCK_TO_CHECK].set_index('trade_date').sort_index()

        ex_date_dt = pd.to_datetime(EX_DATE)

        # 获取除权日前后几天的价格
        price_slice = stock_df.loc[ex_date_dt - pd.Timedelta(days=5): ex_date_dt + pd.Timedelta(days=5)]

        print(f"\n正在检查 {STOCK_TO_CHECK} 在除权日 {EX_DATE} 前后的【不复权】价格：")
        print(price_slice[['close']])  # 我们只关心 close_raw

    except Exception as e:
        print(f"\n读取或处理文件时发生错误: {e}")
def compare_df_rows(df, index1, index2):
    """
    比较 df 中 index1 和 index2 两行，返回它们不一致的列名列表
    """
    row1 = df.loc[index1]
    row2 = df.loc[index2]

    # 比较所有列，找出值不同的列
    diff_cols = [
        col for col in df.columns
        if not (pd.isna(row1[col]) and pd.isna(row2[col])) and row1[col] != row2[col]
    ]
    print(f"Index {index1} 与 Index {index2} 不同的列：")
    print(diff_cols)
    return diff_cols


if __name__ == '__main__':
    fq()
    stock_basic = TushareClient.get_pro().stock_basic(
        list_status='L,D,P',
        fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
    )
    all_ts_codes = stock_basic['ts_code'].unique().tolist()

    # 已有数据
    old_date_df = pd.read_parquet(get_market_data_path('suspend_d.parquet'))

    already_ts_codes = old_date_df['ts_code'].unique()


    # 差集
    diff_ts_codes = np.setdiff1d(all_ts_codes, already_ts_codes)
    info = pd.read_parquet(get_market_data_path('stock_basic.parquet'))





    index_daily = pd.read_parquet(get_market_data_path('index_daily.parquet'))
    index_daily['trade_date'] = pd.to_datetime(index_daily['trade_date'])
    daily_df = index_daily[index_daily['trade_date'] >= pd.to_datetime('2025-04-11')]
    # adj_df = pd.read_parquet(get_market_data_path('adj_factor'))
    # adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
    # adj_df = adj_df.sort_values(by=['ts_code','trade_date'])
    # adj_df = adj_df[adj_df['trade_date'] >= pd.to_datetime('2025-04-11')]
    # adj_df = adj_df.set_index(['ts_code', 'trade_date'])

    daily_df = pd.read_parquet(get_market_data_path('daily'))
    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])
    daily_df = daily_df[daily_df['trade_date'] >= pd.to_datetime('2025-04-11')]
    daily_df = daily_df.set_index(['ts_code', 'trade_date'])


    daily_basic_df = pd.read_parquet(get_market_data_path('daily_basic'))
    daily_basic_df['trade_date'] = pd.to_datetime(daily_basic_df['trade_date'])
    daily_basic_df = daily_basic_df[daily_basic_df['trade_date'] >= pd.to_datetime('2025-04-11')]
    daily_basic_df = daily_basic_df.set_index(['ts_code', 'trade_date'])


    hfq_df = pd.read_parquet(get_market_data_path('daily_hfq'))
    hfq_df['trade_date'] = pd.to_datetime(hfq_df['trade_date'])
    hfq_df = hfq_df[hfq_df['trade_date'] >= pd.to_datetime('2025-04-11')]
    hfq_df = hfq_df[hfq_df['ts_code'].isin(['000001.SZ','000002.SZ'])]


    # 初始化 DataFrame
    com_df = pd.DataFrame(index=daily_df.index)

    # for col in ['close','vol','amount']:
    #     # com_df[col] = daily_df[col] * adj_df['adj_factor']

    # 对比手动复权 vs 已复权
    print(com_df['close','pct_chg'].head())
    print(hfq_df['close','pct_chg'].head())







    # imcome_df['f_ann_date'] = pd.to_datetime(imcome_df['f_ann_date'])
    # imcome_df = imcome_df[imcome_df['f_ann_date'] != imcome_df['ann_date']]
    # dup_check_report_type()
    # df['end_date'] = pd.to_datetime(df['end_date'])
    # df = df[df['end_date'] >= pd.to_datetime('2025')]
    # print(list(df.columns))

    ##
    #
    #
    # D:\lqs\codeAbout\py\env\vector_bt_env\Scripts\python.exe D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\data\test.py
    # [{'name': 'adj_factor', 'fields': Index(['ts_code', 'trade_date', 'adj_factor', 'year'], dtype='object')}, {'name': 'daily', 'fields': Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
    #        'change', 'pct_chg', 'vol', 'amount', 'year'],
    #       dtype='object')}, {'name': 'daily_basic', 'fields': Index(['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
    #        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
    #        'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
    #        'circ_mv', 'year'],
    #       dtype='object')}, {'name': 'daily_hfq', 'fields': Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
    #        'change', 'pct_chg', 'vol', 'amount', 'year'],
    #       dtype='object')}, {'name': 'fina_indicator.parquet', 'fields': Index(['ts_code', 'ann_date', 'end_date', 'eps', 'dt_eps', 'total_revenue_ps',
    #        'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps', 'undist_profit_ps',
    #        ...
    #        'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'q_sales_yoy',
    #        'q_op_qoq', 'equity_yoy', 'update_flag', 'year'],
    #       dtype='object', length=110)}, {'name': 'margin_detail', 'fields': Index(['trade_date', 'ts_code', 'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche',
    #        'rqchl', 'rqmcl', 'rzrqye', 'year'],
    #       dtype='object')}, {'name': 'stk_limit', 'fields': Index(['trade_date', 'ts_code', 'up_limit', 'down_limit', 'year'], dtype='object')}, {'name': 'stock_basic.parquet', 'fields': Index(['ts_code', 'symbol', 'name', 'area', 'industry', 'cnspell', 'market',
    #        'list_date', 'act_name', 'act_ent_type'],
    #       dtype='object')}, {'name': 'trade_cal.parquet', 'fields': Index(['exchange', 'cal_date', 'is_open', 'pretrade_date'], dtype='object')}]
    # 总数187
    #
    # Process finished with exit code 0#
