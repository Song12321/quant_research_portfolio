# 文件名: downloader.py
# 作用：一个健壮的数据下载器，用于从Tushare获取所有需要的A股核心数据，
#      并以高效的Parquet格式保存在本地，为后续的量化研究做准备。
#      本脚本已处理好API的频率和单次条数限制。
import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.local_data_load import load_trading_lists
from quant_lib.config.constant_config import MARKET_DATA_ROOT, get_market_data_path
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.tushare_client import TushareClient
from quant_lib.utils.report_date import get_reporting_period_day_list

# --- 2. 全局配置 ---
START_YEAR = 2018
END_YEAR = datetime.now().year
BATCH_SIZE = 20
INDEX_CODES_for_tushare = {
    "HS300": "000300",     # 沪深300
    "ZZ500": "000905",     # 中证500
    "ZZ800": "000906",     # 中证800
    "ZZ1000": "000852",    # 中证1000
    'ZZ_ALL': "000985",    #中证全指
}

# --- 3. 辅助函数 ---
def get_year_end(year):
    if year == END_YEAR:
        beijing_now = datetime.utcnow() + timedelta(hours=8)
        beijing_yesterday = beijing_now - timedelta(days=1)
        # return beijing_yesterday.strftime('%Y%m%d')
        return '20250711'
    return f'{year}1231'

def get_all_industry_l_n_codes(category:str='一级行业指数'):
    ret = call_pro_tushare_api('index_basic',market='SW')
    ret = ret[ret['category']== category]
    return ret['ts_code']
def download_sw_basic_info():
    ret = call_pro_tushare_api('index_basic',market='SW')
    ret.to_parquet(get_market_data_path('sw_basic_info.parquet'))
    print("保存完成")

def download_index_weights():
    """下载指数成分股历史数据"""
    print("\n===== 开始下载指数成分股数据 =====")

    # 常用指数列表
    INDEX_CODES_for_tushare = [
        '000300.SH',  # 沪深300
        '000905.SH',  # 中证500
        '000906.SH',  # 中证800
        '000852.SH',  # 中证1000
        '000985.SH'#所有大A
    ]

    for index_code in INDEX_CODES_for_tushare:
        index_path = get_market_data_path('index_weights') / f"{index_code.replace('.', '_')}"

        for year in range(START_YEAR, END_YEAR + 1):
            year_path = index_path / f"year={year}"

            if not year_path.exists():
                print(f"--- 正在下载 {year} 年的 {index_code} 成分股数据 ---")

                year_start = f'{year}0101'
                year_end = get_year_end(year)

                # 获取该年度所有交易日
                trade_dates = load_trading_lists(year_start, year_end)

                all_weights = []

                # 逐日获取成分股
                for i, trade_date in enumerate(trade_dates):
                    print(f"    处理交易日 {trade_date} ({i + 1}/{len(trade_dates)})...")

                    try:
                        df_weight = call_pro_tushare_api(
                            'index_weight',
                            index_code=index_code,
                            trade_date=trade_date
                            # fields='index_code,con_code,trade_date,weight'
                        )

                        if not df_weight.empty:
                            all_weights.append(df_weight)
                        else:
                            print(f"      {trade_date} 无成分股数据")

                    except Exception as e:
                        print(f"      {trade_date} 获取失败: {e}")
                        continue

                # 保存年度数据
                if all_weights:
                    final_df = pd.concat(all_weights, ignore_index=True)
                    final_df= final_df.drop_duplicates(inplace=False)

                    year_path.mkdir(parents=True, exist_ok=True)
                    final_df.to_parquet(year_path / 'data.parquet')
                    print(f"成功保存 {year} 年的 {index_code} 成分股数据")
                else:
                    print(f"未获取到 {year} 年的 {index_code} 成分股数据")
            else:
                print(f"{year} 年的 {index_code} 成分股数据已存在，跳过下载")


def delete_suffix_index():
    path = get_market_data_path('index_daily.parquet')
    df = pd.read_parquet(path)
    print(df['ts_code'].unique().tolist())
    df['ts_code'] = df['ts_code'].str.split('.').str[0]
    df.to_parquet(path)

def download_index_daily_info():
    """获取上证指数日线数据"""
    print("\n===== 开始下载指数成分股数据 =====")
    all = []
    for index_code_key,value in INDEX_CODES_for_tushare.items():
        final_df = call_pro_tushare_api('index_daily', ts_code=value, start_date='20100101', end_date='20250711')
        all.append(final_df)
    path = get_market_data_path('index_daily.parquet')
    # 先合并DataFrame
    concatenated_df = pd.concat(all, ignore_index=True)
    ##处理！要把 后缀.sh 去掉 000300.SH -》000300  切记！
    # 再保存
    concatenated_df.to_parquet(path)
    print(f"✓ 所有指数数据已合并并保存至: {path}")
#2025 07 31调用，所以也只有截止到0731的数据！
def download_suspend_d():
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

    # 下载新增数据
    all_data_list = []
    for i, ts_code in enumerate(diff_ts_codes):
        print(f"开始处理第{i + 1}/{len(diff_ts_codes)}股票...")
        data = call_pro_tushare_api('suspend_d', ts_code=ts_code)
        if not data.empty:
            all_data_list.append(data)

    # 合并新旧数据
    if all_data_list:
        new_data_df = pd.concat(all_data_list, ignore_index=True)
        final_df = pd.concat([old_date_df, new_data_df], ignore_index=True)
    else:
        final_df = old_date_df.copy()

    # 去重
    final_df.drop_duplicates(inplace=True)

    # 保存前删除原文件（如果存在）
    file_path = get_market_data_path('suspend_d.parquet')
    if file_path.exists():
        os.remove(file_path)

    # 保存
    final_df.to_parquet(file_path)
    print("download_suspend_d 保存成功")


def download_stock_info(stock_basic_path):
    if not stock_basic_path.exists():
        print("--- 正在下载股票基本信息 ---")
        stock_basic = TushareClient.get_pro().stock_basic(list_status='L,D,P',
                                                          fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type')
        if not stock_basic.empty:
            stock_basic.to_parquet(stock_basic_path)
    else:
        print("股票基本信息已存在，跳过下载。")

#已有数据'20100101','20250711'
def download_cashflow():
    basic_path = get_market_data_path('cashflow.parquet')

    if not basic_path.exists():
        print("--- 正在下载现金流信息 ---")
        #获取报告期日list
        reporting_period_day_list = get_reporting_period_day_list('20100101','20250711')
        all_data_list = []
        for reporting_period_day in reporting_period_day_list:
            df = call_pro_tushare_api('cashflow_vip', period=reporting_period_day)
            all_data_list.append(df)

        # 合并所有季度数据
        final_df = pd.concat(all_data_list, ignore_index=True)
        # 关键：优先保留 update_flag == 1 的最新记录
        final_df = final_df.sort_values(['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False])
        final_df = final_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')
        final_df.to_parquet(basic_path)
        print('现金流保存完毕')
    else:
        print("现金流已存在，跳过下载。")


#已有数据'20100101','20250711'
def download_income():
    basic_path = get_market_data_path('income.parquet')

    if not basic_path.exists():
        print("--- 正在下载利润信息 ---")
        #获取报告期日list
        reporting_period_day_list = get_reporting_period_day_list('20100101','20250711')
        all_data_list = []
        for reporting_period_day in reporting_period_day_list:
            df = call_pro_tushare_api('income_vip', period=reporting_period_day)
            all_data_list.append(df)

        # 合并所有季度数据
        final_df = pd.concat(all_data_list, ignore_index=True)
        # 关键：优先保留 update_flag == 1 的最新记录
        final_df = final_df.sort_values(['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False])
        final_df = final_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')# todo remind 注意
        final_df.to_parquet(basic_path)
        print('利润表保存完毕')
    else:
        print("利润表已存在，跳过下载。")


#已有数据'20100101','20250711'
def download_fina_indicator():
    basic_path = get_market_data_path('fina_indicator.parquet')

    if not basic_path.exists():
        print("--- 正在下载财务指标数据 ---")
        #获取报告期日list
        reporting_period_day_list = get_reporting_period_day_list('20100101','20250711')
        all_data_list = []
        for index,reporting_period_day in enumerate(reporting_period_day_list):
            print(f"处理第{index+1}/{len(reporting_period_day_list)}个季度-财务指标数据")
            df = call_pro_tushare_api('fina_indicator_vip', period=reporting_period_day)
            #只需要当前period_day的
            df['end_date'] = pd.to_datetime(df['end_date'])
            df = df[df['end_date'] == pd.to_datetime(reporting_period_day)]
            all_data_list.append(df)

        # 合并所有季度数据
        final_df = pd.concat(all_data_list, ignore_index=True)
        # 关键：优先保留 update_flag == 1 的最新记录
        final_df = final_df.sort_values(['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False])
        final_df = final_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')# todo remind 注意
        final_df.to_parquet(basic_path)
        print('财务指标数据-保存完毕')
    else:
        print("财务指标数据表已存在，跳过下载。")

#已有数据'20100101','20250711'
def download_balancesheet(name='资产负债表'):
    basic_path = get_market_data_path('balancesheet.parquet')

    if not basic_path.exists():
        print(f"--- 正在下载{name}信息 ---")
        #获取报告期日list
        reporting_period_day_list = get_reporting_period_day_list('20100101','20250711')
        all_data_list = []
        for index,reporting_period_day in enumerate(reporting_period_day_list):
            print(f"处理第{index+1}/{len(reporting_period_day_list)}个季度-{name}数据")
            df = call_pro_tushare_api('balancesheet_vip', period=reporting_period_day)
            df['end_date'] = pd.to_datetime(df['end_date'])
            df = df[df['end_date'] == pd.to_datetime(reporting_period_day)]
            all_data_list.append(df)

        # 合并所有季度数据
        final_df = pd.concat(all_data_list, ignore_index=True)
        # 关键：优先保留 update_flag == 1 的最新记录
        final_df = final_df.sort_values(['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False])
        final_df = final_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')# todo remind 注意
        final_df.to_parquet(basic_path)
        print(f'{name}保存完毕')
    else:
        print(f"{name}已存在，跳过下载。")

def download_sw_daily(name='sw日线行情'):
    basic_path = get_market_data_path('sw_daily.parquet')
    if not basic_path.exists():
        print(f"--- 正在下载{name}信息 ---")
        l1_codes = get_all_industry_l_n_codes()
        # 2. 循环获取每个行业的日线行情
        all_industry_daily = []
        for code in l1_codes:  #
            df = call_pro_tushare_api('sw_daily', ts_code=code, start_date='20100711', end_date='20250711')
            all_industry_daily.append(df)

        industry_daily_df = pd.concat(all_industry_daily)

        industry_daily_df['trade_date'] = pd.to_datetime(industry_daily_df['trade_date'])
        industry_daily_df = industry_daily_df.sort_values(['ts_code', 'trade_date'])
        industry_daily_df = industry_daily_df.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=False)

        # 4. 格式化输出，方便后续合并
        # **注意：这里的行业代码格式是 801010.SI，需要处理成 801010 以便和你的映射工具对齐**
        # industry_daily_df['l1_code'] = industry_daily_df['ts_code'].str.split('.').str[0]
        industry_daily_df.to_parquet(basic_path)
        print(f'{name}保存完毕')
    else:
        print(f"{name}已存在，跳过下载。")
def get_all_stock_basic_from_api():
    stock_list = call_pro_tushare_api("stock_basic", list_status='L,D,P', fields='ts_code')['ts_code'].tolist()
    return  stock_list
def get_all_stock_basic_from_local():
    stock_list = pd.read_parquet(get_market_data_path('stock_basic.parquet'))['ts_code'].tolist()
    return  stock_list

#目前6000股票/60
def download_industry_record():
    path = get_market_data_path('industry_record.parquet')
    if  path.exists():
        raise ValueError("已存在 industry_record，无需下载")
    """
    一次性拉取并构建包含所有股票历史行业归属的主数据表。
    """
    print("--- 开始构建行业历史主数据表 ---")
    # 1. 获取所有A股列表 (作为查询目标)
    all_ts_codes = get_all_stock_basic_from_api()
    suspend_d_df  = pd.read_parquet(get_market_data_path('suspend_d.parquet'))
    namechange  = pd.read_parquet(get_market_data_path('namechange.parquet'))

    # 2. 循环获取每只股票的行业隶属历史
    all_members = []
    for ts_code in tqdm(all_ts_codes, desc="获取股票行业隶属历史"):
        try:
            # is_new='N' 获取所有历史记录，而不仅仅是最新记录
            hisotry_df = call_pro_tushare_api('index_member_all', ts_code=ts_code, is_new='N')
            new_df = call_pro_tushare_api('index_member_all', ts_code=ts_code, is_new='Y')
            dfs = [df for df in [hisotry_df, new_df] if df is not None and not df.empty]
            if dfs:
                entiry_record_for_one_stock_df = pd.concat(dfs, ignore_index=True)
            else:
               continue
            all_members.append(entiry_record_for_one_stock_df)
            print(f"  成功获取 {ts_code} 的行业历史...")
        except Exception as e:
            raise ValueError(f"  获取 {ts_code} 失败: {e}")

    industry_history_df = pd.concat(all_members, ignore_index=True)

    # 4. 【核心处理】执行必要的后处理，生成干净数据
    # 转换日期格式
    industry_history_df['in_date'] = pd.to_datetime(industry_history_df['in_date'], format='%Y%m%d')
    industry_history_df['out_date'] = pd.to_datetime(industry_history_df['out_date'], format='%Y%m%d')
    # 使用一个遥远的未来日期填充NaT，便于查询
    # industry_history_df['out_date'] = pd.to_datetime(industry_history_df['out_date'], format='%Y%m%d').fillna(pd.Timestamp('2099-12-31'))

    # 去重并排序，确保数据唯一性和有序性
    # master_df = industry_history_df.drop_duplicates(subset=['ts_code', 'industry_code', 'in_date']).sort_values(
    #     by=['ts_code', 'in_date'])
    master_df = industry_history_df.sort_values(
        by=['ts_code', 'in_date'])

    # 4. 保存到本地
    master_df.to_parquet(path)
    print("done 获取股票行业隶属历史")

    print("--- 行业历史主数据表构建完成并已保存到本地 ---")
def download_stock_change_name_details():
    # 在 downloader.py 的“下载配套数据”部分，增加以下逻辑

    namechange_path = get_market_data_path('namechange.parquet')
    if not namechange_path.exists():
        print("--- 正在下载股票名称变更历史 ---")
        # Tushare的namechange接口可能需要循环获取，因为它有单次返回限制
        # 一个稳健的做法是获取所有股票列表，然后逐个调用
        stock_list = call_pro_tushare_api("stock_basic", list_status='L,D,P', fields='ts_code')['ts_code'].tolist()
        all_changes = []
        startIdx = 0
        for stock in stock_list:
            print(f"开始处理第{startIdx + 1}/{len(stock_list)}只股票")
            df = call_pro_tushare_api("namechange", ts_code=stock)
            df.drop_duplicates(inplace=True)
            all_changes.append(df)
            startIdx += 1

        namechange_df = pd.concat(all_changes)
        if not namechange_df.empty:
            namechange_df.to_parquet(namechange_path)
            print(f"股票名称变更历史已保存到 {namechange_path}")
    else:
        print("股票名称变更历史已存在，跳过下载。")

def download_dividend(name='分红送股'):
    basic_path = get_market_data_path('dividend.parquet')#

    if not basic_path.exists():
        print(f"--- 正在下载{name}信息 ---")
        #获取报告期日list
        all_data_list = []
        stocks = get_all_stock_basic_from_api()
        for index,stock_code in enumerate(stocks):
            print(f"处理第{index+1}/{len(stocks)}个股票-{name}数据")
            df = call_pro_tushare_api('dividend',ts_code=stock_code)
            all_data_list.append(df)

        # 合并所有季度数据
        final_df = pd.concat(all_data_list, ignore_index=True)#final_df[final_dfduplicated()]
        final_df = final_df.drop_duplicates()
        final_df.to_parquet(basic_path)
        print(f'{name}保存完毕')
    else:
        print(f"{name}已存在，跳过下载。")

# --- 4. 主下载逻辑 ---
if __name__ == '__main__':
    #todo daily_hfq 曾发生过 数据year字段重复问题！！ 务必重视
    if not MARKET_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"市场数据根目录不存在: {MARKET_DATA_ROOT}")

    # --- 下载配套数据 (逻辑不变) ---
    print("\n===== 开始下载全局配套数据 =====")
    trade_cal_path = get_market_data_path('trade_cal.parquet')
    if not trade_cal_path.exists():
        print("--- 正在下载交易日历 ---")
        trade_cal = call_pro_tushare_api("trade_cal", start_date=f'{START_YEAR}0101', end_date=f'{END_YEAR}1231')
        if not trade_cal.empty:
            trade_cal.to_parquet(trade_cal_path)
            print(f"交易日历已保存到 {trade_cal_path}")
    else:
        print("交易日历已存在，跳过下载。")

    stock_basic_path = get_market_data_path('stock_basic.parquet')

    download_stock_info(stock_basic_path)

    # --- 按年份循环下载核心数据 ---
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n===== 开始处理年份: {year} =====")
        year_start = f'{year}0101'
        year_end = get_year_end(year)

        all_stocks_df = pd.read_parquet(stock_basic_path)
        symbols = all_stocks_df[~all_stocks_df['ts_code'].str.endswith('.BJ')]['ts_code'].unique().tolist()
        print(f"获取到 {year} 年全市场 {len(symbols)} 只股票作为处理对象。")

        data_to_download_tasks = {
            # 模式: by_stock (逐个股票循环，适用于pro_bar等)
            'daily_hfq': {'func': 'pro_bar', 'params': {'adj': 'hfq', 'asset': 'E'}, 'mode': 'by_stock',
                          'api_type': 'ts'},
            'margin_detail': {'func': 'margin_detail', 'params': {}, 'mode': 'by_stock'},  # <-- 修正下载模式为 'by_stock'
            'stk_limit': {'func': 'stk_limit', 'params': {}, 'mode': 'by_stock'},

            # 模式: batch_stock (按股票代码分批，适用于大多数pro接口)
            'daily': {'func': 'daily', 'params': {}, 'mode': 'batch_stock'},
            'daily_basic': {'func': 'daily_basic', 'params': {}, 'mode': 'batch_stock'}

        }

        for name, info in data_to_download_tasks.items():
            year_path = get_market_data_path(name) / f"year={year}"
            if not year_path.exists():
                print(f"--- 正在下载 {year} 年的 {name} 数据 ---")
                all_data_list = []

                download_mode = info['mode']

                if download_mode == 'by_stock':
                    # 【新逻辑】按单个股票循环下载，保证数据完整性
                    print(f"  采用'逐个股票'模式下载...")
                    for i, symbol in enumerate(symbols):
                        print(f"    处理股票 {symbol} ({i + 1}/{len(symbols)})...")
                        api_params = {'ts_code': symbol, 'start_date': year_start, 'end_date': year_end,
                                      **info.get('params', {})}

                        api_type = info.get('api_type', 'pro')
                        if api_type == 'ts':
                            df_batch = call_ts_tushare_api(info['func'], **api_params)
                        else:
                            df_batch = call_pro_tushare_api(info['func'], **api_params)
                        all_data_list.append(df_batch)

                elif download_mode == 'batch_stock':
                    # 【旧逻辑】按股票分批下载
                    print(f"  采用'按股票分批'模式下载...")
                    for i in range(0, len(symbols), BATCH_SIZE):
                        batch_symbols = symbols[i: i + BATCH_SIZE]
                        symbols_str = ",".join(batch_symbols)
                        print(f"    处理批次 {i // BATCH_SIZE + 1} ({len(batch_symbols)}只股票)...")

                        api_params = {'ts_code': symbols_str, 'start_date': year_start, 'end_date': year_end,
                                      **info['params']}

                        df_batch = call_pro_tushare_api(info['func'], **api_params)
                        all_data_list.append(df_batch)

                # --- 通用的数据合并与保存逻辑 ---
                if all_data_list:
                    final_df = pd.concat(all_data_list, ignore_index=True)

                    final_df.drop_duplicates(inplace=True)

                    year_path.mkdir(parents=True, exist_ok=True)
                    final_df.to_parquet(year_path / 'data.parquet')
                    print(f"成功保存 {year} 年的 {name} 数据。")
            else:
                print(f"{year} 年的 {name} 数据已存在，跳过下载。")

    print("\n===== 所有数据下载任务完成！ =====")
# todo 注意 股票状态 需要每天刷新！
# todo trade_date 每天刷新！
# 也就是每天重新下载download_stock_info ，更新股票状态1
