# 文件名: backfiller.py
# 作用：一个靶向数据回填工具，用于下载指定股票列表的历史数据，
#      并将其安全地合并到现有的、按年分区的Parquet数据库中。
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# --- 请确保这些导入路径和你的项目结构一致 ---
from quant_lib.config.constant_config import get_market_data_path
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.data.downloader import BATCH_SIZE

# --- 1. 核心配置 ---

# 【请在这里填入你缺失的200只股票的代码】
status_d_p_stocks = call_pro_tushare_api('stock_basic', list_status='D,P',
                                         fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type')
status_d_p_stocks['delist_date'] = pd.to_datetime(status_d_p_stocks['delist_date'])
filter =( ~status_d_p_stocks['ts_code'].str.endswith('.BJ') ) & (status_d_p_stocks['delist_date'] <=  pd.to_datetime('20250711')) & (status_d_p_stocks['delist_date']>=pd.to_datetime('20180101'))

MISSING_STOCKS_LIST = status_d_p_stocks[filter]['ts_code'].unique().tolist()

# 【请确认要回填的数据集】(通常与downloader.py中的任务一致)
DATASETS_TO_BACKFILL = [
     'margin_detail', 'stk_limit', 'daily',
    'daily_basic'
]

# 【请确认回填的时间范围】
START_YEAR = 2018
END_YEAR = datetime.now().year

# --- 2. 任务定义 (与你的downloader.py保持一致) ---
# 我们从你的downloader.py中复制任务定义，以保证下载逻辑的统一性
data_to_download_tasks = {
    # 模式: by_stock (逐个股票循环，适用于pro_bar等)

    'margin_detail': {'func': 'margin_detail', 'params': {}, 'mode': 'by_stock'},  # <-- 修正下载模式为 'by_stock'
    'stk_limit': {'func': 'stk_limit', 'params': {}, 'mode': 'by_stock'},

    # 模式: batch_stock (按股票代码分批，适用于大多数pro接口)
    'daily': {'func': 'daily', 'params': {}, 'mode': 'batch_stock'},
    'daily_basic': {'func': 'daily_basic', 'params': {}, 'mode': 'batch_stock'}



}


# --- 3. 辅助函数 ---
def get_year_end(year):
    if year == END_YEAR:
        beijing_now = datetime.utcnow() + timedelta(hours=8)
        beijing_yesterday = beijing_now - timedelta(days=1)
        return '20250711'
    return f'{year}1231'


def safe_merge_and_save(year_path: Path, new_data_df: pd.DataFrame, dataset_name: str):
    """安全地合并新数据到现有的Parquet文件"""
    if not new_data_df.empty:
        if year_path.exists():
            # 读取-修改-写回 模式
            print(f"    -> 正在合并数据到现有文件: {year_path}")
            existing_df = pd.read_parquet(year_path)
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)

            # 【核心】去重，保证数据唯一性。'last'会保留新下载的数据

            combined_df=combined_df.drop_duplicates(inplace=False)
            combined_df.to_parquet(year_path, index=False)
            print(f"    -> 合并完成。现有记录: {len(existing_df)}, 新增: {len(new_data_df)}, 合并后: {len(combined_df)}")
        else:
            # 如果年份文件不存在，则直接创建
            print(f"    -> 文件不存在，直接创建新文件: {year_path}")
            year_path.parent.mkdir(parents=True, exist_ok=True)
            new_data_df.to_parquet(year_path, index=False)
    else:
        print("    -> 未下载到新数据，无需合并。")


# --- 4. 主回填逻辑 ---
if __name__ == '__main__':
    if not MISSING_STOCKS_LIST:
        print("错误：请先在 MISSING_STOCKS_LIST 列表中填入需要回填的股票代码。")
    else:
        print(f"===== 开始对 {len(MISSING_STOCKS_LIST)} 只指定股票进行数据回填 =====")

        for dataset_name in DATASETS_TO_BACKFILL:
            print(f"\n--- 正在处理数据集: {dataset_name} ---")
            info = data_to_download_tasks.get(dataset_name)
            if not info:
                print(f"  - 未在任务列表中找到 {dataset_name} 的定义，跳过。")
                continue


            for year in range(START_YEAR, END_YEAR + 1):
                print(f"  -> 处理年份: {year}")
                year_start = f'{year}0101'
                year_end = get_year_end(year)

                all_new_data_list = []
                download_mode = info['mode']

                # 【核心修改】在这里区分下载模式
                if download_mode == 'by_stock':
                    print(f"    采用'逐个股票'模式下载...")
                    for i, symbol in enumerate(MISSING_STOCKS_LIST):
                        print(f"      处理股票 {symbol} ({i + 1}/{len(MISSING_STOCKS_LIST)})...")
                        api_params = {'ts_code': symbol, 'start_date': year_start, 'end_date': year_end,
                                      **info.get('params', {})}

                        api_type = info.get('api_type', 'pro')
                        try:
                            if api_type == 'ts':
                                df_batch = call_ts_tushare_api(info['func'], **api_params)
                            else:
                                df_batch = call_pro_tushare_api(info['func'], **api_params)
                            if df_batch is not None and not df_batch.empty:
                                all_new_data_list.append(df_batch)
                        except Exception as e:
                            print(f"\n      下载 {symbol} 在 {year} 年的数据失败: {e}")

                elif download_mode == 'batch_stock':
                    print(f"    采用'按股票分批'模式下载...")
                    for i in range(0, len(MISSING_STOCKS_LIST), BATCH_SIZE):
                        batch_symbols = MISSING_STOCKS_LIST[i: i + BATCH_SIZE]
                        symbols_str = ",".join(batch_symbols)
                        print(f"      处理批次 {i // BATCH_SIZE + 1} ({len(batch_symbols)}只股票)...")

                        api_params = {'ts_code': symbols_str, 'start_date': year_start, 'end_date': year_end,
                                      **info['params']}

                        try:
                            df_batch = call_pro_tushare_api(info['func'], **api_params)
                            if df_batch is not None and not df_batch.empty:
                                all_new_data_list.append(df_batch)
                        except Exception as e:
                            print(f"\n      下载批次 {i // BATCH_SIZE + 1} 在 {year} 年的数据失败: {e}")

                print()  # 换行

                if all_new_data_list:
                    final_new_df = pd.concat(all_new_data_list, ignore_index=True)
                    # 在合并到大文件前，先对本次下载的新数据进行一次去重
                    if not final_new_df.empty:
                        final_new_df=final_new_df.drop_duplicates(inplace=False)
                        year_path = get_market_data_path(dataset_name) / f"year={year}" / "data.parquet"
                        safe_merge_and_save(year_path, final_new_df,dataset_name   )

        print("\n===== 所有数据回填任务完成！ =====")
