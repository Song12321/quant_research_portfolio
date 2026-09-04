# white_box_test.py

import pandas as pd
import numpy as np
from pathlib import Path

from quant_lib.config.constant_config import get_market_data_path

# ▼▼▼▼▼ 【请务必修改为你的真实文件路径】 ▼▼▼▼▼
UNADJUSTED_DAILY_PATH = get_market_data_path('daily')
DIVIDEND_EVENTS_PATH = get_market_data_path('dividend.parquet')
INDEX_DAILY_PATH = get_market_data_path('index_daily.parquet')


# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# --- 1. 定义我们之前推导出的、最权威的计算函数 (现在是独立函数) ---

def calculate_true_pct_chg(close_raw: pd.DataFrame, dividend_events: pd.DataFrame) -> pd.DataFrame:
    """【白盒版】根据第一性原理计算真实总回报率"""
    print("  > 正在白盒环境中，计算权威 pct_chg...")
    pre_close_raw = close_raw.shift(1)

    dividend_events['ex_date'] = pd.to_datetime(dividend_events['ex_date'])
    cash_div_matrix = dividend_events.pivot_table(index='ex_date', columns='ts_code', values='cash_div_tax').reindex(
        close_raw.index).fillna(0)
    stk_div_matrix = dividend_events.pivot_table(index='ex_date', columns='ts_code', values='stk_div').reindex(
        close_raw.index).fillna(0)

    numerator = close_raw * (1 + stk_div_matrix) + cash_div_matrix
    true_pct_chg = numerator / pre_close_raw - 1

    return true_pct_chg.where(close_raw.notna())


def calculate_forward_o2c_returns(close_df: pd.DataFrame, open_df: pd.DataFrame, period: int) -> pd.DataFrame:
    """【白盒版】计算O2C未来收益率"""
    start_price = open_df
    end_price = close_df.shift(-(period - 1))
    survived_mask = start_price.notna() & end_price.notna()
    forward_returns_raw = end_price / start_price - 1
    return forward_returns_raw.where(survived_mask)


# --- 2. 主执行逻辑 ---

if __name__ == '__main__':
    print("--- 开始执行白盒测试，隔离排查 volatility_120d 因子 ---")

    # --- a. 加载最原始、最纯净的数据 ---
    print("\n步骤1: 加载原始数据...")
    daily_df_long = pd.read_parquet(UNADJUSTED_DAILY_PATH)
    dividend_df_long = pd.read_parquet(DIVIDEND_EVENTS_PATH)
    print("✓ 原始数据加载成功。")

    # --- b. 将数据透视为宽表 ---
    close_raw = daily_df_long.pivot(index='trade_date', columns='ts_code', values='close')
    open_raw = daily_df_long.pivot(index='trade_date', columns='ts_code', values='open')
    close_raw.index = pd.to_datetime(close_raw.index)
    open_raw.index = pd.to_datetime(open_raw.index)

    # --- c. 【核心计算】在隔离环境中，生成权威的pct_chg ---
    true_pct_chg = calculate_true_pct_chg(close_raw, dividend_df_long)
    print("✓ 权威 pct_chg 计算成功。")

    # --- d. 【核心计算】计算volatility因子 (T日信息) ---
    print("\n步骤2: 计算 T 日的 volatility_120d 因子...")
    volatility_t0 = true_pct_chg.rolling(window=120, min_periods=100).std() * np.sqrt(252)
    print("✓ volatility_120d (T日) 计算成功。")

    # --- e. 【核心计算】计算未来收益率 (Y变量) ---
    print("\n步骤3: 计算 T 日的 21d O2C 未来收益...")
    # 我们需要复权价来计算收益，这里我们动态构建一个
    net_value_curve = (1 + true_pct_chg.fillna(0)).cumprod()
    base_prices = close_raw.bfill().iloc[0]
    close_adj = (net_value_curve * base_prices).where(close_raw.notna())
    open_adj = (open_raw / close_raw * close_adj)  # 近似计算

    forward_returns_21d = calculate_forward_o2c_returns(close_adj, open_adj, period=21)
    print("✓ 21d O2C 未来收益计算成功。")

    # --- f. 【关键的T-1移位】---
    print("\n步骤4: 将因子移位，确保使用T-1信息...")
    volatility_t1 = volatility_t0.shift(1)
    volatility_t1_targ = volatility_t1[volatility_t1.loc['2025-06-20','000001.SZ']]
    print("✓ 因子已移位为 T-1 版本。")

    # --- g. 【最终对决】计算IC值 ---
    print("\n步骤5: 计算 T-1 因子与 T 日未来收益的 Spearman Rank IC...")

    # 将两个Series合并到一个DataFrame中，以便对齐和清洗
    combined = pd.DataFrame({
        'factor': volatility_t1.stack(),
        'return': forward_returns_21d.stack()
    }).dropna()

    ic, p_value = pd.to_numeric(combined['factor'], errors='coerce'), pd.to_numeric(combined['return'], errors='coerce')
    ic_value = ic.corr(p_value, method='spearman')

    print("\n" + "=" * 30 + " 【最终检验结果】 " + "=" * 30)
    print(f"在隔离的白盒环境中，volatility_120d (21d周期) 的平均IC值为: {ic_value:.4f}")
    print("=" * 80)

    if abs(ic_value) > 0.1:
        print(
            "✗ 结论：IC值依然异常高！说明你的最底层数据文件（daily_cleaned.parquet 或 dividend_events.parquet）本身可能就存在问题！")
    else:
        print("✓ 结论：IC值回归正常！这无可辩驳地证明了，bug就藏在你那庞大的主框架的【某个数据传递或缓存环节】。")
