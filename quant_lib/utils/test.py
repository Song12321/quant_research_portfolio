import numpy as np
import pandas as pd
import pandas as pd
import statsmodels.api as sm
from quant_lib.config.constant_config import get_market_data_path


def read(file_path):
    data = pd.read_parquet(file_path)
    check_step(data, 0, 's')
    print(f"data columns--->{data.columns}")


def deal_wide(df, i, step_description):
    print(f"=" * 30 + f" 【第{i}步：{step_description} " + "=" * 30)

    df = df.copy(deep=True)
    df = df['000001.SZ']
    print(df)

def check_step(df, i, step_description,is_wide=False):
    if is_wide:
        return deal_wide(df,i,step_description)
    df = df.copy(deep=True)
    print(f"=" * 30 + f" 【第{i}步：{step_description} " + "=" * 30)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    start_date = pd.to_datetime('2025-05-15')
    end_date = pd.to_datetime('2025-06-24')
    df = df[df['trade_date'].between(start_date, end_date)]
    df = df[df['ts_code'] == "000001.SZ"]

    df=df.sort_values('trade_date', inplace=False)
    print(df[['ts_code','trade_date','pct_chg']])

    print("=" * 80 + "\n")


def mockShift_neutral_V2():
    # --- 0. 模拟基础数据 ---
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))
    stocks = [f'stock_{i}' for i in range(100)]
    o2c_returns = pd.DataFrame(np.random.randn(100, 100) * 0.02, index=dates, columns=stocks)

    # --- 1. 模拟两个被“当日信息”污染的因子 ---
    factor_V = o2c_returns + np.random.randn(100, 100) * 0.05
    factor_B = 0.5 * o2c_returns + np.random.randn(100, 100) * 0.05

    # --- 中性化函数 (不变) ---
    def neutralize(target_df, neutral_df):
        resid_df = target_df.copy()
        for date in target_df.index:
            y = target_df.loc[date].dropna()
            X = neutral_df.loc[date].reindex(y.index).dropna()
            common_index = y.index.intersection(X.index)
            if len(common_index) < 10: continue
            y, X = y[common_index], X[common_index]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            resid_df.loc[date, common_index] = model.resid
        return resid_df

    # --- 2. 【错误流程】先中性化，后移位 (但测试时不移位) ---
    # V(t) ~ beta(t) => 得到一个被污染的 processed_V(t)
    processed_V_contaminated = neutralize(factor_V, factor_B)

    # 【!!! 核心修正: 移除.shift(1), 模拟T日因子直接与T日收益比较的错误场景 !!!】
    # 这才是你系统中真正发生的事情：一个被T日信息污染的因子，直接和T日的收益率发生了关系
    ic_contaminated = processed_V_contaminated.corrwith(o2c_returns, axis=0).mean()

    print(f"【V2.0 错误流程】IC (先中性化，直接比较): {ic_contaminated:.4f}")

    # --- 3. 【正确的流程】先移位，后中性化 ---
    factor_V_shifted = factor_V.shift(1)
    factor_B_shifted = factor_B.shift(1)
    # V(t-1) ~ beta(t-1) => 得到一个干净的 processed_V(t-1)
    processed_V_clean = neutralize(factor_V_shifted, factor_B_shifted)
    # 直接用干净的 T-1 因子，去和 R(t) 比较
    ic_clean = processed_V_clean.corrwith(o2c_returns, axis=0).mean()

    print(f"【V2.0 正确流程】IC (先移位，后中性化): {ic_clean:.4f}")
import pandas as pd


def delete_duplicates_partitioned():
    """
    【分区版】数据清洗脚本
    1. 自动扫描并加载所有分区数据。
    2. 在完整数据集上进行去重。
    3. 将清洗后的数据，重新按年份分区存回硬盘。
    """
    # --- 步骤 0: 定义路径 ---
    # 只需要定义到分区文件夹的根目录
    BASE_DATA_DIR = get_market_data_path('daily_hfq').parent / 'daily_hfq_cleaned'
    # 【强烈建议】将清洗后的数据保存到一个全新的目录，而不是覆盖原始数据！
    CLEANED_DATA_DIR = get_market_data_path('daily_hfq').parent / 'daily_hfq_cleaned_v2'

    print(f"数据源路径: {BASE_DATA_DIR}")
    print(f"清洗后数据保存路径: {CLEANED_DATA_DIR}")

    # --- 步骤 1: 加载所有分区数据 ---
    # pandas 的 read_parquet 非常智能，可以直接读取分区根目录
    print("\n1. 正在加载所有分区数据，这可能需要一些时间...")
    # 【风险提示】如果你的总数据量非常大（例如超过你电脑内存的50%），这里可能会报错。
    # 如果遇到内存问题，请告诉我，我们需要使用更高级的工具（如Dask或PyArrow）。
    try:
        df = pd.read_parquet(BASE_DATA_DIR)
        print(f"✓ 所有分区数据加载成功，总记录数: {len(df)}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return

    # --- 步骤 2: 识别并分析重复数据 (逻辑不变) ---
    duplicates = df[df.duplicated(subset=['ts_code', 'trade_date'], keep=False)]

    if not duplicates.empty:
        print(f"\n2. !!! 发现 {len(duplicates)} 条重复记录，占总数据量的 {len(duplicates) / len(df):.4%} !!!")

        # --- 步骤 3: 执行清洗流程 ---
        print("\n3. 开始执行清洗流程...")

        # 【核心修正 I - 重新生成'year'列】
        # 由于原始的 'year' 分区列可能是错误的来源，我们不能信任它。
        # 我们应该基于 'trade_date' 重新生成一个绝对正确的 'year' 列，用于后续的分区保存。
        print("   > 正在从 'trade_date' 重新生成正确的 'year' 列...")
        df['trade_date'] = pd.to_datetime(df['trade_date'])  # 确保是日期格式
        df['year'] = df['trade_date'].dt.year

        # 验证价量数据一致性 (逻辑不变)
        cols_to_check = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        mismatched_groups = duplicates.groupby(['ts_code', 'trade_date'])[cols_to_check].nunique()
        mismatched = mismatched_groups.max().max() > 1
        if mismatched:
            print("   > 【严重警告】: 重复记录中的核心价量数据存在不一致！需要人工介入检查！")
            # 可以在这里打印出不一致的样本，以便调试
            # print(duplicates[duplicates.groupby(['ts_code', 'trade_date'])['open'].transform('nunique') > 1])
            return  # 遇到严重问题，终止程序
        else:
            print("   > 重复记录中的核心价量数据一致，可以安全地去重。")

        # 执行去重，保留最后一条记录
        df_cleaned = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

        # --- 步骤 4: 验证清洗结果 (逻辑不变) ---
        print(f"\n4. 验证清洗结果...")
        print(f"   > 清洗前数据量: {len(df)}")
        print(f"   > 清洗后数据量: {len(df_cleaned)}")
        assert not df_cleaned.duplicated(subset=['ts_code', 'trade_date']).any(), "清洗失败，仍然存在重复值！"
        print("   > ✓ 数据清洗成功，所有重复记录已被处理。")

        # --- 步骤 5: 【核心修正 II - 按'year'列重新分区保存】 ---
        print(f"\n5. 正在将干净的数据重新分区保存至: {CLEANED_DATA_DIR}...")
        try:
            # 使用 to_parquet 的 partition_cols 参数，pandas会自动按'year'列创建子目录
            df_cleaned.to_parquet(
                CLEANED_DATA_DIR,
                partition_cols=['year'],
                engine='pyarrow'  # 推荐使用 pyarrow 引擎
            )
            print(f"   > ✓ 数据保存成功！")
        except Exception as e:
            print(f"   > ✗ 数据保存失败: {e}")

    else:
        print("\n✓ 恭喜，数据文件中没有发现重复记录。")


if __name__ == '__main__':
    path = get_market_data_path('index_daily.parquet')
    df =  pd.read_parquet(path, engine='pyarrow')
    print(df)
    # download_index_daily_info()
