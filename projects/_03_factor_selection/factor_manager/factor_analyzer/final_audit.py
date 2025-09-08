# final_audit.py
import numpy as np
import pandas as pd


def final_audit(factor_data_path):
    """在无菌环境中，对截获的数据进行最终检验"""

    # --- 1. 加载"证物" ---
    print("--- 正在加载数据快照 ---")
    factor_df = pd.read_parquet(factor_data_path)
    price_df = pd.read_parquet(
        'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\debug_snapshot/price_for_returns.parquet')

    # --- 2. 用"证物"中的价格，重新计算收益率 ---
    #    完全复刻 calcu_forward_returns_close_close 的逻辑
    print("--- 正在重新计算20日 C2C 未来收益 ---")
    period = 20

    # 1. 定义起点和终点价格 (严格遵循T-1原则)
    start_price = price_df.shift(1)
    end_price = price_df.shift(1 - period)

    # 2. 创建"未来存续"掩码 (确保在持有期首尾股价都存在)
    survived_mask = start_price.notna() & end_price.notna()

    # 3. 计算原始收益率，并应用掩码过滤
    forward_returns_raw = end_price / start_price - 1
    forward_returns = forward_returns_raw.where(survived_mask)

    # 4. clip 操作应该在所有计算和过滤完成后进行
    forward_returns = forward_returns.clip(-0.15, 0.15)

    # --- 3. 最终对决：直接计算截面 Spearman 相关性 ---
    print("--- 正在计算因子与未来收益的单调性 ---")
    daily_corr = factor_df.corrwith(forward_returns, axis=1, method='spearman')

    # --- 4. 最终审判 ---
    avg_monotonicity = daily_corr.mean()

    print("\n" + "=" * 30 + " 【最终审判结果】 " + "=" * 30)
    print(f"在隔离的无菌环境中，因子与未来收益的平均单调性为: {avg_monotonicity:.6f}")
    print("=" * 80)

    if abs(avg_monotonicity) > 0.5:
        print(
            "✗ 结论：幽灵依然存在！这意味着你传入`comprehensive_test`的`factor_df`或`returns_calculator`在创建时就已经被污染！")
        print("  请回头检查你的`FactorCalculator`和`prepare_date_for_entity_service`！")
    else:
        print(
            "✓ 结论：幽灵消失了！这意味着你的上游数据和计算全部正确，bug藏在你`core_three_test`下游的某个具体测试函数（如`calculate_quantile_returns`）的实现中！")

    # --- 5. 额外诊断信息 ---
    print(f"\n🔍 诊断信息:")
    print(f"  因子数据形状: {factor_df.shape}")
    print(f"  收益率数据形状: {forward_returns.shape}")
    print(f"  有效相关性数量: {daily_corr.notna().sum()}")
    print(f"  相关性标准差: {daily_corr.std():.6f}")


def debug_spearman_calculation(quantile_means):
    """调试Spearman计算"""
    from scipy.stats import spearmanr

    # 实际的分位数均值
    quantile_ranks = list(range(1, 6))  # [1, 2, 3, 4, 5]

    print("🔍 手动验证Spearman计算:")
    print(f"  分位数序号: {quantile_ranks}")
    print(f"  分位数均值: {quantile_means}")

    # 手动计算Spearman
    corr, p_value = spearmanr(quantile_ranks, quantile_means)
    print(f"  Spearman相关性: {corr:.6f}")

    # 检查排名
    import numpy as np
    means_ranks = np.argsort(np.argsort(quantile_means)) + 1
    print(f"  均值的排名: {means_ranks}")
    print(f"  分位数排名: {quantile_ranks}")

    # 手动计算排名相关性
    rank_corr = np.corrcoef(quantile_ranks, means_ranks)[0, 1]
    print(f"  手动排名相关性: {rank_corr:.6f}")


def compare_daily_correlations(factor_df, forward_returns, n_dates=10):
    """比较多个日期的直接相关性和分层单调性"""

    print("🔍 比较多个日期的相关性模式:")

    common_dates = factor_df.index.intersection(forward_returns.index)[:n_dates]

    for i, date in enumerate(common_dates):
        factor_values = factor_df.loc[date].dropna()
        return_values = forward_returns.loc[date].dropna()

        common_stocks = factor_values.index.intersection(return_values.index)
        if len(common_stocks) < 100:
            continue

        factor_common = factor_values[common_stocks]
        return_common = return_values[common_stocks]

        # 直接相关性
        direct_corr = factor_common.corr(return_common, method='spearman')

        # 分层单调性
        try:
            quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
            df_temp = pd.DataFrame({'factor': factor_common, 'return': return_common, 'quantile': quantiles})
            group_means = df_temp.groupby('quantile')['return'].mean()

            from scipy.stats import spearmanr
            layer_mono, _ = spearmanr(range(len(group_means)), group_means.values)

            print(
                f"  📅 {date.strftime('%Y-%m-%d')}: 直接={direct_corr:.4f}, 分层={layer_mono:.4f}, 差异={abs(direct_corr - layer_mono):.4f}")

        except Exception as e:
            print(f"  ❌ {date}: 计算失败 - {e}")


def debug_perfect_monotonicity(factor_df, forward_returns):
    """调试完美单调性问题"""

    print("🚨 调试完美单调性异常...")

    # 选择第一个日期进行详细分析
    test_date = factor_df.index[2]
    factor_values = factor_df.loc[test_date].dropna()
    return_values = forward_returns.loc[test_date].dropna()

    common_stocks = factor_values.index.intersection(return_values.index)
    factor_common = factor_values[common_stocks]
    return_common = return_values[common_stocks]

    print(f"🔍 调试日期: {test_date}")
    print(f"  股票数量: {len(common_stocks)}")

    # 分组分析
    try:
        quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
        df_temp = pd.DataFrame({
            'factor': factor_common,
            'return': return_common,
            'quantile': quantiles
        })

        # 检查每个分位数的详细信息
        print(f"  📊 各分位数详细信息:")
        group_stats = df_temp.groupby('quantile').agg({
            'factor': ['count', 'mean', 'std', 'min', 'max'],
            'return': ['mean', 'std', 'min', 'max']
        }).round(6)

        print(group_stats)

        # 计算组均值
        group_means = df_temp.groupby('quantile')['return'].mean()
        print(f"  🎯 各组平均收益: {group_means.values}")

        # 检查是否所有组收益都相同
        unique_means = len(group_means.unique())
        print(f"  🚨 唯一均值数量: {unique_means}")

        if unique_means == 1:
            print(f"  ❌ 所有组收益完全相同: {group_means.iloc[0]}")
            print(f"  🔍 这解释了为什么单调性计算异常！")

            # 进一步检查原始收益数据
            print(f"  📈 原始收益统计:")
            print(f"    唯一收益值数量: {return_common.nunique()}")
            print(f"    收益值范围: [{return_common.min():.8f}, {return_common.max():.8f}]")
            print(f"    是否所有收益都相同: {return_common.nunique() == 1}")

        # 手动计算Spearman
        from scipy.stats import spearmanr
        mono_corr, _ = spearmanr(range(len(group_means)), group_means.values)
        print(f"  📊 手动计算单调性: {mono_corr}")

    except Exception as e:
        print(f"  ❌ 分组失败: {e}")


def debug_returns_calculation_detailed(price_df, period=20):
    """详细调试收益率计算"""

    print(f"🔍 详细调试 {period} 日收益率计算...")

    # 复刻你的计算逻辑
    start_price = price_df.shift(1)
    end_price = price_df.shift(1 - period)

    # 检查原始价格数据
    print(f"📊 价格数据检查:")
    print(f"  price_df shape: {price_df.shape}")
    print(f"  价格范围: [{price_df.min().min():.2f}, {price_df.max().max():.2f}]")

    # 选择一个测试日期和股票
    test_date = price_df.index[50]  # 选择中间的日期
    test_stock = price_df.columns[0]  # 选择第一只股票

    print(f"\n🔍 测试股票 {test_stock} 在日期 {test_date}:")

    # 获取具体的价格值
    start_price_val = start_price.loc[test_date, test_stock]
    end_price_val = end_price.loc[test_date, test_stock]

    print(f"  起始价格 (T-1): {start_price_val:.4f}")
    print(f"  结束价格 (T-{period}): {end_price_val:.4f}")

    # 计算收益率
    if pd.notna(start_price_val) and pd.notna(end_price_val) and start_price_val != 0:
        raw_return = end_price_val / start_price_val - 1
        print(f"  原始收益率: {raw_return:.6f}")
        print(f"  原始收益率 (%): {raw_return * 100:.4f}%")

        # 检查是否忘记减1
        ratio_only = end_price_val / start_price_val
        print(f"  仅比率 (未减1): {ratio_only:.6f}")

        # 检查方向是否正确
        if end_price_val > start_price_val:
            print(f"  ✓ 价格上涨，收益率应为正")
        else:
            print(f"  ✓ 价格下跌，收益率应为负")

    # 检查整体收益率分布
    forward_returns_raw = end_price / start_price - 1
    survived_mask = start_price.notna() & end_price.notna()
    forward_returns = forward_returns_raw.where(survived_mask)

    # 统计信息
    returns_flat = forward_returns.values.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]

    print(f"\n📊 整体收益率统计:")
    print(f"  数据点数量: {len(returns_flat)}")
    print(f"  均值: {np.mean(returns_flat):.6f}")
    print(f"  中位数: {np.median(returns_flat):.6f}")
    print(f"  标准差: {np.std(returns_flat):.6f}")
    print(f"  最小值: {np.min(returns_flat):.6f}")
    print(f"  最大值: {np.max(returns_flat):.6f}")
    print(f"  [1%, 99%] 分位数: [{np.percentile(returns_flat, 1):.6f}, {np.percentile(returns_flat, 99):.6f}]")

#
# # 调用这个函数
# def debug_grouping_data_transformation(factor_df,close_df,  period=20):
#     """调试分组过程中的数据变换"""
#
#     print("🔍 调试分组过程中的数据变换...")
#
#     # 1. 获取原始收益率
#     forward_returns = calcu_forward_returns_close_close( period,close_df)
#     print(f"📊 原始收益率统计:")
#     returns_flat = forward_returns.values.flatten()
#     returns_flat = returns_flat[~np.isnan(returns_flat)]
#     print(f"  均值: {np.mean(returns_flat):.6f}")
#     print(f"  标准差: {np.std(returns_flat):.6f}")
#     print(f"  范围: [{np.min(returns_flat):.6f}, {np.max(returns_flat):.6f}]")
#
#     # 2. 选择测试日期
#     test_date = factor_df.index[0]
#     factor_values = factor_df.loc[test_date].dropna()
#     return_values = forward_returns.loc[test_date].dropna()
#
#     print(f"\n🔍 测试日期 {test_date}:")
#     print(f"  收益率数据统计:")
#     print(f"    均值: {return_values.mean():.6f}")
#     print(f"    标准差: {return_values.std():.6f}")
#     print(f"    范围: [{return_values.min():.6f}, {return_values.max():.6f}]")
#     print(f"    前10个值: {return_values.head(10).values}")
#
#     # 3. 合并数据
#     common_stocks = factor_values.index.intersection(return_values.index)
#     factor_common = factor_values[common_stocks]
#     return_common = return_values[common_stocks]
#
#     print(f"\n📊 合并后数据:")
#     print(f"  收益率统计:")
#     print(f"    均值: {return_common.mean():.6f}")
#     print(f"    标准差: {return_common.std():.6f}")
#     print(f"    范围: [{return_common.min():.6f}, {return_common.max():.6f}]")
#
#     # 4. 分组
#     quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
#     df_temp = pd.DataFrame({
#         'factor': factor_common,
#         'return': return_common,
#         'quantile': quantiles
#     })
#
#     # 5. 检查分组后的原始数据
#     print(f"\n🔍 分组后各组原始收益率检查:")
#     for q in range(1, 6):
#         group_data = df_temp[df_temp['quantile'] == q]['return']
#         print(f"  Q{q}: 数量={len(group_data)}, 均值={group_data.mean():.6f}, 标准差={group_data.std():.6f}")
#         print(f"       前5个值: {group_data.head().values}")
#
#         # 检查是否有异常大的值
#         extreme_values = group_data[abs(group_data) > 1.0]  # 收益率>100%
#         if len(extreme_values) > 0:
#             print(f"       🚨 极端值数量: {len(extreme_values)}, 最大值: {extreme_values.max():.6f}")


def debug_monotonicity_skip_nan_dates(factor_df, returns_calculator, period=20):
    """跳过NaN日期，直接调试单调性问题"""

    print("🎯 跳过NaN日期，直接调试单调性...")

    # 获取收益率
    forward_returns = returns_calculator(period=period)

    # 找到有数据的日期，跳过前面的NaN
    common_dates = factor_df.index.intersection(forward_returns.index)

    print(f"📅 总日期数量: {len(common_dates)}")

    # 从第21个日期开始检查（确保有足够的历史数据）
    start_idx = max(21, period + 1)  # 确保跳过NaN期

    for i, test_date in enumerate(common_dates[start_idx:start_idx + 5]):  # 检查5个日期
        factor_values = factor_df.loc[test_date].dropna()
        return_values = forward_returns.loc[test_date].dropna()

        common_stocks = factor_values.index.intersection(return_values.index)

        if len(common_stocks) > 100:  # 确保有足够数据
            print(f"\n🔍 日期: {test_date} (第{start_idx + i + 1}个日期), 股票数量: {len(common_stocks)}")

            factor_common = factor_values[common_stocks]
            return_common = return_values[common_stocks]

            # 直接相关性
            direct_corr = factor_common.corr(return_common, method='spearman')

            # 分层单调性
            try:
                quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
                df_temp = pd.DataFrame({
                    'factor': factor_common,
                    'return': return_common,
                    'quantile': quantiles
                })

                group_means = df_temp.groupby('quantile')['return'].mean()

                from scipy.stats import spearmanr
                layer_mono, _ = spearmanr(range(len(group_means)), group_means.values)

                print(f"  📊 直接相关性: {direct_corr:.4f}")
                print(f"  📊 分层单调性: {layer_mono:.4f}")
                print(f"  📊 差异: {abs(direct_corr - layer_mono):.4f}")
                print(f"  🎯 各组均值: {group_means.values}")

                # 如果发现异常单调性
                if abs(layer_mono) > 0.99:
                    print(f"  🚨 发现异常单调性！")

                    # 检查收益率分布
                    print(f"  📈 收益率统计: 均值={return_common.mean():.6f}, 标准差={return_common.std():.6f}")
                    print(f"  📈 收益率范围: [{return_common.min():.6f}, {return_common.max():.6f}]")

                    # 检查各组的详细统计
                    for q in range(1, 6):
                        group_data = df_temp[df_temp['quantile'] == q]['return']
                        print(
                            f"    Q{q}: 均值={group_data.mean():.6f}, 中位数={group_data.median():.6f}, 数量={len(group_data)}")

            except Exception as e:
                print(f"  ❌ 计算失败: {e}")
        else:
            print(f"  ⚠️ 日期 {test_date}: 数据不足，跳过")


# 调用修正后的函数
def debug_spearman_calculation_detailed(factor_df, returns_calculator, period=20):
    """详细调试Spearman计算过程"""

    print("🔍 详细调试Spearman计算...")

    forward_returns = returns_calculator(period=period)
    common_dates = factor_df.index.intersection(forward_returns.index)

    # 选择一个测试日期
    test_date = common_dates[22]  # 对应2024-01-16

    factor_values = factor_df.loc[test_date].dropna()
    return_values = forward_returns.loc[test_date].dropna()
    common_stocks = factor_values.index.intersection(return_values.index)

    factor_common = factor_values[common_stocks]
    return_common = return_values[common_stocks]

    print(f"🔍 测试日期: {test_date}")
    print(f"📊 股票数量: {len(common_stocks)}")

    # 分组
    quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
    df_temp = pd.DataFrame({
        'factor': factor_common,
        'return': return_common,
        'quantile': quantiles
    })

    group_means = df_temp.groupby('quantile')['return'].mean()
    print(f"🎯 各组均值: {group_means.values}")

    # 手动计算Spearman相关性
    print(f"\n🔍 手动计算Spearman过程:")

    # 方法1：使用scipy
    from scipy.stats import spearmanr
    x_values = list(range(len(group_means)))  # [0, 1, 2, 3, 4]
    y_values = group_means.values

    print(f"  X值 (组序号): {x_values}")
    print(f"  Y值 (组均值): {y_values}")

    spearman_corr, p_value = spearmanr(x_values, y_values)
    print(f"  Scipy结果: 相关性={spearman_corr:.6f}, p值={p_value:.6f}")

    # 方法2：手动计算排名
    import numpy as np
    from scipy.stats import rankdata

    x_ranks = rankdata(x_values)
    y_ranks = rankdata(y_values)

    print(f"  X排名: {x_ranks}")
    print(f"  Y排名: {y_ranks}")

    # 计算Pearson相关性（对排名）
    manual_corr = np.corrcoef(x_ranks, y_ranks)[0, 1]
    print(f"  手动计算: {manual_corr:.6f}")

    # 方法3：检查是否有重复值影响
    unique_y = len(np.unique(y_values))
    print(f"  Y值唯一数量: {unique_y} / {len(y_values)}")

    if unique_y < len(y_values):
        print(f"  🚨 存在重复的组均值！")
        for i, val in enumerate(y_values):
            print(f"    组{i + 1}: {val:.8f}")

    # 方法4：检查是否因为数值精度问题
    print(f"\n🔍 数值精度检查:")
    for i in range(len(y_values) - 1):
        diff = y_values[i + 1] - y_values[i]
        print(f"  组{i + 1} -> 组{i + 2}: 差异 = {diff:.8f}")


def find_perfect_monotonicity_dates(factor_df, returns_calculator, period=20):
    """寻找出现完美单调性(1.0)的日期"""

    print("🎯 寻找完美单调性异常日期...")

    forward_returns = returns_calculator(period=period)
    common_dates = factor_df.index.intersection(forward_returns.index)

    perfect_dates = []

    for i, test_date in enumerate(common_dates[21:]):  # 跳过NaN期
        try:
            factor_values = factor_df.loc[test_date].dropna()
            return_values = forward_returns.loc[test_date].dropna()
            common_stocks = factor_values.index.intersection(return_values.index)

            if len(common_stocks) > 100:
                factor_common = factor_values[common_stocks]
                return_common = return_values[common_stocks]

                # 分组计算单调性
                quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
                df_temp = pd.DataFrame({
                    'factor': factor_common,
                    'return': return_common,
                    'quantile': quantiles
                })

                group_means = df_temp.groupby('quantile')['return'].mean()

                from scipy.stats import spearmanr
                mono_corr, _ = spearmanr(range(len(group_means)), group_means.values)

                # 寻找完美单调性
                if abs(mono_corr) > 0.99:
                    perfect_dates.append((test_date, mono_corr, group_means.values))
                    print(f"🚨 发现完美单调性！日期: {test_date}, 相关性: {mono_corr:.6f}")
                    print(f"   各组均值: {group_means.values}")

        except Exception as e:
            continue

    if perfect_dates:
        print(f"\n🎯 总共发现 {len(perfect_dates)} 个完美单调性日期")

        # 详细分析第一个异常日期
        test_date, mono_corr, group_means = perfect_dates[0]
        print(f"\n🔍 详细分析异常日期: {test_date}")

        factor_values = factor_df.loc[test_date].dropna()
        return_values = forward_returns.loc[test_date].dropna()
        common_stocks = factor_values.index.intersection(return_values.index)

        factor_common = factor_values[common_stocks]
        return_common = return_values[common_stocks]

        # 检查是否存在数据问题
        print(f"📊 收益率详细统计:")
        print(f"  唯一值数量: {return_common.nunique()}")
        print(f"  是否所有值相同: {return_common.nunique() == 1}")
        print(f"  最小值: {return_common.min():.8f}")
        print(f"  最大值: {return_common.max():.8f}")
        print(f"  标准差: {return_common.std():.8f}")

        # 检查分组后的详细情况
        quantiles = pd.qcut(factor_common, 5, labels=False, duplicates='drop') + 1
        df_temp = pd.DataFrame({
            'factor': factor_common,
            'return': return_common,
            'quantile': quantiles
        })

        print(f"📊 各组详细统计:")
        for q in range(1, 6):
            group_data = df_temp[df_temp['quantile'] == q]['return']
            print(f"  Q{q}: 数量={len(group_data)}, 均值={group_data.mean():.8f}")
            print(f"      标准差={group_data.std():.8f}, 唯一值={group_data.nunique()}")

    else:
        print("✅ 没有发现完美单调性异常")


def check_lookahead_bias(factor_df, returns_calculator, period=20):
    """检查是否存在前瞻性偏差"""

    print("🔍 检查前瞻性偏差...")

    forward_returns = returns_calculator(period=period)

    # 选择一个异常日期
    test_date = pd.Timestamp('2024-02-02')

    print(f"🔍 分析日期: {test_date}")

    # 检查因子值的时间戳
    factor_values = factor_df.loc[test_date].dropna()
    return_values = forward_returns.loc[test_date].dropna()

    print(f"📊 因子数据点数: {len(factor_values)}")
    print(f"📊 收益率数据点数: {len(return_values)}")

    # 检查因子值的分布
    print(f"📊 因子值统计:")
    print(f"  最小值: {factor_values.min():.6f}")
    print(f"  最大值: {factor_values.max():.6f}")
    print(f"  均值: {factor_values.mean():.6f}")
    print(f"  标准差: {factor_values.std():.6f}")

    # 检查收益率的时间窗口
    print(f"📊 收益率时间窗口检查:")
    print(f"  当前日期: {test_date}")
    print(f"  收益率应该是从 {test_date} 到 {test_date + pd.Timedelta(days=period)} 的收益")

    # 检查是否因子值与未来收益率有异常相关性
    common_stocks = factor_values.index.intersection(return_values.index)
    factor_common = factor_values[common_stocks]
    return_common = return_values[common_stocks]

    direct_corr = factor_common.corr(return_common, method='spearman')
    print(f"📊 直接Spearman相关性: {direct_corr:.6f}")

    # 如果相关性异常高，检查具体原因
    if abs(direct_corr) > 0.5:
        print(f"🚨 发现异常高相关性！")

        # 检查极端值
        factor_q99 = factor_common.quantile(0.99)
        factor_q01 = factor_common.quantile(0.01)

        high_factor_stocks = factor_common[factor_common > factor_q99].index
        low_factor_stocks = factor_common[factor_common < factor_q01].index

        high_factor_returns = return_common[high_factor_stocks]
        low_factor_returns = return_common[low_factor_stocks]

        print(f"  高因子值股票(前1%)平均收益: {high_factor_returns.mean():.6f}")
        print(f"  低因子值股票(后1%)平均收益: {low_factor_returns.mean():.6f}")
        print(f"  收益差异: {high_factor_returns.mean() - low_factor_returns.mean():.6f}")


# 检查前瞻性偏差

# 寻找异常日期

# 调用详细调试
# 调用调试
# 调用调试函数
# 调用这个函数
# if __name__ == '__main__':
    # factor_df = pd.read_parquet('D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\debug_snapshot/factor_to_test__prcessed.parquet')
    # price_df = pd.read_parquet('D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\debug_snapshot/price_for_returns.parquet')
    # # ✅ 使用正确的O2C函数
    # returns_calculator = partial(calculate_forward_returns_o2o, close_df=price_df, open_df=price_df)
    #
    # check_lookahead_bias(factor_df, returns_calculator, period=20)
    #
    # # debug_spearman_calculation([-0.012930, -0.012934, -0.013231, -0.014663, -0.013641])
    # # debug_spearman_calculation([-12930, -12934, -13231, -14663, -13641])
    # # debug_spearman_calculation([-0.00012, -0.00013, -0.00014, -0.00015, -0.00010])
    # # debug_spearman_calculation([-0.00012, -0.00013, -0.00014, -0.00012, -0.00010])
    # final_audit('D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\debug_snapshot/factor_to_test__prcessed.parquet')
    # final_audit('D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\debug_snapshot/factor_to_test__raw.parquet')
