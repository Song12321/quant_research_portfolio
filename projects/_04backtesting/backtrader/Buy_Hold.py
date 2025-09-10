# -*- coding: utf-8 -*-
# 聚宽策略示例：绘制并计算中证800指数走势及年化收益
import jqdatasdk
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 设置参数 ---
# 设置回测时间
start_date = '2019-04-01'
end_date = '2023-12-31'
# 中证800指数代码
index_code = '000906.XSHG'

# --- 2. 获取数据 ---
# 在聚宽研究环境中运行，会自动进行认证
df = get_price(index_code, start_date=start_date, end_date=end_date, frequency='daily', fields=['close'])

# --- 3. 绘制走势图 (与您的代码相同) ---
plt.figure(figsize=(12,6))
plt.plot(df.index, df['close'], label='CSI 800 Index', color='blue')
plt.title('中证800指数走势 (2019-04 ~ 2023-12)')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.grid(True)
plt.legend()
plt.show()

# --- 4. 【专业版】年化收益率计算 ---
# a) 提取期初和期末价格
start_price = df['close'][0]
end_price = df['close'][-1]

# b) 计算总交易日数
num_trading_days = len(df)

# c) 计算以“年”为单位的时间跨度 (使用252作为每年的平均交易日数)
num_years = num_trading_days / 252.0

# d) 计算复合年化增长率 (CAGR)
# 这是衡量多年期投资回报率的业界标准方法
annualized_return = (end_price / start_price) ** (1 / num_years) - 1

# --- 5. 打印结果 ---
print("-" * 50)
print("中证800指数区间表现分析")
print("-" * 50)
print(f"统计区间: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
print(f"期初价格: {start_price:.2f}")
print(f"期末价格: {end_price:.2f}")
print(f"区间总交易日: {num_trading_days} 天")
print(f"复合年化收益率 (CAGR): {annualized_return:.2%}")
print("-" * 50)