import  pandas as pd
daiy_2026=r'D:\lqs\quantity\market_data\stock\quotes\daily\year=2026'
suspend_d=r'D:\lqs\quantity\market_data\stock\trading_constraints\suspend_d.parquet'
hfq=r'D:\lqs\quantity\market_data\stock\quotes\daily_hfq\year=2025\data.parquet'
adj_factor=r'D:\lqs\quantity\market_data\stock\quotes\adj_factor\year=2026\data.parquet'
x  = pd.read_parquet(adj_factor)
print(x)