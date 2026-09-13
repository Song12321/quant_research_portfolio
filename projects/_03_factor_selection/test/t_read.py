import  pandas as pd
daiy_2026=r'D:\lqs\quantity\market_data\stock\quotes\daily\year=2026'
suspend_d=r'D:\lqs\quantity\market_data\stock\trading_constraints\suspend_d.parquet'
x  = pd.read_parquet(suspend_d)
print(x)