import os

import  pandas as pd
root = r'D:\lqs\quantity\market_data\stock\fundamentals'
daiy_2026=r'D:\lqs\quantity\market_data\stock\quotes\daily\year=2026'
suspend_d=r'D:\lqs\quantity\market_data\stock\trading_constraints\suspend_d.parquet'
hfq=r'D:\lqs\quantity\market_data\stock\quotes\daily_hfq\year=2025\data.parquet'
adj_factor=r'D:\lqs\quantity\market_data\stock\quotes\adj_factor\year=2026\data.parquet'
balancesheet_path = os.path.join(root, 'balancesheet.parquet')
x  = pd.read_parquet(balancesheet_path)
# x[x['ann_date']>x['f_ann_date']] 只有两条，发生于2011年。 结论：ann_date<=f_ann_date
#x[x['f_ann_date']<x['end_date']] 没有。 结论：f_ann_date>=end_date
print(x)