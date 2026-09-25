import os

import  pandas as pd
root = r'D:\lqs\quantity\market_data\stock\fundamentals'
daiy_2026=r'D:\lqs\quantity\market_data\stock\quotes\daily\year=2026'
suspend_d=r'D:\lqs\quantity\market_data\stock\trading_constraints\suspend_d.parquet'
namechange=r'D:\lqs\quantity\market_data\stock\corporate_actions\namechange.parquet'
hfq=r'D:\lqs\quantity\market_data\stock\quotes\daily_hfq\year=2025\data.parquet'
adj_factor=r'D:\lqs\quantity\market_data\stock\quotes\adj_factor\year=2026\data.parquet'
balancesheet_path = os.path.join(root, 'balancesheet.parquet')
df  = pd.read_parquet(namechange)
# x[x['ann_date']>x['start_date']] 只有两条，发生于2011年。 结论：ann_date<=f_ann_date
#x[x['f_ann_date']<x['end_date']] 没有。 结论：f_ann_date>=end_date
df['ann_date'] = pd.to_datetime(df['ann_date'], errors='coerce')
df['is_st'] = df['name'].str.upper().str.match(r'^\*?ST', na=False)

# 检查冲突：同一 ts_code + ann_date 内 is_st 不唯一
g = df.groupby(['ts_code', 'ann_date'])['is_st']
conflict_groups = g.nunique()[g.nunique() > 1]