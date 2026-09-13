import  pandas as pd
daiy_2026=r'D:\lqs\quantity\market_data\stock\quotes\daily\year=2026'
x  = pd.read_parquet(daiy_2026)
print(x)