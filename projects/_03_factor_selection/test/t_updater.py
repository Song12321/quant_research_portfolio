"""直接运行；本次需要哪些数据，就保留对应的函数调用。"""
from quant_lib.tushare.data import market_data_updater as updater

# 本地无数据时的起点，以及本次更新截止日。按需修改，不使用命令行参数。
INITIAL_DATE = '20200101'
END_DATE = '20260911'


if __name__ == '__main__':
    #
    # # 逐股接口依赖本地名单；名单需要刷新时先执行这一行。
    # updater.update_stock_basic()  # 全量刷新股票名单，含上市、退市、暂停上市股票
    # updater.update_industry_record()  # 全量刷新行业归属历史
    # updater.update_daily(INITIAL_DATE, END_DATE)  # 增量更新不复权日线行情
    # updater.update_daily_hfq(INITIAL_DATE, END_DATE)  # 增量更新后复权日线行情
    # updater.update_daily_basic(INITIAL_DATE, END_DATE)  # 增量更新每日指标，如市值、估值、换手率
    # updater.update_stk_limit(INITIAL_DATE, END_DATE)  # 增量更新每日涨跌停价格
    # updater.update_suspend(INITIAL_DATE, END_DATE)  # 增量更新停复牌事件
    # updater.update_balancesheet(INITIAL_DATE, END_DATE)  # 按公告日增量更新资产负债表
    # updater.update_cashflow(INITIAL_DATE, END_DATE)  # 按公告日增量更新现金流量表
    # updater.update_income(INITIAL_DATE, END_DATE)  # 按公告日增量更新利润表
    # updater.update_fina_indicator(INITIAL_DATE, END_DATE)  # 按公告日增量更新财务指标
    # updater.update_dividend()  # 逐股拉取完整分红历史，覆盖保存
    # updater.update_namechange()  # 逐股拉取完整名称变更历史，合并保存

    updater.update_hm_detail(INITIAL_DATE, END_DATE)  # 增量更新停复牌事件
