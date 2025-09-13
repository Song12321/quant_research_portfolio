FACTOR_DIRECTIONS = {
    'bm_ratio': 1,  # 比如 book-to-market 是反向的，值越小越好（代表成长）
    'ep_ratio': 1,  # earning-to-price 是正向的，值越大越好
    'sp_ratio': 1,  # earning-to-price 是正向的，值越大越好
    'beta': -1,
    'log_circ_mv': -1,
    # ...
}
#用这个
NEW_FACTOR_DIRECTIONS = {
    'volatility_40d': -1,  # 比如 book-to-market 是反向的，值越小越好（代表成长）
    'operating_accruals':-1,
    'volatility_120d':-1,
    'volatility_90d':-1,
    'ln_turnover_value_90d':-1,
    'revenue_growth_ttm':1,
    'amihud_liquidity':1,
    'log_circ_mv':1,
    'cfp_ratio': 1,  # earning-to-price 是正向的，值越大越好
    'earnings_stability': 1,  # earning-to-price 是正向的，值越大越好
    'ep_ratio': 1,
    'sp_ratio': 1,
    'revenue_growth_ttm': 1,
    'sp_ratio_orth_vs_cfp_ratio': 1,
    'ep_ratio_orth_vs_earnings_stability': 1,
    'sp_ratio_orth_vs_ep_ratio': 1,
    'ep_ratio_orth_vs_cfp_ratio': 1,
    'operating_accruals_orth_vs_cfp_ratio':-1,
    'amihud_liquidity_orth_vs_ln_turnover_value_90d':1,
    'total_revenue_growth_yoy_orth_vs_revenue_growth_ttm':1,
    'volatility_120d_orth_vs_ln_turnover_value_90d':-1,
    'value_composite_orth_vs_cfp_ratio':1,

    # ...
}
def get_new_factor_direction(factor_name: str) -> int:
    return NEW_FACTOR_DIRECTIONS[factor_name]
