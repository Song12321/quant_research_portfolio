import numpy as np
import pandas as pd
import statsmodels.api as sm

from projects._03_factor_selection.utils.factor_processor import FactorProcessor


def test_neutralization_outputs_only_residuals_for_complete_regression_rows():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    stocks = [f"S{i:02d}" for i in range(32)]
    market_cap = np.arange(32, dtype=float)
    factor_values = market_cap ** 2

    factor = pd.DataFrame(
        [factor_values, [*factor_values[:10], *([np.nan] * 22)]],
        index=dates,
        columns=stocks,
    )
    log_circ_mv = pd.DataFrame(
        [market_cap, market_cap],
        index=dates,
        columns=stocks,
    )
    log_circ_mv.loc[dates[0], "S31"] = np.nan
    processor = FactorProcessor({
        "preprocessing": {
            "neutralization": {
                "enable": True,
                "factors": ["market_cap"],
                "max_skip_ratio": 1.0,
            }
        }
    })

    actual = processor._neutralize(
        factor,
        target_factor_name="value_signal",
        neutral_dfs={"log_circ_mv": log_circ_mv},
        style_category="value",
    )

    complete_stocks = stocks[:31]
    expected = sm.OLS(
        factor.loc[dates[0], complete_stocks],
        sm.add_constant(log_circ_mv.loc[dates[0], complete_stocks]),
    ).fit().resid
    pd.testing.assert_series_equal(
        actual.loc[dates[0], complete_stocks],
        expected,
        check_names=False,
    )
    assert pd.isna(actual.loc[dates[0], "S31"])
    assert actual.loc[dates[1]].isna().all()
