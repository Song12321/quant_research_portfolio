import numpy as np
import pandas as pd
import pytest

from projects._03_factor_selection.utils.factor_processor import FactorProcessor


class StaticIndustryMap:
    def __init__(self, industry_map: pd.DataFrame):
        self.industry_map = industry_map

    def get_map_for_date(self, _date: pd.Timestamp) -> pd.DataFrame:
        return self.industry_map


def build_processor(min_samples: int, threshold: float = 1.0) -> FactorProcessor:
    return FactorProcessor({
        'preprocessing': {
            'winsorization': {
                'method': 'mad',
                'mad_threshold': threshold,
                'by_industry': {
                    'primary_level': 'l2_code',
                    'fallback_level': 'l1_code',
                    'min_samples': min_samples,
                },
            },
        },
    })


def build_factor(values: list[float]) -> pd.DataFrame:
    columns = pd.Index([f'S{i}' for i in range(1, len(values) + 1)], name='ts_code')
    return pd.DataFrame(
        [[np.nan] * len(values), values],
        index=pd.to_datetime(['2024-01-02', '2024-01-03']),
        columns=columns,
    )


def build_map(l1_codes: list[str], l2_codes: list[str]) -> pd.DataFrame:
    codes = pd.Index([f'S{i}' for i in range(1, len(l1_codes) + 1)], name='ts_code')
    return pd.DataFrame({'l1_code': l1_codes, 'l2_code': l2_codes}, index=codes)


def test_l2_small_sample_uses_l1_statistics_and_configured_threshold():
    processor = build_processor(min_samples=3, threshold=1.0)
    factor = build_factor([0.0, 1.0, 100.0])
    industry_map = build_map(['L1'] * 3, ['L2_A', 'L2_A', 'L2_B'])

    result = processor.winsorize_robust(factor, StaticIndustryMap(industry_map))

    assert result.loc['2024-01-03', 'S3'] == pytest.approx(1.0 + 1.4826)


def test_l1_small_sample_is_explicitly_excluded():
    processor = build_processor(min_samples=4)
    factor = build_factor([0.0, 1.0, 100.0])
    industry_map = build_map(['L1'] * 3, ['L2_A', 'L2_A', 'L2_B'])

    result = processor.winsorize_robust(factor, StaticIndustryMap(industry_map))

    assert result.loc['2024-01-03'].isna().all()
    assert processor.winsorization_exclusions == [
        {'date': '2024-01-03', 'ts_code': code, 'reason': 'insufficient_l1_samples'}
        for code in ['S1', 'S2', 'S3']
    ]


def test_missing_industry_is_explicitly_excluded():
    processor = build_processor(min_samples=2)
    factor = build_factor([0.0, 1.0, 100.0])
    industry_map = build_map(['L1', 'L1'], ['L2', 'L2'])

    result = processor.winsorize_robust(factor, StaticIndustryMap(industry_map))

    assert pd.isna(result.loc['2024-01-03', 'S3'])
    assert processor.winsorization_exclusions == [
        {'date': '2024-01-03', 'ts_code': 'S3', 'reason': 'missing_industry'}
    ]


def test_zero_mad_group_keeps_original_values():
    processor = build_processor(min_samples=5)
    factor = build_factor([1.0, 1.0, 1.0, 1.0, 10.0])
    industry_map = build_map(['L1'] * 5, ['L2'] * 5)

    result = processor.winsorize_robust(factor, StaticIndustryMap(industry_map))

    assert result.loc['2024-01-03', 'S5'] == 10.0


def test_first_research_day_is_explicitly_excluded():
    processor = build_processor(min_samples=2)
    factor = build_factor([1.0, 2.0])
    factor.iloc[0] = [3.0, 4.0]
    industry_map = build_map(['L1'] * 2, ['L2'] * 2)

    result = processor.winsorize_robust(factor, StaticIndustryMap(industry_map))

    assert result.iloc[0].isna().all()
    assert processor.winsorization_exclusions[:2] == [
        {'date': '2024-01-02', 'ts_code': code, 'reason': 'missing_previous_trading_day'}
        for code in ['S1', 'S2']
    ]


def test_industry_standardization_does_not_keep_first_day_raw_values():
    processor = build_processor(min_samples=2)
    processor.preprocessing_config['standardization'] = {
        'method': 'zscore',
        'by_industry': {
            'primary_level': 'l2_code',
            'fallback_level': 'l1_code',
            'min_samples': 2,
        },
    }
    factor = build_factor([1.0, 2.0])
    factor.iloc[0] = [3.0, 4.0]
    industry_map = build_map(['L1'] * 2, ['L2'] * 2)

    result = processor._standardize_robust(factor, StaticIndustryMap(industry_map))

    assert result.iloc[0].isna().all()


def test_mad_rejects_unused_quantile_range():
    processor = build_processor(min_samples=2)
    processor.preprocessing_config['winsorization']['quantile_range'] = [0.01, 0.99]

    with pytest.raises(ValueError, match='method=mad 时不得配置 quantile_range'):
        processor.winsorize_robust(build_factor([1.0, 2.0]))
