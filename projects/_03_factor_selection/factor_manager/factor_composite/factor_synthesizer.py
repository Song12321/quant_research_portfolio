"""将已经 processed 的子因子等权合成为一个正式信号。"""

from typing import List

import pandas as pd


class FactorSynthesizer:
    """唯一合成规则：子因子分别预处理后等权平均。"""

    def __init__(self, factor_manager, factor_analyzer, factor_processor):
        if factor_manager is None or factor_analyzer is None or factor_processor is None:
            raise ValueError("等权合成必须传入 FactorManager、FactorAnalyzer 和 FactorProcessor")
        self.factor_manager = factor_manager
        self.factor_analyzer = factor_analyzer
        self.processor = factor_processor

    def get_processed_sub_factor(
        self, factor_name: str, stock_pool_name: str
    ) -> pd.DataFrame:
        factor_shifted = self.factor_manager.get_prepare_aligned_factor_for_analysis(
            factor_name, stock_pool_name, True
        )
        neutral_dfs, style_category = (
            self.factor_analyzer.prepare_data_for_process_factor(
                factor_name,
                factor_shifted.index,
                factor_shifted.columns,
                stock_pool_name,
            )
        )
        return self.processor.process_factor(
            factor_df_shifted=factor_shifted,
            target_factor_name=factor_name,
            neutral_dfs=neutral_dfs,
            pit_map=self.factor_manager.data_manager.pit_map,
            style_category=style_category,
            need_standardize=True,
        )

    def synthesize_equal_factor(
        self, composite_name: str, stock_pool_name: str
    ) -> pd.DataFrame:
        sub_factor_names = (
            self.factor_manager.data_manager.get_cal_require_base_fields_for_composite(
                composite_name
            )
        )
        if not sub_factor_names:
            raise ValueError(f"复合因子 {composite_name} 必须显式配置至少一个子因子")

        processed_factors = []
        for name in sub_factor_names:
            processed = self.get_processed_sub_factor(name, stock_pool_name)
            direction = self.factor_manager.get_inner_resolved_direction(name)
            processed_factors.append(processed * direction)
        composite = self.equal_average(processed_factors)
        return self.processor._standardize_robust(composite)

    @staticmethod
    def equal_average(processed_factors: List[pd.DataFrame]) -> pd.DataFrame:
        """等权平均；任一子因子缺失时不以 0 填充。"""
        if not processed_factors:
            raise ValueError("等权合成至少需要一个 processed 子因子")
        factor_count = len(processed_factors)
        return sum(factor / factor_count for factor in processed_factors)
