##
# 为什么合成时必须“先中性化，再标准化”？我们再次回顾这个核心问题，因为它至关重要。目标: 等权合并。我们希望每个细分因子（如
# bm_ratio, ep_ratio）在最终的复合价值因子中贡献相等的影响力。问题: 不同的因子在经过中性化后，其残差的波动率（标准差）是不相等的。一个与风险因子相关性高的因子，其中性化后的残差波动会很小。解决方案: 必须在合并之前，对每一个中性化后的残差进行标准化，强行将它们的波动率都统一到1。只有这样，后续的“等权相加”才是真正意义上的“等权”#
from typing import List

import pandas as pd

from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import ConfigSnapshotManager
from projects._03_factor_selection.utils.factor_processor import FactorProcessor


class FactorSynthesizer:
    def __init__(self, factor_manager, factor_analyzer, factor_processor):
        """
        初始化因子合成器。
        Args:
            factor_processor: 传入你现有的、包含了预处理方法的对象实例。
                              我们假设这个对象有 .winsorize(), ._neutralize(), ._standardize() 等方法。
        """
        self.factor_manager = factor_manager
        self.factor_analyzer = factor_analyzer
        self.processor = factor_processor
        if factor_processor is None:
            self.processor = FactorProcessor(factor_manager.data_manager.config)
        self.sub_factors = {
            'ep_ratio',
            'bm_ratio',
            'sp_ratio',
        }

    # get_prepare_aligned_factor_for_analysis 现在智能处理：
    # - 价格数据（close/open/high/low）返回T日值（用于计算收益率）
    # - 因子数据返回T-1值（用于交易决策）
    # 因此这里直接使用返回值即可，无需额外处理
    # 注意啊 ，目前有个大坑，如果你用 close open high low 当成 种子因子来参与的话get_prepare_aligned_factor_for_analysis 里面有个判断 ，会返回t日的数据！ 解决：我们这里兼容一下 ，跟着判断 补充好t-1的逻辑即可！
    # 对每个子因子，都走一遍“去极值->中性化->标准化”的流程
    def get_sub_factor_df_do_pre_processed(self, factor_name: str, stock_pool_index_name: str) -> pd.DataFrame:
        """
        【核心】对单个细分因子，
        从raw 拿到
        该计算的做计算

        执行“去极值 -> 中性化 -> 标准化”的完整流程。
        return 最终的df
        """
        print(f"\n--- 正在处理细分因子: {factor_name} ---")

        # 【修正】get_prepare_aligned_factor_for_analysis 现在已经返回T-1值（除了价格数据）
        factor_df_shifted = self.factor_manager.get_prepare_aligned_factor_for_analysis(factor_name,
                                                                                        stock_pool_index_name, True)
        trade_dates = factor_df_shifted.index
        stock_codes = factor_df_shifted.columns

        # 生成t-1的数据 用于因子预处理
        (final_neutral_dfs, style_category
         ) = self.factor_analyzer.prepare_date_for_process_factor(factor_name, trade_dates, stock_codes,
                                                                  stock_pool_index_name)
        # 【删除】不再需要额外的shift(1)，因为get_prepare_aligned_factor_for_analysis已经处理了
        processed_df = self.processor.process_factor(
            factor_df_shifted=factor_df_shifted,
            target_factor_name=factor_name,
            neutral_dfs=final_neutral_dfs,
            pit_map=self.factor_manager.data_manager.pit_map,
            style_category=style_category, need_standardize=True)
        return processed_df

    def get_sub_factor_df_from_local(self, factor_name: str, stock_pool_index_name: str,snapshot_config_id) -> pd.DataFrame:
        """
        """
        manager = ConfigSnapshotManager()
        pool_index, s, e, config_evaluation = manager.get_snapshot_config_content_details(snapshot_config_id)
        version = f'{s}_{e}'
        resultLoadManager = ResultLoadManager(version=version)
        df = resultLoadManager.get_factor_data(factor_name, pool_index, s, e)
        #回想单因子测试：0101的数据经过shift 全行为nan，导致落库的数据从0102存，导致0101数据丢失。 所以后续必须严厉对齐
        return df

    def synthesize_composite_factor(self,
                                    composite_factor_name: str,
                                    stock_pool_index_name: str,
                                    sub_factor_names: List[str],
                                    wights:List[float],
                                    ) -> pd.DataFrame:
        """
        将一组细分因子合成为一个复合因子。

        Args:
            composite_factor_name (str): 最终合成的复合因子的名称 (e.g., 'Value_Composite').
            sub_factor_names (List[str]): 用于合成的细分因子名称列表.

        Returns:
            pd.DataFrame: 合成后的复合因子矩阵。 remind 已经是shift1 之后的
        """
        print(f"\n==============================================")
        print(f"开始合成复合因子: {composite_factor_name}")
        print(f"使用 {len(sub_factor_names)} 个细分因子: {sub_factor_names}")
        print(f"==============================================")

        processed_factors = []
        for factor_name in sub_factor_names:
            # 对每个子因子，都走一遍“去极值->中性化->标准化”的流程
            processed_df = self.get_sub_factor_df_do_pre_processed(factor_name, stock_pool_index_name)

            processed_factors.append(processed_df)

        # 最终合并：等权相加
        # 由于每个 processed_df 都已经是标准化（std=1）的，直接相加就是等波动率贡献
        if not processed_factors:
            raise ValueError("没有任何细分因子被成功处理，无法合成。")

        # 使用 reduce 和 add 来优雅地合并所有DataFrame
        composite_factor_df = self.do_composite_by_config_weights(processed_factors,wights)

        # 对最终结果再做一次标准化，使其成为一个标准的风格因子暴露 全市场
        ##
        # 子因子层面 (Stage 1)：分行业处理，目的是深入到每个行业内部，剔除噪音，挖掘纯粹的相对强弱。
        #
        # 复合因子层面 (Stage 2)：全市场处理，目的是将这些纯粹的信号整合后，进行全局定标，使其成为一个可以跨行业直接比较、并用于最终投资组合构建的标准化风格暴露。#
        composite_factor_df = self.processor._standardize_robust(composite_factor_df) #这里不再进行分行业是正确的

        print(f"\n复合因子 '{composite_factor_name}' 合成成功！")

        return composite_factor_df

    def do_composite_eq_wights(self, factor_name,stock_pool_index_name):
        # 这个合成就是老版本的合成（每次都是重新测试儿因子，效率差） 等权！，不可复用！
        """
        执行因子合成 - 支持等权和IC加权两种模式
        Args:
            factor_name: 合成因子名称
            stock_pool_index_name: 股票池名称
            weighting_config: IC权重配置（可选）
        Returns:
            pd.DataFrame: 合成后的因子数据
        """
        # 获取子因子列表
        sub_factor_names = self.factor_manager.data_manager.get_cal_require_base_fields_for_composite(factor_name)
        weights = self.factor_manager.data_manager.get_per_weights_for_composite(factor_name)
        composite_df = self.synthesize_composite_factor(factor_name, stock_pool_index_name, sub_factor_names,weights)
        return composite_df
#被替代
    def do_composite_wights_by_rolling_ic(self, factor_name, stock_pool_index, weighting_config=None, snap_config_id: str = None):
        """
        Args:
            factor_name: 合成因子名称
            stock_pool_index: 股票池名称
            weighting_config: IC权重配置（可选）
        Returns:
            pd.DataFrame: 合成后的因子数据
        """
        # 获取子因子列表
        sub_factor_names = self.factor_manager.data_manager.get_cal_require_base_fields_for_composite(factor_name)

        from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import (
            ICWeightedSynthesizer, FactorWeightingConfig
        )

        # 创建IC加权合成器
        config = weighting_config or FactorWeightingConfig()
        ic_synthesizer = ICWeightedSynthesizer(
            self.factor_manager,
            self.factor_analyzer,
            self.processor,
            config
        )

        # 执行IC加权合成
        composite_df, report = ic_synthesizer.synthesize_ic_weighted_factor(
            composite_factor_name=factor_name,
            stock_pool_index=stock_pool_index,
            candidate_factor_names=sub_factor_names,
            snap_config_id=snap_config_id

        )
        # 显示报告
        ic_synthesizer.print_synthesis_report(report)

        return composite_df
    def do_composite_route(self, factor_name, stock_pool_index_name, use_ic_weighting=True, weighting_config=None, snap_config_id=None):
        if use_ic_weighting:#升级为带正交化功能
            # 获取子因子列表
            sub_factor_names = self.factor_manager.data_manager.get_cal_require_base_fields_for_composite(factor_name)

            from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import \
                ICWeightedSynthesizer
            composite_df,_=(ICWeightedSynthesizer(self.factor_manager.data_manager, self.factor_analyzer, self.processor).synthesize_with_orthogonalization
             (composite_factor_name=factor_name,
              candidate_factor_names=sub_factor_names
              , snap_config_id=snap_config_id, force_generate_ic=False))
            # composite_df = self.do_composite_wights_by_rolling_ic(factor_name, stock_pool_index, weighting_config,snap_config_id)
        else:
            composite_df = self.do_composite_eq_wights(factor_name, stock_pool_index_name=stock_pool_index_name)
        return composite_df

    def do_composite_by_config_weights(self, processed_factors, weights=None):
        """
        根据给定的权重组合因子
        processed_factors: list[pd.Series 或 pd.DataFrame]
        weights: list[float] 或 None
        """
        from functools import reduce
        import numpy as np
        n = len(processed_factors)
        if weights is None:
            return reduce(lambda left, right: left.add(right, fill_value=0), processed_factors)
        else:
            if len(weights) != n:
                raise ValueError(f"weights 长度必须等于 processed_factors 长度 (got {len(weights)}, expected {n})")

            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()

            # 加权求和
            composite = sum(w * f for w, f in zip(weights, processed_factors))
            return composite




