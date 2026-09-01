"""按 experiments.yaml 批量运行 processed 因子研究。"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from projects._03_factor_selection.config_manager.base_config import workspaces_result_dir
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import (
    ConfigSnapshotManager,
)
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
    FactorAnalyzer,
)
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from quant_lib.config.logger_config import log_success, setup_logger


logger = setup_logger(__name__)


class EnhancedTestRunner:
    """初始化一次研究环境，并严格顺序执行配置中的因子。"""

    def __init__(self):
        config_dir = Path(__file__).parent
        self.config_path = config_dir / "config.yaml"
        self.experiments_config_path = config_dir / "experiments.yaml"
        self.snapshot_manager = ConfigSnapshotManager()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_manager = None
        self.factor_manager = None
        self.factor_analyzer = None

    def initialize_managers(self) -> None:
        self.data_manager = DataManager(
            config_path=str(self.config_path),
            experiments_config_path=str(self.experiments_config_path),
        )
        self.data_manager.prepare_basic_data()
        self.factor_manager = FactorManager(self.data_manager)
        self.factor_manager.clear_cache()
        self.factor_analyzer = FactorAnalyzer(self.factor_manager)

    def create_session_snapshot(self, description: str) -> str:
        experiments = self.data_manager.get_experiments_df()
        context = {
            "session_id": self.session_id,
            "description": description,
            "factor_names": experiments["factor_name"].tolist(),
            "stock_pools": experiments["stock_pool_name"].tolist(),
        }
        snapshot_id = self.snapshot_manager.create_snapshot(
            config=self.data_manager.config,
            snapshot_name=f"因子研究_{self.session_id}",
            test_context=context,
        )
        if not snapshot_id:
            raise RuntimeError("配置快照创建失败，因子研究已停止")
        return snapshot_id

    def run(self, description: str = "processed 因子研究") -> List[Dict]:
        self.initialize_managers()
        experiments = self.data_manager.get_experiments_df()
        if experiments.empty:
            raise ValueError("experiments.yaml 未配置任何因子研究任务")

        snapshot_id = self.create_session_snapshot(description)
        self._save_prices(experiments["stock_pool_name"].unique().tolist())
        results = []
        for row in experiments.itertuples(index=False):
            research_result = self.factor_analyzer.test_factor_entity_service_route(
                factor_name=row.factor_name,
                stock_pool_index_name=row.stock_pool_name,
            )
            self._link_result(snapshot_id, row.factor_name, row.stock_pool_name)
            results.append(
                {
                    "factor_name": row.factor_name,
                    "stock_pool_name": row.stock_pool_name,
                    "result": research_result,
                    "snapshot_id": snapshot_id,
                }
            )
        log_success(f"因子研究完成: {len(results)} 个因子")
        return results

    def _link_result(self, snapshot_id: str, factor_name: str, pool_name: str) -> None:
        pool_code = self.data_manager.get_stock_pool_index_code_by_name(pool_name)
        linked = self.snapshot_manager.link_test_result(
            snapshot_id=snapshot_id,
            factor_name=factor_name,
            stock_pool=pool_code,
            calc_type="o2o",
            version=(
                f"{self.data_manager.backtest_start_date}_"
                f"{self.data_manager.backtest_end_date}"
            ),
            test_description=f"批量测试_{self.session_id}",
        )
        if not linked:
            raise RuntimeError(f"配置快照关联失败: factor={factor_name}, pool={pool_name}")

    def _save_prices(self, pool_names: List[str]) -> None:
        for pool_name in pool_names:
            pool_code = self.data_manager.get_stock_pool_index_code_by_name(pool_name)
            version = (
                f"{self.data_manager.backtest_start_date}_"
                f"{self.data_manager.backtest_end_date}"
            )
            for price_type in ("close_hfq", "open_hfq", "high_hfq", "low_hfq"):
                price = self.factor_manager.get_prepare_aligned_factor_for_analysis(
                    price_type, pool_name, True
                )
                if price is None or price.empty:
                    raise ValueError(f"价格数据为空: pool={pool_name}, field={price_type}")
                output_dir = workspaces_result_dir / pool_code / price_type / version
                output_dir.mkdir(parents=True, exist_ok=True)
                price.to_parquet(output_dir / f"{price_type}.parquet")


def run_test_by_config(session_description: str = "processed 因子研究") -> List[Dict]:
    """正式批量研究入口。"""
    return EnhancedTestRunner().run(session_description)


if __name__ == "__main__":
    run_test_by_config()
