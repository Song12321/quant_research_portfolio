"""按明确的 Inner 配置运行 processed 单因子研究。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from projects._03_factor_selection.config_manager.inner_direction_store import (
    resolve_and_store_inner_direction,
)
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
    FactorAnalyzer,
)
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.storage.run_storage import (
    create_run_dir,
    write_effective_config,
    write_manifest,
)
from quant_lib.config.logger_config import log_success, setup_logger


logger = setup_logger(__name__)
DEFAULT_INNER_CONFIG = Path(__file__).parents[1] / "configs" / "research" / "inner.yaml" #todonew 不一定是inner


class EnhancedTestRunner:
    """顺序运行 Inner 因子研究，并在每个因子结束后释放临时数据。"""

    def __init__(self, research_config_path: str | Path = DEFAULT_INNER_CONFIG):
        self.research_config_path = Path(research_config_path).resolve()
        self.run_dir: Path | None = None
        self.direction_output_path: Path | None = None
        self.data_manager = None
        self.factor_manager = None
        self.factor_analyzer = None

    def initialize_managers(self, experiments_path: Path) -> None:
        config_path = self.run_dir / "effective_config.yaml"
        self.data_manager = DataManager(str(config_path), str(experiments_path))
        self.data_manager.prepare_basic_data()
        self.factor_manager = FactorManager(
            self.data_manager,
            results_dir=self.run_dir / "artifacts",
            apply_configured_direction=False,
        )
        self.factor_manager.clear_cache()
        self.factor_analyzer = FactorAnalyzer(self.factor_manager)
        self.factor_analyzer.factor_results_manager.results_dir = self.run_dir / "artifacts"

    def run(self, description: str = "Inner processed 因子研究") -> List[Dict]:
        config = self._load_effective_config(description)
        stage = config["stage"]
        experiment_name = config["experiment_name"]
        self.run_dir = create_run_dir(Path(config["output_root"]), stage, experiment_name)
        write_effective_config(self.run_dir, config)
        experiments_path = self._write_experiments(
            config["experiments"], config["stock_pool_name"]
        )
        write_manifest(self.run_dir, stage, experiment_name, "running")
        try:
            results = self._execute_experiments(experiments_path, config)
            self._write_summary(results)
            write_manifest(self.run_dir, stage, experiment_name, "completed")
            log_success(f"Inner 因子研究完成: {len(results)} 个因子，run={self.run_dir.name}")
            return results
        except Exception:
            write_manifest(self.run_dir, stage, experiment_name, "failed")
            raise

    def _execute_experiments(self, experiments_path: Path, config: dict) -> List[Dict]:
        self.initialize_managers(experiments_path)
        experiments = self.data_manager.get_experiments_df()
        self._save_prices(experiments["stock_pool_name"].unique().tolist())
        self.factor_manager.clear_cache()
        results = []
        for row in experiments.itertuples(index=False):
            try:
                research_result = self.factor_analyzer.test_factor_entity_service_route(
                    factor_name=row.factor_name,
                    stock_pool_index_name=row.stock_pool_name,
                )
                direction = self._store_direction(row.factor_name, research_result, config)
                self.factor_manager.store_inner_resolved_direction(row.factor_name, direction)
                results.append(self._result_row(row, direction))
                self._snapshot_direction_config(results)
                del research_result
            finally:
                self.factor_manager.clear_cache()
        return results

    def _load_effective_config(self, description: str) -> dict[str, Any]:
        config = self._load_yaml_mapping(self.research_config_path)
        if config.get("stage") != "inner":
            raise ValueError(f"当前入口仅支持 stage=inner，实际={config.get('stage')!r}")
        experiments = config.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("inner.yaml.experiments 必须是非空列表")
        self._validate_experiments(experiments)
        self._validate_inner_evaluation(config.get("evaluation"))
        self._require_non_empty_string(config, "stock_pool_name")
        self._require_non_empty_string(config, "experiment_name")
        self._require_non_empty_string(config, "output_root")
        factor_path = self._resolve_config_path(config, "factor_definition_file")
        self.direction_output_path = self._resolve_config_path(config, "direction_output_file")
        factor_config = self._load_yaml_mapping(factor_path)
        definitions = factor_config.get("factor_definition")
        if not isinstance(definitions, list) or not definitions:
            raise ValueError(f"因子配置缺少非空 factor_definition: path={factor_path}")
        definition_names = [row.get("name") for row in definitions if isinstance(row, dict)]
        missing = sorted(set(row["factor_name"] for row in experiments) - set(definition_names))
        if missing:
            raise ValueError(f"因子配置缺少 Inner 目标因子定义: factors={missing}")
        config["factor_definition"] = definitions
        config["factor_definition_file"] = str(factor_path)
        config["direction_output_file"] = str(self.direction_output_path)
        config["description"] = description
        self._validate_composite_dependencies(experiments, definitions)
        self._validate_direction_target(experiments)
        return config

    def _store_direction(self, factor_name: str, research_result: dict, config: dict) -> int:
        if set(research_result) != {"o2o"}:
            raise ValueError(f"Inner 方向只接受唯一 o2o 结果，实际={list(research_result)}")
        stats = research_result["o2o"]["ic_stats_periods_dict_processed"]
        return resolve_and_store_inner_direction(
            factor_name=factor_name,
            configured_periods=config["evaluation"]["forward_periods"],
            ic_stats_periods_dict_processed=stats,
            inner_run_id=self.run_dir.name,
            output_path=self.direction_output_path,
        )

    def _save_prices(self, pool_names: List[str]) -> None:
        for pool_name in pool_names:
            pool_code = self.data_manager.get_stock_pool_index_code_by_name(pool_name)
            output_dir = self.run_dir / "artifacts" / "prices" / pool_code
            output_dir.mkdir(parents=True, exist_ok=False)
            for price_type in ("close_hfq", "open_hfq", "high_hfq", "low_hfq"):
                price = self.factor_manager.get_prepare_aligned_factor_for_analysis(
                    price_type, pool_name, True
                )
                if price is None or price.empty:
                    raise ValueError(f"价格数据为空: pool={pool_name}, field={price_type}")
                price.to_parquet(output_dir / f"{price_type}.parquet")

    def _write_experiments(
        self, experiments: list[dict], stock_pool_name: str
    ) -> Path:
        path = self.run_dir / "experiments.yaml"
        runtime_experiments = [
            {"factor_name": row["factor_name"], "stock_pool_name": stock_pool_name}
            for row in experiments
        ]
        path.write_text(
            yaml.safe_dump(runtime_experiments, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def _write_summary(self, results: List[Dict]) -> None:
        summary = {
            "run_id": self.run_dir.name,
            "stage": "inner",
            "factors": [
                {"factor_name": row["factor_name"], "direction": row["direction"]}
                for row in results
            ],
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _snapshot_direction_config(self, results: List[Dict]) -> None:
        document = self._load_yaml_mapping(self.direction_output_path)
        factors = document.get("factors")
        names = [row["factor_name"] for row in results]
        if not isinstance(factors, dict) or any(name not in factors for name in names):
            raise RuntimeError(f"方向配置缺少本次 Inner 结果: factors={names}")
        snapshot = {"factors": {name: factors[name] for name in names}}
        target = self.run_dir / "resolved_factors.yaml"
        target.write_text(
            yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def _result_row(self, row, direction: int) -> dict:
        return {
            "factor_name": row.factor_name,
            "stock_pool_name": row.stock_pool_name,
            "direction": direction,
            "run_id": self.run_dir.name,
        }

    def _resolve_config_path(self, config: dict, key: str) -> Path:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"inner.yaml 缺少非空路径字段: {key}")
        return (self.research_config_path.parent / value).resolve()

    def _validate_direction_target(self, experiments: list[dict]) -> None:
        document = self._load_yaml_mapping(self.direction_output_path)
        factors = document.get("factors")
        if not isinstance(factors, dict) or set(document) != {"factors"}:
            raise ValueError(f"方向配置结构非法: path={self.direction_output_path}")
        duplicates = sorted(row["factor_name"] for row in experiments if row["factor_name"] in factors)
        if duplicates:
            raise ValueError(f"Inner 因子方向已存在，禁止覆盖: factors={duplicates}")

    @staticmethod
    def _validate_composite_dependencies(experiments: list[dict], definitions: list[dict]) -> None:
        definitions_by_name = {definition.get("name"): definition for definition in definitions}
        for index, experiment in enumerate(experiments):
            definition = definitions_by_name[experiment["factor_name"]]
            if definition.get("action") != "composite":
                continue
            sub_factor_names = definition.get("cal_require_base_fields")
            if not isinstance(sub_factor_names, list) or not sub_factor_names:
                raise ValueError(
                    f"复合因子 {experiment['factor_name']} 必须配置非空子因子列表"
                )
            earlier = {row["factor_name"]: row for row in experiments[:index]}
            missing = [name for name in sub_factor_names if name not in earlier]
            if missing:
                raise ValueError(
                    f"复合因子 {experiment['factor_name']} 的子因子必须在同次 Inner 中提前完成: "
                    f"factors={missing}"
                )
    @staticmethod
    def _validate_experiments(experiments: list[dict]) -> None:
        expected = {"factor_name"}
        for index, row in enumerate(experiments):
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError(
                    f"inner.yaml.experiments[{index}] 字段非法，实际={row!r}，预期={sorted(expected)}"
                )
            if not all(isinstance(row[key], str) and row[key] for key in expected):
                raise ValueError(f"inner.yaml.experiments[{index}] 的名称必须是非空字符串")
        names = [row["factor_name"] for row in experiments]
        if len(names) != len(set(names)):
            raise ValueError(f"Inner 同一运行不得重复研究同名因子: factors={names}")

    @staticmethod
    def _require_non_empty_string(config: dict, key: str) -> None:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"inner.yaml 缺少非空字符串字段: {key}")

    @staticmethod
    def _validate_inner_evaluation(evaluation: object) -> None:
        if not isinstance(evaluation, dict):
            raise ValueError("inner.yaml.evaluation 必须是映射")
        periods = evaluation.get("forward_periods")
        if not isinstance(periods, list) or not periods:
            raise ValueError("inner.yaml.evaluation.forward_periods 必须是非空列表")
        if any(isinstance(period, bool) or not isinstance(period, int) or period <= 0 for period in periods):
            raise ValueError(f"Inner 周期必须是正整数: periods={periods!r}")
        if len(periods) != len(set(periods)):
            raise ValueError(f"Inner 周期不得重复: periods={periods!r}")
        if evaluation.get("returns_calculator") != ["o2o"]:
            raise ValueError("Inner 当前仅支持 returns_calculator: ['o2o']")

    @staticmethod
    def _load_yaml_mapping(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"配置文件必须是 YAML 映射: {path}")
        return payload


def run_test_by_config(
    session_description: str = "Inner processed 因子研究",
    research_config_path: str | Path = DEFAULT_INNER_CONFIG,
) -> List[Dict]:
    """正式 Inner 研究入口。"""
    return EnhancedTestRunner(research_config_path).run(session_description)


if __name__ == "__main__":
    run_test_by_config()
