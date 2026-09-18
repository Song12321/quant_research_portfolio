"""按明确的 Inner 配置运行 processed 单因子研究。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from projects._03_factor_selection.config_manager.factor_definition_loader import (
    load_factor_definitions,
)
from projects._03_factor_selection.config_manager.inner_direction_store import (
    resolve_and_store_inner_direction,
)
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
    FactorAnalyzer,
)
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager, FactorResultsManager
from projects._03_factor_selection.factor_manager.storage.run_storage import (
    create_run_dir,
    write_effective_config,
)
from quant_lib.config.logger_config import log_success, setup_logger


logger = setup_logger(__name__)
DEFAULT_INNER_CONFIG = Path(__file__).parents[1] / "configs" / "research" / "inner.yaml" #todonew 不一定是inner


class EnhancedTestRunner:
    """顺序运行 Inner 因子研究，并在每个因子结束后释放临时数据。"""

    def __init__(self, research_config_path: str | Path = DEFAULT_INNER_CONFIG):
        # 记录入口配置路径，并预置本次运行中会变更的状态对象。
        self.research_config_path = Path(research_config_path).resolve()
        self.run_dir: Path | None = None
        self.direction_output_path: Path | None = None
        self.data = None
        self.factor_engine = None
        self.evaluator = None
        self.result_store = None

    def _initialize_components(self, config: dict) -> None:
        self.data = DataManager(config)
        self.factor_engine = FactorManager(
            self.data,
            results_dir=self.run_dir / "artifacts",
            apply_configured_direction=False,
        )
        self.evaluator = FactorAnalyzer(self.factor_engine)
        self.result_store = FactorResultsManager(results_dir=self.run_dir / "artifacts")

    def run(self, description: str = "Inner processed 因子研究") -> List[Dict]:
        config = self._prepare_run(description)
        self._initialize_components(config)
        self.data.prepare()
        results = self._research_factors(config)
        self._save_run_summary(results, config)
        return results

    def _prepare_run(self, description: str) -> dict:
        config = self._load_effective_config(description)
        self.run_dir = create_run_dir(
            Path(config["output_root"]),
            config["stage"],
            config["experiment_name"],
        )
        write_effective_config(self.run_dir, config)
        return config

    def _save_run_summary(self, results: List[Dict], config: dict) -> None:
        self._snapshot_direction_config(results)
        self._write_summary(results, config["stock_pool_name"])
        log_success(f"Inner 因子研究完成: {len(results)} 个因子，run={self.run_dir.name}")

    def _research_factors(self, config: dict) -> List[Dict]:
        stock_pool_name = config["stock_pool_name"]
        storage_name = self.data.get_stock_pool_storage_name_by_name(stock_pool_name)
        window = config["research_window"]
        results = []
        for experiment in config["experiments"]:
            factor_name = experiment["factor_name"]
            try:
                research_result = self.evaluator.evaluate_factor(
                    factor_name=factor_name,
                    stock_pool_index_name=stock_pool_name,
                )
                for calculator_name, result in research_result.items():
                    self.result_store._save_factor_results(
                        factor_name=factor_name,
                        stock_index=storage_name,
                        start_date=window["start_date"],
                        end_date=window["end_date"],
                        returns_calculator_func_name=calculator_name,
                        results=result,
                    )
                    del result
                direction = self._store_direction(factor_name, research_result, config)
                self.factor_engine.store_inner_resolved_direction(factor_name, direction)
                results.append(self._result_row(factor_name, stock_pool_name, direction))
                del research_result
            finally:
                self.factor_engine.clear_cache()
        return results

    def _load_effective_config(self, description: str) -> dict[str, Any]:
        # 加载配置并做严格校验，再补齐路径、定义与上下文字段后返回生效配置。
        config = self._load_yaml_mapping(self.research_config_path)
        if config.get("stage") != "inner":
            raise ValueError(f"当前入口仅支持 stage=inner，实际={config.get('stage')!r}")
        experiments = config.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("inner.yaml.experiments 必须是非空列表")
        self._validate_experiments(experiments)
        self._validate_inner_evaluation(config.get("evaluation"))
        self._require_non_empty_string(config, "stock_pool_name")
        profiles = config.get("stock_pool_profiles")
        if not isinstance(profiles, dict) or config["stock_pool_name"] not in profiles:
            raise ValueError("stock_pool_name 必须存在于 stock_pool_profiles")
        self._require_non_empty_string(config, "experiment_name")
        self._require_non_empty_string(config, "output_root")
        output_root = self._resolve_config_path(config, "output_root")
        factor_dir = self._resolve_config_path(config, "factor_definition_dir")
        self.direction_output_path = self._resolve_config_path(config, "direction_output_file")
        definitions = load_factor_definitions(factor_dir)
        definition_names = [row.get("name") for row in definitions if isinstance(row, dict)]
        missing = sorted(set(row["factor_name"] for row in experiments) - set(definition_names))
        if missing:
            raise ValueError(f"因子配置缺少 Inner 目标因子定义: factors={missing}")
        config["factor_definition"] = definitions
        config["output_root"] = str(output_root)
        config["factor_definition_dir"] = str(factor_dir)
        config["direction_output_file"] = str(self.direction_output_path)
        config["description"] = description
        self._validate_composite_dependencies(experiments, definitions)
        return config

    def _store_direction(self, factor_name: str, research_result: dict, config: dict) -> int:
        # 校验 inner 结果结构为 o2o，并写入单因子方向信息。
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

    def _write_summary(self, results: List[Dict], stock_pool_name: str) -> None:
        # 汇总运行结果并落盘，便于外部脚本快速读取。
        summary = {
            "run_id": self.run_dir.name,
            "stage": "inner",
            "stock_pool_name": stock_pool_name,
            "factors": [
                {"factor_name": row["factor_name"], "direction": row["direction"]}
                for row in results
            ],
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _snapshot_direction_config(self, results: List[Dict]) -> None:
        # 全部完成后保存一次方向快照；每个因子的方向已及时持久化。
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

    def _result_row(self, factor_name: str, stock_pool_name: str, direction: int) -> dict:
        # 标准化单条实验结果模型，供汇总与返回值统一消费。
        return {
            "factor_name": factor_name,
            "stock_pool_name": stock_pool_name,
            "direction": direction,
            "run_id": self.run_dir.name,
        }

    def _resolve_config_path(self, config: dict, key: str) -> Path:
        # 把配置中的相对路径转为配置文件所在目录的绝对路径。
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"inner.yaml 缺少非空路径字段: {key}")
        return (self.research_config_path.parent / value).resolve()


    @staticmethod
    def _validate_composite_dependencies(experiments: list[dict], definitions: list[dict]) -> None:
        # 检查复合因子依赖的子因子是否已在前面定义，避免运行时依赖未满足。
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
        # 验证每个实验条目结构统一、字段合法且因子名不重复。
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
        # 通用字符串字段校验，确保关键配置明确存在且不为空。
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"inner.yaml 缺少非空字符串字段: {key}")

    @staticmethod
    def _validate_inner_evaluation(evaluation: object) -> None:
        # 仅允许有效的 inner 评估配置：正整数周期、无重复、固定 o2o 计算方式。
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
        # 安全读取 YAML 并确认结果为字典结构，拒绝非法文件内容。
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"配置文件必须是 YAML 映射: {path}")
        return payload


if __name__ == "__main__":
    """正式 Inner 研究入口。"""
    EnhancedTestRunner(DEFAULT_INNER_CONFIG).run('Inner processed 因子研究')
