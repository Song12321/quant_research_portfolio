"""按 stage 运行 processed 因子研究并冻结或沿用方向。"""

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
DEFAULT_INNER_CONFIG = Path(__file__).parents[1] / "configs" / "research/inner" / "test.yaml"


class EnhancedTestRunner:
    """顺序运行因子研究，Inner 确定方向，Out / Finalout 沿用方向。"""

    def __init__(self, research_config_path: str | Path = DEFAULT_INNER_CONFIG):
        # 记录入口配置路径，并预置本次运行中会变更的状态对象。
        self.research_config_path = Path(research_config_path).resolve()
        self.run_dir: Path | None = None
        self.direction_path: Path | None = None
        self.stage = None
        self.data = None
        self.factor_engine = None
        self.evaluator = None
        self.result_store = None

    def _initialize_components(self, config: dict) -> None:
        # 数据管理器持有研究配置和共享缓存；实际日期、股票池在 prepare() 中准备。
        self.data = DataManager(config)
        # 同一配置决定阶段；后期引擎仅在初始化时读取一次方向文件。
        self.factor_engine = FactorManager(
            self.data,
            results_dir=self.run_dir / "artifacts",
            config=config,
        )
        # 评估器负责预处理及 IC、分层、换手计算，结果存储器负责写入本轮 artifacts。
        self.evaluator = FactorAnalyzer(self.factor_engine)
        self.result_store = FactorResultsManager(results_dir=self.run_dir / "artifacts")

    def run(self, description: str = "processed 因子研究") -> List[Dict]:
        # 校验研究配置、加载因子定义，创建独立运行目录并保存生效配置。
        config = self._prepare_run(description)
        # 组装共享数据管理器、因子计算引擎、评估器和结果存储器。
        self._initialize_components(config)
        # 读取交易日历，按因子依赖解析预热期，并构建本轮共用的研究股票池。

        self.data.prepare()
        # 按实验顺序研究因子：评估、保存产物、确定方向，并逐因子清理临时缓存。

        results = self._research_factors(config)
        # 全部因子成功后，保存本轮方向快照及 summary.json，返回各因子的结果记录。

        self._save_run_summary(results, config)
        return results

    def _prepare_run(self, description: str) -> dict:
        # 先完成配置和复合因子依赖校验；校验失败时不会创建运行目录。
        config = self._load_effective_config(description)
        self.stage = config["stage"]
        # 按输出根目录、研究阶段、时间戳和实验名创建目录，同时建立 artifacts 子目录。
        self.run_dir = create_run_dir(
            Path(config["output_root"]),
            config["stage"],
            config["experiment_name"],
        )
        # 将补齐因子定义、绝对路径和描述后的配置写入 effective_config.yaml。
        write_effective_config(self.run_dir, config)
        return config

    def _save_run_summary(self, results: List[Dict], config: dict) -> None:
        # 从已增量写入的方向配置中提取本轮因子，保存为运行目录内的独立快照。
        self._snapshot_direction_config(results)
        # 写入运行标识、股票池及因子方向列表；全部落盘后才打印完成日志。
        self._write_summary(results, config["stock_pool_name"])
        log_success(f"{self.stage} 因子研究完成: {len(results)} 个因子，run={self.run_dir.name}")

    def _research_factors(self, config: dict) -> List[Dict]:
        # 股票池名称用于评估，存储名称用于结果目录；所有实验使用同一研究窗口。
        stock_pool_name = config["stock_pool_name"]
        storage_name = self.data.get_stock_pool_storage_name_by_name(stock_pool_name)
        window = config["research_window"]
        results = []
        for experiment in config["experiments"]:
            factor_name = experiment["factor_name"]
            try:
                # 获取单因子或合成因子数据，完成预处理及 IC、分层收益和换手评估。
                # evaluate_factor 仅返回内存结果，下面再由结果存储器统一落盘。
                research_result = self.evaluator.evaluate_factor(
                    factor_name=factor_name,
                    stock_pool_index_name=stock_pool_name,
                )
                for calculator_name, result in research_result.items():
                    # 按股票池/因子/收益口径/日期窗口保存统计 JSON 和因子、IC、分层序列。
                    self.result_store._save_factor_results(
                        factor_name=factor_name,
                        stock_index=storage_name,
                        start_date=window["start_date"],
                        end_date=window["end_date"],
                        returns_calculator_func_name=calculator_name,
                        results=result,
                    )
                    del result
                if config["stage"] == "inner":
                    direction = self._store_direction(factor_name, research_result, config)
                    self.factor_engine.store_inner_resolved_direction(factor_name, direction)
                else:
                    direction = self.factor_engine.get_resolved_direction(factor_name)
                # 返回列表只保留轻量元信息，当前因子的完整评估数据已保存至 artifacts。
                results.append(self._result_row(factor_name, stock_pool_name, direction))
                del research_result
            finally:
                # 成功或异常都清理因子缓存和临时原始字段；异常继续向上传播，终止本轮。
                self.factor_engine.clear_cache()
        return results

    def _load_effective_config(self, description: str) -> dict[str, Any]:
        # 加载配置并做严格校验，再补齐路径、定义与上下文字段后返回生效配置。
        config = self._load_yaml_mapping(self.research_config_path)
        # 阶段决定是否推导方向，不能把拼写错误当成后期阶段。
        if config["stage"] not in ("inner", "out", "finalout"):
            raise ValueError(f"不支持的 stage: {config['stage']!r}")
        experiments = config.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("test.yaml.experiments 必须是非空列表")
        # 分别检查实验字段/重复因子，以及评估周期和唯一允许的 o2o 收益口径。
        self._validate_experiments(experiments)
        self._validate_inner_evaluation(config.get("evaluation"))
        # 股票池必须引用已配置的 profile，实验名和输出根目录也必须明确填写。
        self._require_non_empty_string(config, "stock_pool_name")
        profiles = config.get("stock_pool_profiles")
        if not isinstance(profiles, dict) or config["stock_pool_name"] not in profiles:
            raise ValueError("stock_pool_name 必须存在于 stock_pool_profiles")
        self._require_non_empty_string(config, "experiment_name")
        # 所有相对路径都以入口 YAML 所在目录解析，避免受启动工作目录影响。
        output_root = self._resolve_config_path(config, "output_root")
        factor_dir = self._resolve_config_path(config, "factor_definition_dir")
        self.direction_path = self._resolve_config_path(config, "direction_file")
        # 按文件名顺序加载分类定义，校验风格分类和重名。
        definitions = load_factor_definitions(factor_dir)
        # 将解析后的定义和路径回填为实际运行配置，供各组件使用并保存快照。
        config["factor_definition"] = definitions
        config["output_root"] = str(output_root)
        config["factor_definition_dir"] = str(factor_dir)
        config["direction_file"] = str(self.direction_path)
        config["description"] = description
        if config["stage"] == "inner":
            self._validate_composite_dependencies(experiments, definitions)
        return config

    def _store_direction(self, factor_name: str, research_result: dict, config: dict) -> int:
        # 校验 inner 结果结构为 o2o，并写入单因子方向信息。
        if set(research_result) != {"o2o"}:
            raise ValueError(f"Inner 方向只接受唯一 o2o 结果，实际={list(research_result)}")
        stats = research_result["o2o"]["ic_stats_periods_dict_processed"]
        # 下游严格校验周期和 IC 统计，以有效节点数加权 IC 均值的符号确定 ±1。
        # 加权得分为零或目标因子已存在都会报错；成功则连同统计依据和 run_id 一起写入。
        return resolve_and_store_inner_direction(
            factor_name=factor_name,
            configured_periods=config["evaluation"]["forward_periods"],
            ic_stats_periods_dict_processed=stats,
            inner_run_id=self.run_dir.name,
            output_path=self.direction_path,
        )

    def _write_summary(self, results: List[Dict], stock_pool_name: str) -> None:
        # 汇总运行结果并落盘，便于外部脚本快速读取。
        summary = {
            "run_id": self.run_dir.name,
            "stage": self.stage,
            "stock_pool_name": stock_pool_name,
            "factors": [
                {"factor_name": row["factor_name"], "direction": row["direction"]}
                for row in results
            ],
        }
        # 仅保存运行级索引信息，详细评估矩阵和统计已由逐因子存储流程另行保存。
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _snapshot_direction_config(self, results: List[Dict]) -> None:
        # 后期只保存内存中实际用过的记录（含子因子），不重读可能已变更的文件。
        if self.stage == "inner":
            factors = self._load_yaml_mapping(self.direction_path)["factors"]
            factors = {row["factor_name"]: factors[row["factor_name"]] for row in results}
        else:
            factors = self.factor_engine.used_direction_records
        snapshot = {"factors": factors}
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
            raise ValueError(f"test.yaml 缺少非空路径字段: {key}")
        return (self.research_config_path.parent / value).resolve()


    @staticmethod
    def _validate_composite_dependencies(experiments: list[dict], definitions: list[dict]) -> None:
        # 检查复合因子依赖的子因子是否已在前面定义，避免运行时依赖未满足。
        definitions_by_name = {definition.get("name"): definition for definition in definitions}
        for index, experiment in enumerate(experiments):
            definition = definitions_by_name[experiment["factor_name"]]
            if definition.get("action") != "composite":
                # 普通因子没有子因子执行顺序要求，直接检查下一项。
                continue
            # 复合因子的基础字段在此表示子因子名，必须显式提供非空列表。
            sub_factor_names = definition.get("cal_require_base_fields")
            if not isinstance(sub_factor_names, list) or not sub_factor_names:
                raise ValueError(
                    f"复合因子 {experiment['factor_name']} 必须配置非空子因子列表"
                )
            # 只认可当前实验之前的因子；缺失或排在后面的子因子均不满足依赖。
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
                    f"test.yaml.experiments[{index}] 字段非法，实际={row!r}，预期={sorted(expected)}"
                )
            if not all(isinstance(row[key], str) and row[key] for key in expected):
                raise ValueError(f"test.yaml.experiments[{index}] 的名称必须是非空字符串")
        # 条目逐个合法后，再检查整轮名称唯一，避免同名因子重复研究和写入方向。
        names = [row["factor_name"] for row in experiments]
        if len(names) != len(set(names)):
            raise ValueError(f"Inner 同一运行不得重复研究同名因子: factors={names}")

    @staticmethod
    def _require_non_empty_string(config: dict, key: str) -> None:
        # 通用字符串字段校验，确保关键配置明确存在且不为空。
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"test.yaml 缺少非空字符串字段: {key}")

    @staticmethod
    def _validate_inner_evaluation(evaluation: object) -> None:
        # 仅允许有效的 inner 评估配置：正整数周期、无重复、固定 o2o 计算方式。
        # 周期必须是非空的正整数列表；bool 虽属于 int 子类，也不能作为天数。
        periods = evaluation["forward_periods"]
        if not isinstance(periods, list) or not periods:
            raise ValueError("test.yaml.evaluation.forward_periods 必须是非空列表")
        if any(isinstance(period, bool) or not isinstance(period, int) or period <= 0 for period in periods):
            raise ValueError(f"Inner 周期必须是正整数: periods={periods!r}")
        if len(periods) != len(set(periods)):
            raise ValueError(f"Inner 周期不得重复: periods={periods!r}")
        # 本入口只接受唯一 o2o 口径，以便后续从同一口径的 IC 统计确定方向。
        if evaluation.get("returns_calculator") != ["o2o"]:
            raise ValueError("Inner 当前仅支持 returns_calculator: ['o2o']")

    @staticmethod
    def _load_yaml_mapping(path: Path) -> dict[str, Any]:
        # 安全读取 YAML 并确认结果为字典结构，拒绝非法文件内容。
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"配置文件必须是 YAML 映射: {path}")
        return payload


if __name__ == "__main__":
    """阶段、窗口和方向路径均由研究配置指定。"""
    EnhancedTestRunner(DEFAULT_INNER_CONFIG).run()
