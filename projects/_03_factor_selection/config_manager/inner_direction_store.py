"""根据 Inner 多周期 IC 均值确定方向，并增量保存结果。"""

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Union

import yaml


PathLike = Union[str, os.PathLike]


def _period_keys(configured_periods: Sequence[int]) -> list[str]:
    if not configured_periods:
        raise ValueError("Inner 方向计算失败：配置周期列表不能为空")
    keys = []
    for period in configured_periods:
        if isinstance(period, bool) or not isinstance(period, Integral) or period <= 0:
            raise ValueError(
                f"Inner 方向计算失败：周期实际值={period!r}，预期为正整数天数"
            )
        keys.append(f"{int(period)}d")
    if len(keys) != len(set(keys)):
        raise ValueError(f"Inner 方向计算失败：配置周期存在重复，实际值={keys}")
    return keys


def _extract_ic_means_and_valid_days(
    period_keys: list[str], ic_stats: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, float], dict[str, int]]:
    if not isinstance(ic_stats, Mapping):
        raise TypeError(
            "Inner 方向计算失败：ic_stats_periods_dict_processed 必须是映射"
        )
    actual_periods = list(ic_stats)
    if set(actual_periods) != set(period_keys):
        raise ValueError(
            "Inner 方向计算失败：IC 统计周期与配置周期不一致，"
            f"实际={actual_periods!r}, 预期={period_keys!r}"
        )
    means = {}
    valid_days = {}
    for key in period_keys:
        stats = ic_stats[key]
        if not isinstance(stats, Mapping) or "ic_mean" not in stats:
            raise ValueError(f"Inner 方向计算失败：周期 {key} 缺少唯一的 ic_mean")
        value = stats["ic_mean"]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(
                f"Inner 方向计算失败：周期 {key} 的 ic_mean={value!r}，预期为有限数"
            )
        if "ic_Valid Days" not in stats:
            raise ValueError(f"Inner 方向计算失败：周期 {key} 缺少 ic_Valid Days")
        count = stats["ic_Valid Days"]
        if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
            raise ValueError(
                f"Inner 方向计算失败：周期 {key} 的 ic_Valid Days={count!r}，"
                "预期为正整数"
            )
        means[key] = float(value)
        valid_days[key] = int(count)
    return means, valid_days


def _calculate_sample_weighted_score(
    means: Mapping[str, float], valid_days: Mapping[str, int]
) -> tuple[float, dict[str, float]]:
    total_valid_days = math.fsum(valid_days.values())
    weights = {key: valid_days[key] / total_valid_days for key in means}
    score = math.fsum(means[key] * weights[key] for key in means)
    return score, weights


def _load_direction_document(output_path: Path) -> dict:
    if not output_path.is_file():
        raise FileNotFoundError(f"方向配置不存在，无法增量写入：path={output_path}")
    with output_path.open("r", encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if not isinstance(document, dict) or set(document) != {"factors"}:
        raise ValueError(
            f"方向配置结构非法：path={output_path}，预期仅包含 factors 根节点"
        )
    if not isinstance(document["factors"], dict):
        raise ValueError(f"方向配置结构非法：path={output_path}，factors 必须是映射")
    return document


def _atomic_write_yaml(output_path: Path, document: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            yaml.safe_dump(document, file, allow_unicode=True, sort_keys=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_and_store_inner_direction(
    factor_name: str,
    configured_periods: Sequence[int],
    ic_stats_periods_dict_processed: Mapping[str, Mapping[str, object]],
    inner_run_id: str,
    output_path: PathLike,
) -> int:
    """按各周期实际非重叠 IC 有效节点数加权确定方向。"""
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise ValueError("Inner 方向写入失败：factor_name 必须是非空字符串")
    if not isinstance(inner_run_id, str) or not inner_run_id.strip():
        raise ValueError("Inner 方向写入失败：inner_run_id 必须是非空字符串")
    period_keys = _period_keys(configured_periods)
    means, valid_days = _extract_ic_means_and_valid_days(
        period_keys, ic_stats_periods_dict_processed
    )
    direction_score, weights = _calculate_sample_weighted_score(means, valid_days)
    if direction_score == 0:
        raise ValueError(
            f"Inner 方向计算失败：因子 {factor_name} 的加权多周期 ic_mean 为 0"
        )
    direction = 1 if direction_score > 0 else -1
    target = Path(output_path)
    document = _load_direction_document(target)
    if factor_name in document["factors"]:
        raise ValueError(
            f"Inner 方向写入失败：因子 {factor_name} 已存在，禁止覆盖，path={target}"
        )
    document["factors"][factor_name] = {
        "direction": direction,
        "direction_score": direction_score,
        "ic_mean_by_period": means,
        "ic_valid_days_by_period": valid_days,
        "direction_weight_by_period": weights,
        "inner_run_id": inner_run_id,
    }
    _atomic_write_yaml(target, document)
    return direction
