"""严格加载按风格类型拆分的因子定义。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_factor_definitions(definitions_dir: str | Path) -> list[dict[str, Any]]:
    """按文件名顺序合并目录内的因子定义，并拒绝分类或名称冲突。"""
    definitions_dir = Path(definitions_dir).resolve()
    if not definitions_dir.is_dir():
        raise NotADirectoryError(f"因子定义目录不存在: path={definitions_dir}")
    definition_paths = sorted(definitions_dir.glob("*.yaml"))
    if not definition_paths:
        raise ValueError(f"因子定义目录没有 YAML 文件: path={definitions_dir}")

    definitions: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for definition_path in definition_paths:
        expected_category = definition_path.stem
        payload = _load_yaml_mapping(definition_path)
        if set(payload) != {"factor_definition"}:
            raise ValueError(f"因子类型文件字段非法: path={definition_path}, actual={sorted(payload)}")
        rows = payload["factor_definition"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"因子类型文件必须包含非空 factor_definition: path={definition_path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"因子定义必须是映射: path={definition_path}, index={index}, actual={row!r}")
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"因子名称必须是非空字符串: path={definition_path}, index={index}")
            if row.get("style_category") != expected_category:
                raise ValueError(
                    f"因子类型与文件名不一致: path={definition_path}, factor={name}, "
                    f"actual={row.get('style_category')!r}, expected={expected_category!r}"
                )
            if name in seen_names:
                raise ValueError(f"因子名称重复: factor={name}, path={definition_path}")
            seen_names.add(name)
            definitions.append(row)
    return definitions


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"因子配置文件不存在: path={path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"因子配置文件必须是 YAML 映射: path={path}, actual={type(payload).__name__}")
    return payload
