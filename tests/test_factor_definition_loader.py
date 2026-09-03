from pathlib import Path

import pytest
import yaml

from projects._03_factor_selection.config_manager.factor_definition_loader import (
    load_factor_definitions,
)


FACTOR_DEFINITION_DIR = (
    Path(__file__).parents[1]
    / "projects"
    / "_03_factor_selection"
    / "configs"
    / "factors"
    / "definitions"
)


def test_factor_definitions_are_split_one_style_per_file():
    definition_paths = sorted(FACTOR_DEFINITION_DIR.glob("*.yaml"))
    flattened = []

    assert len(definition_paths) == 17
    for definition_path in definition_paths:
        rows = yaml.safe_load(definition_path.read_text(encoding="utf-8"))["factor_definition"]
        assert {row["style_category"] for row in rows} == {definition_path.stem}
        flattened.extend(rows)

    assert len(flattened) == 63
    assert load_factor_definitions(FACTOR_DEFINITION_DIR) == flattened


def test_factor_definition_loader_rejects_style_file_mismatch(tmp_path):
    (tmp_path / "value.yaml").write_text(
        "factor_definition:\n  - name: 'demo'\n    style_category: 'quality'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="因子类型与文件名不一致"):
        load_factor_definitions(tmp_path)


def test_factor_definition_loader_rejects_duplicate_names(tmp_path):
    for category in ("value", "quality"):
        (tmp_path / f"{category}.yaml").write_text(
            "factor_definition:\n"
            f"  - name: 'demo'\n    style_category: '{category}'\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="因子名称重复"):
        load_factor_definitions(tmp_path)
