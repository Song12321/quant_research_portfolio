"""为每次因子研究创建自包含、不可覆盖的运行目录。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]


def create_run_dir(output_root: Path, stage: str, experiment_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / "runs" / stage / f"{stamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    return run_dir


def write_effective_config(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "effective_config.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_manifest(
    run_dir: Path,
    stage: str,
    experiment_name: str,
    status: str,
) -> None:
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"研究运行状态非法: status={status!r}")
    now = datetime.now().isoformat()
    manifest_path = run_dir / "manifest.json"
    created_at = now
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        created_at = existing["created_at"]
        code_state = existing["code"]
    else:
        code_state = _get_git_state()
    payload = {
        "run_id": run_dir.name,
        "stage": stage,
        "experiment_name": experiment_name,
        "status": status,
        "created_at": created_at,
        "updated_at": now,
        "code": code_state,
        "artifacts_dir": "artifacts",
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_git_state() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    dirty = bool(_run_git("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
    )
    return completed.stdout.strip()
