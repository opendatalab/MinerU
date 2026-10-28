"""保护 Tasks 共享契约、业务服务与具体部署实现之间的依赖边界。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["contracts", "errors", "execution", "service"])
def test_shared_tasks_import_without_http_or_parser_runtime(module: str, tmp_path: Path) -> None:
    """独立导入共享层时不得装载 HTTP、本地解析器或推理运行时。"""
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
for name in (
    "fastapi", "starlette", "torch", "transformers", "pypdfium2",
    "mineru.parser", "mineru.backend", "mineru.model", "mineru.config",
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, f"mineru.tasks.{module}"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "MINERU_CONFIG": str(tmp_path / "missing.yaml")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_router_import_does_not_load_local_api_server(tmp_path: Path) -> None:
    """Router 通过共享 Tasks 层组装路由，不借用本地 API Server 的实现。"""
    script = """
import sys

from mineru.kit.router.app import create_app
from mineru.kit.router.workers import RouterSettings

for name in (
    "mineru.parser.api_server", "mineru.parser.task_backend", "mineru.parser.task_api",
    "torch", "transformers", "pypdfium2",
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "MINERU_CONFIG": str(tmp_path / "missing.yaml")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_removed_private_task_api_module_has_no_forwarding_shell() -> None:
    """旧私有模块必须彻底移除，所有内部消费者显式依赖新的职责归属。"""
    old_module = Path(__file__).resolve().parents[2] / "mineru" / "parser" / "task_api.py"
    assert not old_module.exists()
