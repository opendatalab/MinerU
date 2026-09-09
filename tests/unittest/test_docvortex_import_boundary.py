"""守卫宿主只通过声明的公开接口消费 DocVortex。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from docvortex.public_api import PUBLIC_API


def _dotted_name(node: ast.AST) -> str:
    """读取静态属性链，函数调用和下标表达式不冒充模块路径。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _check_source(source: str, *, model_layer: bool = False) -> list[str]:
    """检查包括惰性和类型导入的全部语法节点，并追踪模块别名。"""
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    errors: list[str] = []
    model_modules = {"docvortex.schema", "docvortex.options", "docvortex.geometry", "docvortex.assets"}

    def check_module(module: str, line: int) -> None:
        """按公开清单和宿主层级校验模块依赖。"""
        if module not in PUBLIC_API:
            errors.append(f"{line}: undeclared module {module}")
        if model_layer and module not in model_modules:
            errors.append(f"{line}: model layer cannot import {module}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").startswith("docvortex"):
            module = node.module or ""
            check_module(module, node.lineno)
            for alias in node.names:
                if alias.name not in PUBLIC_API.get(module, ()):
                    errors.append(f"{node.lineno}: undeclared symbol {module}.{alias.name}")
                target = f"{module}.{alias.name}"
                if target in PUBLIC_API:
                    aliases[alias.asname or alias.name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "docvortex" or alias.name.startswith("docvortex."):
                    check_module(alias.name, node.lineno)
                    aliases[alias.asname or "docvortex"] = alias.name if alias.asname else "docvortex"
        elif isinstance(node, ast.Call) and _dotted_name(node.func) in {"importlib.import_module", "__import__"}:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                module = node.args[0].value
                if module == "docvortex" or module.startswith("docvortex."):
                    check_module(module, node.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_name(node)
        root, _, suffix = dotted.partition(".")
        if root not in aliases:
            continue
        qualified = f"{aliases[root]}.{suffix}"
        if qualified in PUBLIC_API:
            continue
        module, _, symbol = qualified.rpartition(".")
        if module in PUBLIC_API and symbol not in PUBLIC_API[module]:
            errors.append(f"{node.lineno}: undeclared member {qualified}")
    return errors


def test_mineru_uses_declared_docvortex_api() -> None:
    """扫描活动生产代码，不让新增依赖穿透底层实现。"""
    root = Path(__file__).parents[2] / "mineru"
    errors = []
    for path in root.rglob("*.py"):
        errors.extend(
            f"{path.relative_to(root)}:{error}"
            for error in _check_source(path.read_text(), model_layer=path.is_relative_to(root / "model"))
        )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "source",
    [
        "from docvortex.foundation._geometry import normalize_to_int_bbox",
        "def load():\n    from docvortex.analyzers.native.pdf import native_text",
        "if TYPE_CHECKING:\n    from docvortex.geometry import unknown",
        "import docvortex.geometry as geometry\ngeometry.unknown()",
        "from docvortex.document import page_range as ranges\nranges.unknown()",
        "import docvortex\ndocvortex.foundation._geometry.normalize_to_int_bbox([])",
        "import importlib\nimportlib.import_module('docvortex.foundation._geometry')",
    ],
)
def test_guard_rejects_unpublished_imports(source: str) -> None:
    """覆盖不同导入形式和模块成员访问，避免只检查 import 行前缀。"""
    assert _check_source(source)


def test_model_layer_cannot_import_analyzers() -> None:
    """即使符号公开，模型层也不能反向依赖分析器。"""
    assert _check_source("from docvortex.analyzers.pdf import prepare_text_evidence", model_layer=True)
