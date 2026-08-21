from __future__ import annotations

import ast
from pathlib import Path

from mineru.backend.postprocess import table_merge

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COPYRIGHT_HEADER = "# Copyright (c) Opendatalab. All rights reserved."
_CHINESE_DOCSTRING_EXTRA_PATHS = (
    "mineru/model/flash/office",
    "mineru/model/flash/xycut.py",
    "mineru/model/model_types.py",
    "mineru/utils/spatial_text.py",
    "mineru/utils/text_utils.py",
)


def _absolute_imports(path: Path) -> set[str]:
    """读取 Python 文件中的绝对 import 模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def _relative_imports(path: Path) -> set[str]:
    """读取 Python 文件对同包模块的相对 import 名称。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module}


def _contains_chinese(text: str) -> bool:
    """判断职责说明中是否至少包含一个中文字符。"""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _iter_refactored_python_paths() -> list[Path]:
    """收集 backend 及本次从 utils 迁出的公共 Python 模块。"""
    paths = list((_PROJECT_ROOT / "mineru/backend").rglob("*.py"))
    for relative_path in _CHINESE_DOCSTRING_EXTRA_PATHS:
        path = _PROJECT_ROOT / relative_path
        paths.extend(path.rglob("*.py") if path.is_dir() else [path])
    return sorted(paths)


def test_backend_python_files_have_copyright_header() -> None:
    """守卫 backend 下每个 Python 文件都以统一版权声明开头。"""
    offenders = [
        str(path.relative_to(_PROJECT_ROOT))
        for path in (_PROJECT_ROOT / "mineru/backend").rglob("*.py")
        if path.read_text(encoding="utf-8").splitlines()[0] != _COPYRIGHT_HEADER
    ]
    assert not offenders


def test_refactored_definitions_have_chinese_docstrings() -> None:
    """守卫 backend 及迁出模块中的函数、方法和类均有中文职责说明。"""
    offenders: list[str] = []
    for path in _iter_refactored_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node)
            if not docstring or not _contains_chinese(docstring):
                relative_path = path.relative_to(_PROJECT_ROOT)
                offenders.append(f"{relative_path}:{node.lineno}:{node.name}")
    assert not offenders


def test_model_and_types_do_not_import_backend() -> None:
    """守卫模型层和公开类型层不再反向依赖 backend。"""
    paths = [*_PROJECT_ROOT.glob("mineru/model/**/*.py"), _PROJECT_ROOT / "mineru/types.py"]
    offenders = {
        str(path.relative_to(_PROJECT_ROOT)): sorted(
            module for module in _absolute_imports(path) if module.startswith("mineru.backend")
        )
        for path in paths
    }
    assert not {path: modules for path, modules in offenders.items() if modules}


def test_analysis_does_not_import_postprocess() -> None:
    """守卫 model-list 生产层不反向调用 Middle JSON 后处理层。"""
    offenders = {
        str(path.relative_to(_PROJECT_ROOT)): sorted(
            module for module in _absolute_imports(path) if module.startswith("mineru.backend.postprocess")
        )
        for path in _PROJECT_ROOT.glob("mineru/backend/analysis/**/*.py")
    }
    assert not {path: modules for path, modules in offenders.items() if modules}


def test_backend_utils_package_is_removed() -> None:
    """守卫 backend/utils 不再承载任何 Python 源码。"""
    utils_path = _PROJECT_ROOT / "mineru/backend/utils"
    assert not utils_path.exists()


def test_table_merge_package_keeps_one_way_internal_dependencies() -> None:
    """守卫 table_merge 低层模块不反向导入内容合并或文档编排模块。"""
    package_path = _PROJECT_ROOT / "mineru/backend/postprocess/table_merge"
    allowed_imports = {
        "models.py": set(),
        "html.py": {"models"},
        "blocks.py": {"html", "models", "rules"},
        "structure.py": {"blocks", "html", "models"},
        "content.py": {"blocks", "html", "models", "structure"},
        "document.py": {"blocks", "models", "structure"},
        "rules.py": set(),
    }
    actual_imports = {filename: _relative_imports(package_path / filename) for filename in allowed_imports}
    assert actual_imports == allowed_imports


def test_table_merge_public_contract_remains_callable() -> None:
    """验证本地入口和 mineru-vl-utils 使用的七个公开函数保持可调用。"""
    function_names = [
        "merge_table",
        "merge_table_content",
        "build_table_state_from_html",
        "build_row_rendered_cell_segments",
        "can_merge_by_structure",
        "calculate_row_rendered_segments",
        "detect_table_headers",
    ]
    assert table_merge.__all__ == function_names
    assert all(callable(getattr(table_merge, name, None)) for name in function_names)
