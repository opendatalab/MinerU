from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

from mineru.backend.postprocess import table_merge

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COPYRIGHT_HEADER = "# Copyright (c) Opendatalab. All rights reserved."
_SOURCE_ROOTS = (
    _PROJECT_ROOT / "mineru/backend",
    _PROJECT_ROOT / "mineru/model",
    _PROJECT_ROOT / "mineru/render",
    _PROJECT_ROOT / "mineru/utils",
    _PROJECT_ROOT / "mineru/parser",
)
_HEADER_PATHS = (
    _PROJECT_ROOT / "mineru/backend",
    _PROJECT_ROOT / "mineru/model/flash",
    _PROJECT_ROOT / "mineru/model/runtime",
    _PROJECT_ROOT / "mineru/model/registry.py",
    _PROJECT_ROOT / "mineru/model/download.py",
    _PROJECT_ROOT / "mineru/model/ocr/geometry.py",
    _PROJECT_ROOT / "mineru/model/ocr/image.py",
    _PROJECT_ROOT / "mineru/model/ocr/language.py",
    _PROJECT_ROOT / "mineru/model/ocr/results.py",
    _PROJECT_ROOT / "mineru/render",
    _PROJECT_ROOT / "mineru/utils",
    _PROJECT_ROOT / "mineru/parser/file_type.py",
    _PROJECT_ROOT / "mineru/parser/page_range.py",
    _PROJECT_ROOT / "mineru/parser/process_control.py",
    _PROJECT_ROOT / "mineru/parser/writer.py",
)
_CHINESE_DOCSTRING_PATHS = (
    _PROJECT_ROOT / "mineru/backend",
    _PROJECT_ROOT / "mineru/model/runtime/device.py",
    _PROJECT_ROOT / "mineru/model/runtime/memory.py",
    _PROJECT_ROOT / "mineru/model/ocr/geometry.py",
    _PROJECT_ROOT / "mineru/model/ocr/image.py",
    _PROJECT_ROOT / "mineru/model/ocr/results.py",
    _PROJECT_ROOT / "mineru/render/markdown.py",
    _PROJECT_ROOT / "mineru/render/html.py",
    _PROJECT_ROOT / "mineru/render/docx.py",
    _PROJECT_ROOT / "mineru/render/structured_content.py",
    _PROJECT_ROOT / "mineru/utils/image.py",
)
_REMOVED_INTERNAL_MODULES = (
    "mineru.backend.local_model_runtime",
    "mineru.model.model_types",
    "mineru.model.flash.model",
    "mineru.model.flash.native_pdf",
    "mineru.model.flash.office.docx.tools.math",
    "mineru.model.utils",
    "mineru.render.writer",
    "mineru.render._internal.common.inline",
    "mineru.utils.backend_options",
    "mineru.utils.config_reader",
    "mineru.utils.model_registry",
    "mineru.utils.native_pdf_table",
    "mineru.utils.ocr_utils",
    "mineru.utils.pdf_document",
)


def _module_name(path: Path) -> str:
    """把项目内 Python 路径转换为完整模块名。"""
    relative = path.relative_to(_PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolved_imports(path: Path) -> set[str]:
    """把绝对和相对 import 都解析为完整模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_name = _module_name(path)
    package_parts = module_name.split(".") if path.name == "__init__.py" else module_name.split(".")[:-1]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module:
                imports.add(node.module)
            continue
        resolved_parts = package_parts[:]
        if node.level > 1:
            resolved_parts = resolved_parts[: -(node.level - 1)]
        if node.module:
            resolved_parts.extend(node.module.split("."))
        imports.add(".".join(resolved_parts))
    return imports


def _relative_imports(path: Path) -> set[str]:
    """读取 Python 文件对同包模块的相对 import 名称。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}


def _contains_chinese(text: str) -> bool:
    """判断职责说明中是否至少包含一个中文字符。"""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _iter_python_paths(path: Path) -> list[Path]:
    """返回文件自身或目录下全部 Python 文件。"""
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def test_target_python_files_have_copyright_header() -> None:
    """守卫本次目标目录中的一方 Python 文件使用统一版权头。"""
    paths = [
        path
        for configured_path in _HEADER_PATHS
        for path in _iter_python_paths(configured_path)
        if "_internal/pytorchocr" not in path.as_posix() and path.name != "cli_parser.py"
    ]
    offenders = [
        str(path.relative_to(_PROJECT_ROOT)) for path in paths if path.read_text().splitlines()[0] != _COPYRIGHT_HEADER
    ]
    assert not offenders


def test_new_first_party_definitions_have_chinese_docstrings() -> None:
    """守卫本次新增的一方函数、方法和类都有中文职责说明。"""
    offenders: list[str] = []
    for configured_path in _CHINESE_DOCSTRING_PATHS:
        for path in _iter_python_paths(configured_path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                docstring = ast.get_docstring(node)
                if not docstring or not _contains_chinese(docstring):
                    offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}:{node.name}")
    assert not offenders


def test_active_mineru_imports_use_relative_form() -> None:
    """守卫活动生产代码的 MinerU 内部引用统一使用相对 import。"""
    offenders: list[str] = []
    for path in (_PROJECT_ROOT / "mineru").rglob("*.py"):
        if "cli_old" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module == "mineru" or (node.module or "").startswith("mineru."))
            ):
                offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "mineru" or alias.name.startswith("mineru."):
                        offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{node.lineno}:{alias.name}")
    assert not offenders


def test_layer_dependencies_are_one_way() -> None:
    """守卫 utils、model、backend 与 render 的单向依赖边界。"""
    offenders: dict[str, list[str]] = {}
    model_paths = [*_PROJECT_ROOT.glob("mineru/model/**/*.py"), _PROJECT_ROOT / "mineru/types.py"]
    for path in model_paths:
        invalid = sorted(module for module in _resolved_imports(path) if module.startswith(("mineru.backend", "mineru.render")))
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    for path in _PROJECT_ROOT.glob("mineru/backend/**/*.py"):
        invalid = sorted(module for module in _resolved_imports(path) if module.startswith("mineru.render"))
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    for path in _PROJECT_ROOT.glob("mineru/utils/**/*.py"):
        invalid = sorted(
            module
            for module in _resolved_imports(path)
            if module.startswith(("mineru.backend", "mineru.model", "mineru.render", "mineru.parser"))
        )
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    for path in _PROJECT_ROOT.glob("mineru/backend/analysis/**/*.py"):
        invalid = sorted(module for module in _resolved_imports(path) if module.startswith("mineru.backend.postprocess"))
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    allowed_render_backend = (
        "mineru.backend.postprocess.inline",
        "mineru.backend.postprocess.table_merge",
    )
    for path in _PROJECT_ROOT.glob("mineru/render/**/*.py"):
        invalid = sorted(
            module
            for module in _resolved_imports(path)
            if module.startswith("mineru.backend")
            and not any(module == allowed or module.startswith(f"{allowed}.") for allowed in allowed_render_backend)
        )
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    assert not offenders


def test_flash_office_does_not_depend_on_pdf_implementation() -> None:
    """守卫 Office 格式只复用中立能力，不反向依赖 Flash PDF 实现。"""
    offenders = {
        str(path.relative_to(_PROJECT_ROOT)): sorted(
            module for module in _resolved_imports(path) if module.startswith("mineru.model.flash.pdf")
        )
        for path in (_PROJECT_ROOT / "mineru/model/flash/office").rglob("*.py")
    }
    assert not {path: imports for path, imports in offenders.items() if imports}


def test_flash_spreadsheet_dependencies_are_one_way() -> None:
    """守卫 XLS/XLSX 只依赖中立 spreadsheet 层且共享层不反向引用格式实现。"""
    xls_offenders = {
        str(path.relative_to(_PROJECT_ROOT)): sorted(
            module for module in _resolved_imports(path) if module.startswith("mineru.model.flash.office.xlsx")
        )
        for path in (_PROJECT_ROOT / "mineru/model/flash/office/xls").rglob("*.py")
    }
    spreadsheet_offenders = {
        str(path.relative_to(_PROJECT_ROOT)): sorted(
            module
            for module in _resolved_imports(path)
            if module.startswith(("mineru.model.flash.office.xls", "mineru.model.flash.office.xlsx"))
        )
        for path in (_PROJECT_ROOT / "mineru/model/flash/office/spreadsheet").rglob("*.py")
    }
    assert not {path: imports for path, imports in xls_offenders.items() if imports}
    assert not {path: imports for path, imports in spreadsheet_offenders.items() if imports}


def test_package_initializers_define_explicit_all() -> None:
    """守卫目标目录下每个包入口都显式声明 __all__。"""
    offenders = [
        str(path.relative_to(_PROJECT_ROOT))
        for root in _SOURCE_ROOTS
        for path in root.rglob("__init__.py")
        if "__all__" not in path.read_text(encoding="utf-8")
    ]
    assert not offenders


def test_removed_private_module_paths_have_no_active_references() -> None:
    """守卫严格迁移后的活动生产代码不再引用旧私有路径。"""
    offenders: list[str] = []
    for path in (_PROJECT_ROOT / "mineru").rglob("*.py"):
        if "cli_old" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for module_name in _REMOVED_INTERNAL_MODULES:
            if module_name in text:
                offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{module_name}")
    assert not offenders


def test_mineru_kit_does_not_depend_on_cli_old() -> None:
    """守卫正式 mineru-kit 命令与服务实现不再依赖待删除的 cli_old。"""
    offenders: dict[str, list[str]] = {}
    for path in (_PROJECT_ROOT / "mineru/kit").rglob("*.py"):
        invalid = sorted(module for module in _resolved_imports(path) if module.startswith("mineru.cli_old"))
        if invalid:
            offenders[str(path.relative_to(_PROJECT_ROOT))] = invalid
    assert not offenders


def test_removed_private_module_paths_are_not_importable() -> None:
    """验证严格切换后不存在可被误用的旧私有模块壳。"""
    offenders: list[str] = []
    for module_name in _REMOVED_INTERNAL_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            offenders.append(module_name)
    assert not offenders


def test_public_facade_imports_do_not_load_heavy_dependencies_or_mutate_env() -> None:
    """验证三个稳定门面导入时不加载重依赖或修改环境变量。"""
    script = """
import os
import sys

before_env = dict(os.environ)
import mineru.backend.analyze
import mineru.render
from mineru.model.flash import PdfModel

assert PdfModel.__name__ == "PdfModel"
for prefix in ("torch", "cv2", "pypdfium2", "pdftext", "docx", "bs4", "lxml", "nh3"):
    assert prefix not in sys.modules, prefix
    assert not any(name.startswith(prefix + ".") for name in sys.modules), prefix
assert before_env == dict(os.environ)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
