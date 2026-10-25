"""保护 Doclib 内容服务、产物访问和历史协议读取之间的依赖边界。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    """解析模块及符号导入，包括函数内和类型检查分支中的相对引用。"""
    package = ".".join(path.relative_to(_ROOT).parts[:-1])
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            modules.add(module)
            modules.update(f"{module}.{alias.name}" for alias in node.names)
    return modules


def test_doclib_core_does_not_depend_on_application_services() -> None:
    """产物访问和数据库基础层不得反向借用服务或 HTTP 接口实现。"""
    forbidden = ("mineru.doclib.services", "mineru.doclib.server", "mineru.doclib.app")
    offenders = {
        str(path.relative_to(_ROOT)): sorted(module for module in _imports(path) if module.startswith(forbidden))
        for path in (_ROOT / "mineru/doclib/core").glob("*.py")
    }
    assert not {path: imports for path, imports in offenders.items() if imports}


def test_content_read_service_is_independent_of_http_and_parse_scheduling() -> None:
    """内容读取使用显式数据库和目录，不导入路由、应用状态或解析调度服务。"""
    forbidden = ("fastapi", "starlette", "mineru.doclib.server", "mineru.doclib.app", "mineru.doclib.services.parse_svc")
    imports = _imports(_ROOT / "mineru/doclib/services/content_read_svc.py")
    assert not sorted(module for module in imports if module.startswith(forbidden))


def test_legacy_protocol_conversion_stays_at_cached_middle_json_boundary() -> None:
    """Doclib 仅由指定缓存读取模块进入旧协议转换，不让业务层自行解析旧格式。"""
    boundary = _ROOT / "mineru/doclib/core/middle_json.py"
    consumers = {
        path
        for path in (_ROOT / "mineru/doclib").rglob("*.py")
        if any(module.startswith("mineru.backend.postprocess.legacy_") for module in _imports(path))
    }
    assert consumers == {boundary}
