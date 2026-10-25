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
    forbidden = (
        "fastapi",
        "starlette",
        "mineru.doclib.server",
        "mineru.doclib.app",
        "mineru.doclib.services.parse_queue_svc",
        "mineru.doclib.services.parse_execution_svc",
    )
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


def test_ingest_service_does_not_depend_on_parse_execution_or_http() -> None:
    """文件入库拥有独立实现，不反向导入解析服务、HTTP 层或宿主应用。"""
    forbidden = (
        "fastapi",
        "starlette",
        "mineru.doclib.server",
        "mineru.doclib.app",
        "mineru.doclib.services.parse_queue_svc",
        "mineru.doclib.services.parse_execution_svc",
    )
    imports = _imports(_ROOT / "mineru/doclib/services/ingest_svc.py")
    assert not sorted(module for module in imports if module.startswith(forbidden))


def test_file_discovery_workers_depend_on_ingest_service() -> None:
    """扫描和文件事件处理依赖入库边界，解析工作池独立保留解析服务依赖。"""
    for name in ("services/scan_svc.py", "background/watch.py", "background/ingest.py"):
        imports = _imports(_ROOT / "mineru/doclib" / name)
        assert "mineru.doclib.services.ingest_svc.IngestService" in imports, name
        assert not any(
            module.startswith(("mineru.doclib.services.parse_queue_svc", "mineru.doclib.services.parse_execution_svc"))
            for module in imports
        ), name


def test_parse_queue_does_not_import_execution_or_http_clients() -> None:
    """队列和缓存状态处理不反向导入执行器、解析客户端或宿主 HTTP 层。"""
    forbidden = (
        "mineru.doclib.services.parse_execution_svc",
        "mineru.parser.api_client",
        "mineru.parser.base",
        "mineru.doclib.server",
        "mineru.doclib.app",
        "fastapi",
        "httpx",
    )
    imports = _imports(_ROOT / "mineru/doclib/services/parse_queue_svc.py")
    assert not sorted(module for module in imports if module.startswith(forbidden))


def test_parse_execution_is_independent_of_ingestion_and_queue_requests() -> None:
    """执行器只消费已抢占的数据库行，不回调入库或请求排队服务。"""
    forbidden = ("mineru.doclib.services.parse_queue_svc", "mineru.doclib.services.ingest_svc")
    imports = _imports(_ROOT / "mineru/doclib/services/parse_execution_svc.py")
    assert not sorted(module for module in imports if module.startswith(forbidden))


def test_hosted_services_do_not_import_process_defaults() -> None:
    """宿主业务服务使用注入的配置与健康状态，避免再次引入进程默认读取。"""
    forbidden = {"mineru.config.get_config", "mineru.doclib.background.parse_server_health.get_health"}
    for name in ("ingest_svc.py", "parse_queue_svc.py", "parse_execution_svc.py", "parse_policy.py", "content_read_svc.py"):
        imports = _imports(_ROOT / "mineru/doclib/services" / name)
        assert not imports & forbidden, name


def test_health_checker_has_no_implicit_state_or_config_read() -> None:
    """独立日志工具可保留 CLI 默认值，宿主健康检查类不能依赖全局对象。"""
    tree = ast.parse((_ROOT / "mineru/doclib/background/parse_server_health.py").read_text())
    checker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ParseServerHealthCheck")
    calls = {node.func.id for node in ast.walk(checker) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"get_config", "get_health"}
