import asyncio
import inspect
from typing import Any, cast, get_type_hints

import pytest
from pydantic import ValidationError

from mineru.config import PatchedConfig
from mineru.errors import InvalidRequestError
from mineru.doclib.app import create_app
from mineru.doclib.base import AsyncDoclibInterface, DoclibInterface
from mineru.doclib.client import DoclibClient
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.server import DoclibServer
from mineru.doclib.services.cleanup_svc import CleanupService
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.types import (
    ConfigResponse,
    DocInfo,
    DocContentExportRequest,
    ErrorBucket,
    ErrorInfo,
    ErrorResponse,
    ErrorSummary,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    ExcludeRuleRequest,
    FileInfo,
    FileInfoResponse,
    ListDocsResponse,
    ListFilesResponse,
    ListParsesResponse,
    FindResponse,
    FindResult,
    ForgetPathRequest,
    ForgetPathResponse,
    LocalParseServerStatus,
    ParseCoverage,
    ParseInfo,
    ParseRequest,
    ParseResponse,
    ParseStatus,
    ParseServerStatus,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParsingRuleRequest,
    RemoteParseServerStatus,
    RemoveExcludeRuleResponse,
    RemoveParsingRuleResponse,
    RemoveWatchResponse,
    SearchResponse,
    SearchResult,
    ServerStatusResponse,
    ShutdownResponse,
    ScanInfo,
    ScanListResponse,
    ScanRequest,
    TierParseInfo,
    WatchInfo,
    WatchListResponse,
    WatchStats,
)
from mineru.doclib.utils.route_utils import RouteInfo


def test_doclib_interface_declares_expected_methods() -> None:
    expected = {
        "get_server_status",
        "shutdown_server",
        "ensure_parse",
        "list_parses",
        "get_parse",
        "invalidate",
        "forget_path",
        "create_scan",
        "list_scans",
        "get_scan",
        "list_files",
        "list_docs",
        "get_doc_by_path",
        "get_doc",
        "get_doc_content",
        "read_content",
        "export_doc_content",
        "search",
        "find",
        "get_file_by_path",
        "get_config",
        "get_config_key",
        "set_config",
        "unset_config",
        "add_watch",
        "list_watches",
        "remove_watch",
        "add_exclude_rule",
        "list_exclude_rules",
        "remove_exclude_rule",
        "add_parsing_rule",
        "list_parsing_rules",
        "remove_parsing_rule",
        "cleanup_deleted_files",
        "cleanup_orphan_docs",
        "cleanup_temp_files",
        "get_telemetry_status",
        "get_telemetry_preview",
        "telemetry_action",
        "record_observations",
    }

    assert DoclibInterface.__abstractmethods__ == expected


def test_sync_and_async_doclib_interfaces_stay_aligned() -> None:
    assert AsyncDoclibInterface.__abstractmethods__ == DoclibInterface.__abstractmethods__

    for method_name in sorted(DoclibInterface.__abstractmethods__):
        sync_method = getattr(DoclibInterface, method_name)
        async_method = getattr(AsyncDoclibInterface, method_name)

    assert inspect.signature(async_method) == inspect.signature(sync_method), method_name
    assert not inspect.iscoroutinefunction(sync_method), method_name
    assert inspect.iscoroutinefunction(async_method), method_name


def test_doclib_server_implements_async_interface_with_route_metadata() -> None:
    assert DoclibServer.__abstractmethods__ == frozenset()

    for method_name in sorted(AsyncDoclibInterface.__abstractmethods__):
        interface_method = getattr(AsyncDoclibInterface, method_name)
        server_method = getattr(DoclibServer, method_name)

        assert inspect.signature(server_method) == inspect.signature(interface_method), method_name
        assert inspect.iscoroutinefunction(server_method), method_name
        assert hasattr(server_method, "_route_info"), method_name


def test_doclib_client_implements_sync_interface_with_route_metadata() -> None:
    assert DoclibClient.__abstractmethods__ == frozenset()

    for method_name in sorted(DoclibInterface.__abstractmethods__):
        interface_method = getattr(DoclibInterface, method_name)
        client_method = getattr(DoclibClient, method_name)

        assert inspect.signature(client_method) == inspect.signature(interface_method), method_name
        assert not inspect.iscoroutinefunction(client_method), method_name
        assert hasattr(client_method, "_route_info"), method_name


def test_doclib_client_and_server_route_maps_stay_aligned() -> None:
    for method_name in sorted(DoclibInterface.__abstractmethods__):
        client_route = getattr(DoclibClient, method_name)._route_info
        server_route = getattr(DoclibServer, method_name)._route_info

        assert client_route.method == server_route.method, method_name
        assert client_route.path == server_route.path, method_name
        assert client_route.tags == server_route.tags, method_name


def test_list_parses_status_filter_is_typed() -> None:
    expected = ParseStatus | None

    assert get_type_hints(DoclibInterface.list_parses)["status"] == expected
    assert get_type_hints(AsyncDoclibInterface.list_parses)["status"] == expected
    assert get_type_hints(DoclibClient.list_parses)["status"] == expected
    assert get_type_hints(DoclibServer.list_parses)["status"] == expected


def test_list_methods_expose_consistent_pagination_contract() -> None:
    for response_model in (ListFilesResponse, ListDocsResponse, ListParsesResponse, ScanListResponse):
        assert "total" in response_model.model_fields, response_model.__name__
        assert "limit" in response_model.model_fields, response_model.__name__
        assert "offset" in response_model.model_fields, response_model.__name__

    for method_name in ("list_files", "list_docs", "list_parses", "list_scans"):
        signature = inspect.signature(getattr(DoclibInterface, method_name))
        assert "limit" in signature.parameters, method_name
        assert "offset" in signature.parameters, method_name


def test_interface_app_uses_doclib_server_routes(tmp_path) -> None:
    cfg = PatchedConfig(doclib={"log": {"dir": str(tmp_path / "logs")}})

    app = create_app(cfg)
    route_names = {getattr(route, "name", "") for route in app.routes}

    for method_name in AsyncDoclibInterface.__abstractmethods__:
        assert method_name in route_names

    route_paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/v1/server/status" in route_paths
    assert "/api/v1/server/shutdown" in route_paths
    assert "/api/v1/telemetry/status" in route_paths
    assert "/api/v1/telemetry/preview" in route_paths
    assert "/api/v1/telemetry/actions/{action}" in route_paths
    assert "/api/v1/observations" in route_paths
    assert "/api/v1/configs" in route_paths
    assert "/api/v1/configs/{key}" in route_paths
    assert "/api/v1/watches" in route_paths
    assert "/api/v1/watches/{watch_id}" in route_paths
    assert "/api/v1/exclude-rules" in route_paths
    assert "/api/v1/exclude-rules/{rule_id}" in route_paths
    assert "/api/v1/parsing-rules" in route_paths
    assert "/api/v1/parsing-rules/{rule_id}" in route_paths
    assert "/api/v1/files" in route_paths
    assert "/api/v1/docs" in route_paths
    assert "/api/v1/docs/{doc_ref}" in route_paths
    assert "/api/v1/content" in route_paths
    assert "/api/v1/docs/{doc_ref}/exports" in route_paths
    assert "/api/docs" in route_paths
    assert "/docs" not in route_paths
    assert "/api/v1/config" not in route_paths
    assert "/api/v1/config/watch" not in route_paths
    assert "/api/v1/config/exclude" not in route_paths
    assert "/api/v1/config/parsing-rules" not in route_paths
    assert "/api/v1/shutdown" not in route_paths

    assert "create_parses" not in route_names
    assert "parse_content" not in route_names


def test_add_watch_rejects_missing_directory(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        with pytest.raises(InvalidRequestError, match="Watch path does not exist"):
            await service.add_watch(str(tmp_path / "not-exist"))

    asyncio.run(_run())


def test_add_watch_normalizes_user_path_before_storing(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        root = tmp_path / "root"
        root.mkdir()
        unnormalized = str(root / ".." / root.name)

        watch = await service.add_watch(unnormalized)

        assert watch["path"] == str(root)

    asyncio.run(_run())


def test_config_service_rejects_unsupported_rule_type_with_structured_error(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ConfigService(db)

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.list_rules(cast(Any, "bad-rule-type"))

        assert exc_info.value.code == "invalid_request"
        assert exc_info.value.param == "rule_type"

    asyncio.run(_run())


def test_forget_watch_root_is_allowed_with_warning(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        watch_root = tmp_path / "watch-root"
        watch_root.mkdir()
        await ConfigService(db).add_watch(str(watch_root))
        service = CleanupService(db, str(tmp_path))

        result = await service.forget_path(str(watch_root), dry_run=True)

        assert result["path"] == str(watch_root)
        assert result["matched_as"] == "none"
        assert result["forgotten_files"] == 0
        assert result["warnings"] == ["Path is a configured watch root and may be rediscovered on the next scan."]

    asyncio.run(_run())


def test_cleanup_temp_rejects_negative_older_than(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = CleanupService(db, str(tmp_path))

        with pytest.raises(InvalidRequestError, match="older_than_days must be non-negative"):
            await service.cleanup_temp_files(-1)

    asyncio.run(_run())


def test_core_doclib_schemas_are_instantiable() -> None:
    parse_request = ParseRequest(path="/tmp/a.pdf", tier=None, page_range="1~5")
    forget_request = ForgetPathRequest(path="/tmp/a.pdf")
    forget_response = ForgetPathResponse(path="/tmp/a.pdf", matched_as="file", forgotten_files=1, dry_run=True)
    scan_request = ScanRequest(path="/tmp", kind="manual", source="cli")
    scan_info = ScanInfo(id=1, path="/tmp", kind="manual", source="cli", status="pending", created_at=10, updated_at=10)
    recent_scan = ScanInfo(
        id=1,
        path="/tmp",
        kind="manual",
        source="cli",
        status="done",
        error_msg="scan detail",
        created_at=10,
        updated_at=10,
    )
    scan_list = ScanListResponse(scans=[scan_info], total=1, limit=50, offset=0)
    watch_stats = WatchStats(
        watch_id=1,
        path="/tmp",
        status="active",
        total_files=3,
        active_files=2,
        deleted_files=1,
        doc_count=2,
    )
    error_summary = ErrorSummary(file_errors=[ErrorBucket(code="stat_failed", count=1)])
    status = ServerStatusResponse(
        running=True,
        socket_path="/tmp/doclib.sock",
        data_dir="/tmp/mineru",
        sqlite_path="/tmp/doclib.db",
        log_path="/tmp/doclib.log",
        watch_stats=[watch_stats],
        recent_scans=[recent_scan],
        error_summary=error_summary,
    )
    file_info = FileInfo(
        path="/tmp/a.pdf",
        filename="a.pdf",
        ext=".pdf",
        size_bytes=123,
        mtime_ms=11,
        sha256="abc",
        short_id="abc",
        watch_id=1,
        status="active",
        error_code="scan_failed",
        error_msg="failed",
        first_seen_at=20,
        updated_at=30,
        deleted_at=None,
    )
    files = ListFilesResponse(files=[file_info], total=1, limit=200, offset=0)
    doc_info = DocInfo(
        sha256="abc",
        short_id="abc",
        size_bytes=123,
        file_type="pdf",
        subject="demo",
        keywords="a,b",
        page_count=3,
        is_image_based=0,
        meta_tier="high",
        first_seen_at=40,
        updated_at=50,
        files=[file_info],
    )
    coverage = ParseCoverage(done_page_range="1", active_page_range="2", missing_page_range="")
    parse_info = ParseInfo(
        id=1,
        sha256="abc",
        short_id="abc",
        tier="high",
        page_range="1~2",
        status="pending",
        priority=10,
        privacy="local",
        via="local",
        coverage=coverage,
        created_at=60,
        updated_at=70,
        error_code="parse_failed",
        error_msg="failed",
    )
    tier_info = TierParseInfo(tier="high", page_range="1~2", status="pending")
    info = FileInfoResponse(
        file=file_info,
        doc=doc_info,
        parsed_tiers=[tier_info],
        active_parses=[parse_info],
    )
    watch = WatchInfo(id=1, path="/tmp", recursive=True, status="active", last_scan_at=100, last_scan_files=3)
    exclude_rule = ExcludeRuleInfo(id=2, pattern="*.tmp", hit_count=4)
    parsing_rule = ParsingRuleInfo(id=3, pattern="*.pdf", tier="high", page_range="1~5", remote=True)
    watches = WatchListResponse(watches=[watch])
    exclude_rules = ExcludeRuleListResponse(rules=[exclude_rule])
    parsing_rules = ParsingRuleListResponse(rules=[parsing_rule])
    parse_server = ParseServerStatus(
        local=LocalParseServerStatus(
            mode="managed",
            healthy=True,
            url="http://127.0.0.1:16580",
            port=16580,
            managed_pid=1234,
            last_probe_at=1000,
            last_success_at=900,
            last_failure_at=800,
            supported_tiers=["high"],
        ),
        remote=RemoteParseServerStatus(
            healthy=False,
            url="https://example.com/api",
            port=443,
            last_probe_at=2000,
            last_success_at=1900,
            last_failure_at=1800,
        ),
    )
    search_result = SearchResult(
        sha256="abc",
        short_id="abc",
        filename="a.pdf",
        tier="high",
        snippet="matched text",
        paths=["/tmp/a.pdf"],
    )
    find_result = FindResult(filename="a.pdf", ext="pdf", size_bytes=123, page_count=3, paths=["/tmp/a.pdf"])
    search_response = SearchResponse(results=[search_result], total=1, query="matched")
    find_response = FindResponse(results=[find_result], total=1, query="a")
    config = ConfigResponse(config={"parse_server.local.mode": "managed"}, sources={"parse_server.local.mode": "override"})
    shutdown_response = ShutdownResponse(accepted=True, message="Server shutting down...")
    remove_watch_response = RemoveWatchResponse(watch_id=1, removed=True)
    export_request = DocContentExportRequest(tier="high", output="/tmp/a.md")
    remove_exclude_response = RemoveExcludeRuleResponse(rule_id=2, removed=True)
    remove_parsing_response = RemoveParsingRuleResponse(rule_id=3, removed=True)
    exclude_rule_request = ExcludeRuleRequest(pattern="*.tmp")
    parsing_rule_request = ParsingRuleRequest(pattern="*.pdf", tier="high", page_range="1~5")
    error_info = ErrorInfo(type="invalid_request_error", code="file_not_found", message="missing", param="path")
    error_response = ErrorResponse(error=error_info)

    assert parse_request.path == "/tmp/a.pdf"
    assert forget_request.dry_run is True
    assert forget_response.matched_as == "file"
    assert scan_request.kind == "manual"
    assert scan_list.scans[0].id == 1
    assert files.files == [file_info]
    assert files.total == 1
    assert "format" not in ParseRequest.model_fields
    assert status.files_total == 0
    assert status.watch_stats == [watch_stats]
    assert status.recent_scans == [recent_scan]
    assert status.error_summary == error_summary
    assert status.recent_scans[0].error_msg == "scan detail"
    assert doc_info.files == [file_info]
    assert doc_info.is_image_based is False
    assert info.file == file_info
    assert info.doc == doc_info
    assert info.parsed_tiers == [tier_info]
    assert info.active_parses == [parse_info]
    assert parse_info.coverage == coverage
    assert parse_info.error_code == "parse_failed"
    assert parse_info.error_msg == "failed"
    assert parse_info.priority == 10
    assert parse_info.created_at == 60
    assert file_info.error_code == "scan_failed"
    assert file_info.updated_at == 30
    assert doc_info.file_type == "pdf"
    assert doc_info.first_seen_at == 40
    assert watches.watches == [watch]
    assert exclude_rules.rules == [exclude_rule]
    assert parsing_rules.rules == [parsing_rule]
    assert watch.recursive
    assert exclude_rule.hit_count == 4
    assert parsing_rule.remote
    assert parse_server.local.mode == "managed"
    assert search_response.results == [search_result]
    assert find_response.results == [find_result]
    assert "sha256" not in FindResult.model_fields
    assert "snippet" not in FindResult.model_fields
    assert config.config == {"parse_server.local.mode": "managed"}
    assert config.sources == {"parse_server.local.mode": "override"}
    assert shutdown_response.accepted
    assert remove_watch_response.watch_id == 1
    assert export_request.output == "/tmp/a.md"
    assert remove_exclude_response.rule_id == 2
    assert remove_parsing_response.rule_id == 3
    assert exclude_rule_request.pattern == "*.tmp"
    assert parsing_rule_request.tier == "high"
    assert error_response.error.code == "file_not_found"


def test_parse_request_rejects_auto_tier() -> None:
    with pytest.raises(ValidationError):
        ParseRequest(path="/tmp/a.pdf", tier="auto", page_range="1~5")


def test_parse_response_status_is_submit_state_only() -> None:
    with pytest.raises(ValidationError):
        ParseResponse(sha256="a" * 64, tier="flash", page_range="1", status="parsing")

    response = ParseResponse(sha256="a" * 64, tier="flash", page_range="1", status="pending")

    assert response.status == "pending"


def test_route_info_exports_from_route_utils() -> None:
    route_info = RouteInfo(method="GET", path="/server/status")
    assert route_info.method == "GET"


def test_get_semantic_methods_do_not_use_request_body_models() -> None:
    request_body_names = {
        "request",
        "req",
    }
    get_semantic_methods = [
        "list_parses",
        "list_scans",
        "list_files",
        "list_docs",
        "get_doc_by_path",
        "get_doc_content",
        "read_content",
        "search",
        "find",
        "get_file_by_path",
    ]

    for method_name in get_semantic_methods:
        signature = inspect.signature(getattr(DoclibInterface, method_name))
        assert request_body_names.isdisjoint(signature.parameters)


def test_get_doc_expands_files_only_when_requested() -> None:
    signature = inspect.signature(DoclibInterface.get_doc)
    assert "expand_files" in signature.parameters
    assert signature.parameters["expand_files"].default is False
    assert "path" not in inspect.signature(DoclibInterface.list_docs).parameters


def test_next_cli_doclib_route_shapes_are_declared() -> None:
    expected_routes = {
        "shutdown_server": ("POST", "/server/shutdown"),
        "list_parses": ("GET", "/parses"),
        "list_files": ("GET", "/files"),
        "get_file_by_path": ("GET", "/files/by-path"),
        "list_docs": ("GET", "/docs"),
        "get_doc_by_path": ("GET", "/docs/by-path"),
        "get_doc": ("GET", "/docs/{doc_ref}"),
        "get_doc_content": ("GET", "/docs/{doc_ref}/content"),
        "read_content": ("GET", "/content"),
        "export_doc_content": ("POST", "/docs/{doc_ref}/exports"),
        "get_config": ("GET", "/configs"),
        "get_config_key": ("GET", "/configs/{key}"),
        "set_config": ("PUT", "/configs/{key}"),
        "unset_config": ("DELETE", "/configs/{key}"),
        "add_watch": ("POST", "/watches"),
        "list_watches": ("GET", "/watches"),
        "remove_watch": ("DELETE", "/watches/{watch_id}"),
        "add_exclude_rule": ("POST", "/exclude-rules"),
        "list_exclude_rules": ("GET", "/exclude-rules"),
        "remove_exclude_rule": ("DELETE", "/exclude-rules/{rule_id}"),
        "add_parsing_rule": ("POST", "/parsing-rules"),
        "list_parsing_rules": ("GET", "/parsing-rules"),
        "remove_parsing_rule": ("DELETE", "/parsing-rules/{rule_id}"),
    }

    for method_name, (method, path) in expected_routes.items():
        client_route = getattr(DoclibClient, method_name)._route_info
        server_route = getattr(DoclibServer, method_name)._route_info
        assert client_route.method == method, method_name
        assert client_route.path == path, method_name
        assert server_route.method == method, method_name
        assert server_route.path == path, method_name

    assert "output" not in inspect.signature(DoclibInterface.get_doc_content).parameters
    assert "output" not in inspect.signature(DoclibInterface.read_content).parameters
    assert "limit" in inspect.signature(DoclibInterface.list_parses).parameters
    assert "limit" in inspect.signature(DoclibInterface.list_docs).parameters
    assert "file_type" in inspect.signature(DoclibInterface.list_docs).parameters
    assert "watch_id" in inspect.signature(DoclibInterface.remove_watch).parameters
    assert "path" not in inspect.signature(DoclibInterface.remove_watch).parameters
