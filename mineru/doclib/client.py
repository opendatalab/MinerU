"""Synchronous HTTP client implementation of the doclib public interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, TypeVar

import httpx
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ..config import config
from ..errors import MineruError, ServerNotRunningError
from ..types import Tier
from .base import DoclibInterface
from .types import (
    CleanupDeletedRequest,
    CleanupDeletedResponse,
    CleanupOrphansRequest,
    CleanupOrphansResponse,
    CleanupTempRequest,
    CleanupTempResponse,
    ConfigResponse,
    ConfigSetRequest,
    ConfigSetResponse,
    ConfigUnsetResponse,
    ConfigValueResponse,
    DocContentExportRequest,
    DocContentExportResponse,
    DocContentResponse,
    DocInfo,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    ExcludeRuleRequest,
    FileInfoResponse,
    FindResponse,
    ForgetPathRequest,
    ForgetPathResponse,
    InvalidateRequest,
    InvalidateResponse,
    ListDocsResponse,
    ListFilesResponse,
    ListParsesResponse,
    ParseInfo,
    ParseRequest,
    ParseResponse,
    ParseStatus,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParsingRuleRequest,
    RemoveExcludeRuleResponse,
    RemoveParsingRuleResponse,
    RemoveWatchResponse,
    ScanInfo,
    ScanKind,
    ScanListResponse,
    ScanRequest,
    FileStatus,
    ScanStatus,
    SearchResponse,
    ServerStatusResponse,
    ShutdownResponse,
    WatchInfo,
    WatchListResponse,
    WatchRequest,
)
from .utils.route_utils import get_route_info, route

T = TypeVar("T", bound=BaseModel)
_CONFIG_UDS_PATH: Final = object()


class DoclibClient(DoclibInterface):
    """httpx-backed synchronous client for the doclib public interface."""

    def __init__(
        self,
        *,
        base_url: str = "http://mineru",
        socket_path: str | None | object = _CONFIG_UDS_PATH,
        timeout: int = 60,
        api_prefix: str = "/api/v1",
    ) -> None:
        if socket_path is _CONFIG_UDS_PATH:
            socket_path = config.doclib.uds.path
        transport = httpx.HTTPTransport(uds=socket_path) if isinstance(socket_path, str) and socket_path else None
        self._client = httpx.Client(transport=transport, base_url=base_url, timeout=timeout)
        self._api_prefix = api_prefix.rstrip("/")

    def close(self) -> None:
        self._client.close()

    @route("GET", "/server/status", tags=("server",))
    def get_server_status(self) -> ServerStatusResponse:
        return self._request_model(ServerStatusResponse)

    @route("POST", "/server/shutdown", tags=("server",))
    def shutdown_server(self) -> ShutdownResponse:
        return self._request_model(ShutdownResponse)

    @route("POST", "/parses", tags=("parse",))
    def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        return self._request_model(ParseResponse, body=request)

    @route("GET", "/parses", tags=("parse",))
    def list_parses(
        self,
        *,
        ids: list[int] | None = None,
        sha256: str | None = None,
        tier: Tier | None = None,
        status: ParseStatus | None = None,
        page_range: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> ListParsesResponse:
        return self._request_model(
            ListParsesResponse,
            params={
                "ids": ids,
                "sha256": sha256,
                "tier": tier,
                "status": status,
                "page_range": page_range,
                "include_superseded": include_superseded,
                "limit": limit,
                "offset": offset,
            },
        )

    @route("GET", "/parses/{parse_id}", tags=("parse",))
    def get_parse(self, parse_id: int) -> ParseInfo:
        return self._request_model(ParseInfo, path_params={"parse_id": parse_id})

    @route("POST", "/invalidate", tags=("parse",))
    def invalidate(self, request: InvalidateRequest) -> InvalidateResponse:
        return self._request_model(InvalidateResponse, body=request)

    @route("POST", "/forget", tags=("files",))
    def forget_path(self, request: ForgetPathRequest) -> ForgetPathResponse:
        return self._request_model(ForgetPathResponse, body=request)

    @route("POST", "/scans", tags=("scan",))
    def create_scan(self, request: ScanRequest) -> ScanInfo:
        return self._request_model(ScanInfo, body=request)

    @route("GET", "/scans", tags=("scan",))
    def list_scans(
        self,
        *,
        limit: int = 50,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
        offset: int = 0,
    ) -> ScanListResponse:
        return self._request_model(
            ScanListResponse,
            params={"limit": limit, "status": status, "kind": kind, "watch_id": watch_id, "offset": offset},
        )

    @route("GET", "/scans/{scan_id}", tags=("scan",))
    def get_scan(self, scan_id: int) -> ScanInfo:
        return self._request_model(ScanInfo, path_params={"scan_id": scan_id})

    @route("GET", "/files", tags=("files",))
    def list_files(
        self,
        *,
        status: FileStatus | None = None,
        ext: str | None = None,
        watch_id: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListFilesResponse:
        return self._request_model(
            ListFilesResponse,
            params={"status": status, "ext": ext, "watch_id": watch_id, "limit": limit, "offset": offset},
        )

    @route("GET", "/docs", tags=("docs",))
    def list_docs(
        self,
        *,
        file_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListDocsResponse:
        return self._request_model(ListDocsResponse, params={"file_type": file_type, "limit": limit, "offset": offset})

    @route("GET", "/docs/by-path", tags=("docs",))
    def get_doc_by_path(self, path: str) -> DocInfo:
        return self._request_model(DocInfo, params={"path": path})

    @route("GET", "/docs/{sha256}", tags=("docs",))
    def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo:
        return self._request_model(DocInfo, path_params={"sha256": sha256}, params={"expand_files": expand_files})

    @route("GET", "/docs/{sha256}/content", tags=("docs",))
    def get_doc_content(
        self,
        sha256: str,
        *,
        tier: Tier,
        page_range: str | None = None,
        after: str | None = None,
        limit: int = 30000,
        format: str = "markdown",
        no_marker: bool = False,
    ) -> DocContentResponse:
        return self._request_model(
            DocContentResponse,
            path_params={"sha256": sha256},
            params={
                "tier": tier,
                "page_range": page_range,
                "after": after,
                "limit": limit,
                "format": format,
                "no_marker": no_marker,
            },
        )

    @route("POST", "/docs/{sha256}/exports", tags=("docs",))
    def export_doc_content(self, sha256: str, request: DocContentExportRequest) -> DocContentExportResponse:
        return self._request_model(DocContentExportResponse, path_params={"sha256": sha256}, body=request)

    @route("GET", "/search", tags=("search",))
    def search(
        self,
        query: str,
        *,
        file_type: str | None = None,
        tier: Tier | None = None,
        min_tier: Tier | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResponse:
        return self._request_model(
            SearchResponse,
            params={
                "query": query,
                "file_type": file_type,
                "tier": tier,
                "min_tier": min_tier,
                "limit": limit,
                "offset": offset,
            },
        )

    @route("GET", "/find", tags=("search",))
    def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> FindResponse:
        return self._request_model(FindResponse, params={"query": query, "ext": ext, "limit": limit})

    @route("GET", "/files/by-path", tags=("files",))
    def get_file_by_path(self, path: str) -> FileInfoResponse:
        return self._request_model(FileInfoResponse, params={"path": path})

    @route("GET", "/configs", tags=("config",))
    def get_config(self) -> ConfigResponse:
        return self._request_model(ConfigResponse)

    @route("GET", "/configs/{key}", tags=("config",))
    def get_config_key(self, key: str) -> ConfigValueResponse:
        return self._request_model(ConfigValueResponse, path_params={"key": key})

    @route("PUT", "/configs/{key}", tags=("config",))
    def set_config(self, key: str, request: ConfigSetRequest) -> ConfigSetResponse:
        return self._request_model(ConfigSetResponse, path_params={"key": key}, body=request)

    @route("DELETE", "/configs/{key}", tags=("config",))
    def unset_config(self, key: str) -> ConfigUnsetResponse:
        return self._request_model(ConfigUnsetResponse, path_params={"key": key})

    @route("POST", "/watches", tags=("watches",))
    def add_watch(self, request: WatchRequest) -> WatchInfo:
        return self._request_model(WatchInfo, body=request)

    @route("GET", "/watches", tags=("watches",))
    def list_watches(self) -> WatchListResponse:
        return self._request_model(WatchListResponse)

    @route("DELETE", "/watches/{watch_id}", tags=("watches",))
    def remove_watch(self, watch_id: int) -> RemoveWatchResponse:
        return self._request_model(RemoveWatchResponse, path_params={"watch_id": watch_id})

    @route("POST", "/exclude-rules", tags=("rules",))
    def add_exclude_rule(self, request: ExcludeRuleRequest) -> ExcludeRuleInfo:
        return self._request_model(ExcludeRuleInfo, body=request)

    @route("GET", "/exclude-rules", tags=("rules",))
    def list_exclude_rules(self) -> ExcludeRuleListResponse:
        return self._request_model(ExcludeRuleListResponse)

    @route("DELETE", "/exclude-rules/{rule_id}", tags=("rules",))
    def remove_exclude_rule(self, rule_id: int) -> RemoveExcludeRuleResponse:
        return self._request_model(RemoveExcludeRuleResponse, path_params={"rule_id": rule_id})

    @route("POST", "/parsing-rules", tags=("rules",))
    def add_parsing_rule(self, request: ParsingRuleRequest) -> ParsingRuleInfo:
        return self._request_model(ParsingRuleInfo, body=request)

    @route("GET", "/parsing-rules", tags=("rules",))
    def list_parsing_rules(self) -> ParsingRuleListResponse:
        return self._request_model(ParsingRuleListResponse)

    @route("DELETE", "/parsing-rules/{rule_id}", tags=("rules",))
    def remove_parsing_rule(self, rule_id: int) -> RemoveParsingRuleResponse:
        return self._request_model(RemoveParsingRuleResponse, path_params={"rule_id": rule_id})

    @route("POST", "/cleanup/deleted-files", tags=("cleanup",))
    def cleanup_deleted_files(self, request: CleanupDeletedRequest) -> CleanupDeletedResponse:
        return self._request_model(CleanupDeletedResponse, body=request)

    @route("POST", "/cleanup/orphan-docs", tags=("cleanup",))
    def cleanup_orphan_docs(self, request: CleanupOrphansRequest) -> CleanupOrphansResponse:
        return self._request_model(CleanupOrphansResponse, body=request)

    @route("POST", "/cleanup/temp", tags=("cleanup",))
    def cleanup_temp_files(self, request: CleanupTempRequest) -> CleanupTempResponse:
        return self._request_model(CleanupTempResponse, body=request)

    def _request_model(
        self,
        response_model: type[T],
        *,
        path_params: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        body: BaseModel | None = None,
    ) -> T:
        route_info = get_route_info(self._calling_route_method())
        path = self._api_prefix + _format_path(route_info.path, path_params or {})
        query_params = _compact_params(params or {})
        json_data = to_jsonable_python(body) if body is not None else None

        try:
            if route_info.method == "GET":
                resp = self._client.get(path, params=query_params)
            elif route_info.method == "POST":
                resp = self._client.post(path, params=query_params, json=json_data or {})
            elif route_info.method == "PUT":
                resp = self._client.put(path, params=query_params, json=json_data or {})
            elif route_info.method == "DELETE":
                resp = self._client.delete(path, params=query_params)
            else:
                raise MineruError("internal_error", f"Unsupported client method: {route_info.method}")
        except httpx.ConnectError:
            raise ServerNotRunningError() from None

        data = _decode_response(resp)
        return response_model.model_validate(data)

    def _calling_route_method(self) -> Callable[..., Any]:
        import inspect

        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            raise MineruError("internal_error", "Cannot resolve client route method.")
        method_name = frame.f_back.f_back.f_code.co_name
        return getattr(self, method_name)


def _format_path(path: str, path_params: dict[str, Any]) -> str:
    for key, value in path_params.items():
        path = path.replace("{" + key + "}", str(value))
    return path


def _compact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _decode_response(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except Exception:
        resp.raise_for_status()
        raise MineruError("internal_error", f"Invalid server response: HTTP {resp.status_code}") from None
    if not isinstance(data, dict):
        resp.raise_for_status()
        raise MineruError("internal_error", "Invalid server response: expected JSON object")
    if "error" in data:
        error = data["error"]
        if isinstance(error, dict):
            raise MineruError(error.get("code", "internal_error"), error.get("message", ""), error.get("param"))
    resp.raise_for_status()
    return data
