"""Synchronous HTTP client implementation of the doclib public interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, TypeVar

import httpx
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ..errors import MineruError, ServerNotRunningError
from ..types import Tier
from .base import DoclibInterface
from .endpoint import (
    DOCLIB_UDS_BASE_URL,
    EndpointTransport,
    config_transports,
    default_endpoint_path,
    read_endpoint_file,
    uds_available,
)
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
    ContentFormat,
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
    TelemetryAction,
    TelemetryActionResponse,
    TelemetryObservationsRequest,
    TelemetryObservationsResponse,
    TelemetryPreviewResponse,
    TelemetryStatusResponse,
    WatchInfo,
    WatchListResponse,
    WatchRequest,
)
from .telemetry import get_telemetry_context, infer_default_client_context
from .telemetry.context import TelemetryContext
from .utils.route_utils import get_route_info, route

T = TypeVar("T", bound=BaseModel)
_DISCOVER_ENDPOINT: Final = object()


class DoclibClient(DoclibInterface):
    """httpx-backed synchronous client for the doclib public interface."""

    def __init__(
        self,
        *,
        endpoint_path: str | Path | None = None,
        socket_path: str | Path | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        api_prefix: str = "/api/v1",
    ) -> None:
        if socket_path is not None and base_url is not None:
            raise ValueError("socket_path and base_url cannot both be provided.")
        self._clients = _build_clients(
            endpoint_path=endpoint_path,
            socket_path=socket_path,
            base_url=base_url,
            timeout=timeout,
        )
        self._active_client: httpx.Client | None = self._clients[0] if self._clients else None
        self._api_prefix = api_prefix.rstrip("/")
        self._default_telemetry_context = infer_default_client_context(source="sdk")

    def close(self) -> None:
        for client in self._clients:
            client.close()

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
        doc_ref: str | None = None,
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
                "doc_ref": doc_ref,
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

    @route("GET", "/docs/{doc_ref}", tags=("docs",))
    def get_doc(self, doc_ref: str, *, expand_files: bool = False) -> DocInfo:
        return self._request_model(DocInfo, path_params={"doc_ref": doc_ref}, params={"expand_files": expand_files})

    @route("GET", "/docs/{doc_ref}/content", tags=("docs",))
    def get_doc_content(
        self,
        doc_ref: str,
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
            path_params={"doc_ref": doc_ref},
            params={
                "tier": tier,
                "page_range": page_range,
                "after": after,
                "limit": limit,
                "format": format,
                "no_marker": no_marker,
            },
        )

    @route("GET", "/content", tags=("docs",))
    def read_content(
        self,
        locator: str,
        *,
        context: int = 0,
        limit: int = 30000,
        format: ContentFormat = "markdown",
        no_marker: bool = False,
    ) -> DocContentResponse:
        return self._request_model(
            DocContentResponse,
            params={
                "locator": locator,
                "context": context,
                "limit": limit,
                "format": format,
                "no_marker": no_marker,
            },
        )

    @route("POST", "/docs/{doc_ref}/exports", tags=("docs",))
    def export_doc_content(self, doc_ref: str, request: DocContentExportRequest) -> DocContentExportResponse:
        return self._request_model(DocContentExportResponse, path_params={"doc_ref": doc_ref}, body=request)

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

    @route("GET", "/telemetry/status", tags=("telemetry",))
    def get_telemetry_status(self) -> TelemetryStatusResponse:
        return self._request_model(TelemetryStatusResponse)

    @route("GET", "/telemetry/preview", tags=("telemetry",))
    def get_telemetry_preview(self) -> TelemetryPreviewResponse:
        return self._request_model(TelemetryPreviewResponse)

    @route("POST", "/telemetry/actions/{action}", tags=("telemetry",))
    def telemetry_action(self, action: TelemetryAction) -> TelemetryActionResponse:
        return self._request_model(TelemetryActionResponse, path_params={"action": action})

    @route("POST", "/observations", tags=("telemetry",))
    def record_observations(self, request: TelemetryObservationsRequest) -> TelemetryObservationsResponse:
        return self._request_model(TelemetryObservationsResponse, body=request)

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

        resp = self._send_request(
            route_info.method,
            path,
            params=query_params,
            json_data=json_data,
        )

        data = _decode_response(resp)
        return response_model.model_validate(data)

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        json_data: Any,
    ) -> httpx.Response:
        if not self._clients:
            raise ServerNotRunningError()
        clients = self._ordered_clients()
        for client in clients:
            try:
                if method == "GET":
                    resp = client.get(path, params=params, headers=self._telemetry_headers())
                elif method == "POST":
                    resp = client.post(path, params=params, json=json_data or {}, headers=self._telemetry_headers())
                elif method == "PUT":
                    resp = client.put(path, params=params, json=json_data or {}, headers=self._telemetry_headers())
                elif method == "DELETE":
                    resp = client.delete(path, params=params, headers=self._telemetry_headers())
                else:
                    raise MineruError("internal_error", f"Unsupported client method: {method}")
            except httpx.ConnectError:
                continue
            self._active_client = client
            return resp
        raise ServerNotRunningError() from None

    def _ordered_clients(self) -> list[httpx.Client]:
        if self._active_client is None:
            return list(self._clients)
        return [self._active_client, *[client for client in self._clients if client is not self._active_client]]

    def _telemetry_headers(self) -> dict[str, str]:
        ctx = get_telemetry_context()
        if ctx == TelemetryContext():
            ctx = self._default_telemetry_context
        return {
            "X-MinerU-Telemetry-Source": ctx.source,
            "X-MinerU-Telemetry-Caller": ctx.caller,
        }

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


def _build_clients(
    *,
    endpoint_path: str | Path | None,
    socket_path: str | Path | None,
    base_url: str | None,
    timeout: int,
) -> list[httpx.Client]:
    if socket_path is not None:
        client = _client_for_transport(EndpointTransport(type="uds", path=str(socket_path)), timeout=timeout)
        return [client] if client is not None else []
    if base_url is not None:
        client = _client_for_transport(EndpointTransport(type="tcp", base_url=base_url), timeout=timeout)
        return [client] if client is not None else []

    discovery_path = str(endpoint_path) if endpoint_path is not None else default_endpoint_path()
    transports = read_endpoint_file(discovery_path)
    if not transports:
        transports = config_transports()

    clients: list[httpx.Client] = []
    for transport in sorted(transports, key=lambda item: 0 if item.type == "uds" else 1):
        client = _client_for_transport(transport, timeout=timeout)
        if client is not None:
            clients.append(client)
    return clients


def _client_for_transport(transport: EndpointTransport, *, timeout: int) -> httpx.Client | None:
    if transport.type == "uds":
        if not transport.path or not uds_available():
            return None
        return httpx.Client(
            transport=httpx.HTTPTransport(uds=transport.path),
            base_url=DOCLIB_UDS_BASE_URL,
            timeout=timeout,
            trust_env=False,
        )
    if not transport.base_url:
        return None
    return httpx.Client(base_url=transport.base_url, timeout=timeout, trust_env=False)


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
