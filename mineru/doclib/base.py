"""Synchronous doclib public interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Tier
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
    FileStatus,
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
    ScanStatus,
    SearchResponse,
    ServerStatusResponse,
    ShutdownResponse,
    WatchInfo,
    WatchListResponse,
    WatchRequest,
)


class DoclibInterface(ABC):
    """Synchronous contract for the doclib SDK client and sync implementations.

    Implementations must raise ``MineruError`` subclasses for contract-level
    failures. HTTP implementations serialize those failures as ``ErrorResponse``
    and SDK clients translate them back to ``MineruError``.
    """

    @abstractmethod
    def get_server_status(self) -> ServerStatusResponse:
        """Return local doclib server status.

        Raises:
            ServerNotRunningError: when an SDK client cannot connect to the local server.
            MineruError: for invalid or non-JSON server responses.
        """
        raise NotImplementedError()

    @abstractmethod
    def shutdown_server(self) -> ShutdownResponse:
        """Request graceful doclib server shutdown.

        Raises:
            ServerNotRunningError: when an SDK client cannot connect to the local server.
            MineruError: for server-side shutdown failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        """Ensure parse records exist for a document request.

        Raises:
            InvalidRequestError: for invalid paths, invalid page ranges, unsupported file types, or invalid tier values.
            NotFoundError: when the source file cannot be found.
            PermissionError_: when the source file cannot be read.
            EngineError: when the requested quality tier cannot be served or parsing cannot be queued.
            MineruError: for other server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
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
        """List parse records and optional coverage information.

        Raises:
            InvalidRequestError: for malformed ids, status filters, page ranges, or tier filters.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_parse(self, parse_id: int) -> ParseInfo:
        """Return one parse record.

        Raises:
            NotFoundError: when ``parse_id`` does not exist.
            InvalidRequestError: when ``parse_id`` is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def invalidate(self, request: InvalidateRequest) -> InvalidateResponse:
        """Invalidate cached doclib artifacts.

        Raises:
            InvalidRequestError: when the target, path, sha256, or tier is invalid.
            NotFoundError: when the target document cannot be resolved.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def forget_path(self, request: ForgetPathRequest) -> ForgetPathResponse:
        """Forget file rows matching a file or directory path.

        Raises:
            InvalidRequestError: when the path is malformed or is a configured watch root.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_scan(self, request: ScanRequest) -> ScanInfo:
        """Create or reuse a server-side scan task.

        Raises:
            InvalidRequestError: when the path, kind, source, or watch id is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_scans(
        self,
        *,
        limit: int = 50,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
        offset: int = 0,
    ) -> ScanListResponse:
        """List recent scan tasks.

        Raises:
            InvalidRequestError: when filters are invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_scan(self, scan_id: int) -> ScanInfo:
        """Return one scan task.

        Raises:
            NotFoundError: when ``scan_id`` does not exist.
            InvalidRequestError: when ``scan_id`` is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_files(
        self,
        *,
        status: FileStatus | None = None,
        ext: str | None = None,
        watch_id: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListFilesResponse:
        """List file rows known to doclib.

        Raises:
            InvalidRequestError: when filters, limit, or offset are invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_docs(
        self,
        *,
        file_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListDocsResponse:
        """List active docs, optionally filtered by file type.

        Raises:
            InvalidRequestError: when filters or limit are invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_doc_by_path(self, path: str) -> DocInfo:
        """Return the current doc bound to a local file path.

        Raises:
            NotFoundError: when the path is unknown, not active, or not currently bound to a doc.
            InvalidRequestError: when the path is malformed.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo:
        """Return one doc by content hash.

        Raises:
            NotFoundError: when ``sha256`` does not exist in doclib.
            InvalidRequestError: when ``sha256`` is malformed.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
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
        """Render stored doc content from parsed JSON artifacts.

        Raises:
            NotFoundError: when the document or requested parsed content is not cached.
            InvalidRequestError: when tier, page_range, format, or marker options are invalid.
            MineruError: for render or server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def export_doc_content(self, sha256: str, request: DocContentExportRequest) -> DocContentExportResponse:
        """Render stored doc content and write it to a server-visible output path.

        Raises:
            NotFoundError: when the document or requested parsed content is not cached.
            InvalidRequestError: when tier, page_range, format, output, or marker options are invalid.
            MineruError: for render, filesystem, or server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
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
        """Search parsed document content.

        Raises:
            InvalidRequestError: when query, filters, limit, or offset are invalid.
            MineruError: for search backend failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> FindResponse:
        """Search filenames.

        Raises:
            InvalidRequestError: when query, ext, or limit is invalid.
            MineruError: for search backend failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_file_by_path(self, path: str) -> FileInfoResponse:
        """Return file, doc, and parse state for a local path.

        Raises:
            NotFoundError: when the path is unknown to doclib.
            InvalidRequestError: when the path is malformed.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_config(self) -> ConfigResponse:
        """Return runtime doclib key-value config.

        Raises:
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_config_key(self, key: str) -> ConfigValueResponse:
        """Return one runtime config key.

        Raises:
            InvalidRequestError: when the key is unknown.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_config(self, key: str, request: ConfigSetRequest) -> ConfigSetResponse:
        """Set one runtime config key.

        Raises:
            InvalidRequestError: when the key is unknown, read-only, or the value is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def unset_config(self, key: str) -> ConfigUnsetResponse:
        """Remove one runtime config override and return the resulting default-backed value.

        Raises:
            InvalidRequestError: when the key is unknown.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_watch(self, request: WatchRequest) -> WatchInfo:
        """Add a watch target.

        Raises:
            InvalidRequestError: when the path is not absolute, not normalized, or otherwise invalid.
            NotFoundError: when the path does not exist.
            PermissionError_: when the path cannot be read.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_watches(self) -> WatchListResponse:
        """List watch targets.

        Raises:
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_watch(self, watch_id: int) -> RemoveWatchResponse:
        """Remove a watch target.

        Raises:
            NotFoundError: when the watch target does not exist.
            InvalidRequestError: when ``watch_id`` is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_exclude_rule(self, request: ExcludeRuleRequest) -> ExcludeRuleInfo:
        """Add an exclude rule.

        Raises:
            InvalidRequestError: when the pattern, priority, or name is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_exclude_rules(self) -> ExcludeRuleListResponse:
        """List exclude rules.

        Raises:
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_exclude_rule(self, rule_id: int) -> RemoveExcludeRuleResponse:
        """Remove an exclude rule.

        Raises:
            NotFoundError: when the rule does not exist.
            InvalidRequestError: when ``rule_id`` is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_parsing_rule(self, request: ParsingRuleRequest) -> ParsingRuleInfo:
        """Add a parsing rule.

        Raises:
            InvalidRequestError: when the pattern, tier, page_range, priority, or name is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_parsing_rules(self) -> ParsingRuleListResponse:
        """List parsing rules.

        Raises:
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_parsing_rule(self, rule_id: int) -> RemoveParsingRuleResponse:
        """Remove a parsing rule.

        Raises:
            NotFoundError: when the rule does not exist.
            InvalidRequestError: when ``rule_id`` is invalid.
            MineruError: for server-side failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_deleted_files(self, request: CleanupDeletedRequest) -> CleanupDeletedResponse:
        """Clean DB/file entries for deleted source files.

        Raises:
            InvalidRequestError: when cleanup options are invalid.
            MineruError: for cleanup failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_orphan_docs(self, request: CleanupOrphansRequest) -> CleanupOrphansResponse:
        """Clean orphan doc artifacts.

        Raises:
            InvalidRequestError: when cleanup options are invalid.
            MineruError: for cleanup failures.
        """
        raise NotImplementedError()

    @abstractmethod
    def cleanup_temp_files(self, request: CleanupTempRequest) -> CleanupTempResponse:
        """Clean temporary files.

        Raises:
            InvalidRequestError: when cleanup options are invalid.
            MineruError: for cleanup failures.
        """
        raise NotImplementedError()


class AsyncDoclibInterface(ABC):
    """Asynchronous contract matching ``DoclibInterface`` method-for-method.

    The async interface has the same method names, parameters, return
    annotations, and error contract as ``DoclibInterface``. Implementations must
    raise ``MineruError`` subclasses for contract-level failures.
    """

    @abstractmethod
    async def get_server_status(self) -> ServerStatusResponse:
        """Async version of ``DoclibInterface.get_server_status``."""
        raise NotImplementedError()

    @abstractmethod
    async def shutdown_server(self) -> ShutdownResponse:
        """Async version of ``DoclibInterface.shutdown_server``."""
        raise NotImplementedError()

    @abstractmethod
    async def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        """Async version of ``DoclibInterface.ensure_parse``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_parses(
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
        """Async version of ``DoclibInterface.list_parses``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_parse(self, parse_id: int) -> ParseInfo:
        """Async version of ``DoclibInterface.get_parse``."""
        raise NotImplementedError()

    @abstractmethod
    async def invalidate(self, request: InvalidateRequest) -> InvalidateResponse:
        """Async version of ``DoclibInterface.invalidate``."""
        raise NotImplementedError()

    @abstractmethod
    async def forget_path(self, request: ForgetPathRequest) -> ForgetPathResponse:
        """Async version of ``DoclibInterface.forget_path``."""
        raise NotImplementedError()

    @abstractmethod
    async def create_scan(self, request: ScanRequest) -> ScanInfo:
        """Async version of ``DoclibInterface.create_scan``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_scans(
        self,
        *,
        limit: int = 50,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
        offset: int = 0,
    ) -> ScanListResponse:
        """Async version of ``DoclibInterface.list_scans``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_scan(self, scan_id: int) -> ScanInfo:
        """Async version of ``DoclibInterface.get_scan``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_files(
        self,
        *,
        status: FileStatus | None = None,
        ext: str | None = None,
        watch_id: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListFilesResponse:
        """Async version of ``DoclibInterface.list_files``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_docs(
        self,
        *,
        file_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListDocsResponse:
        """Async version of ``DoclibInterface.list_docs``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_doc_by_path(self, path: str) -> DocInfo:
        """Async version of ``DoclibInterface.get_doc_by_path``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo:
        """Async version of ``DoclibInterface.get_doc``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_doc_content(
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
        """Async version of ``DoclibInterface.get_doc_content``."""
        raise NotImplementedError()

    @abstractmethod
    async def export_doc_content(self, sha256: str, request: DocContentExportRequest) -> DocContentExportResponse:
        """Async version of ``DoclibInterface.export_doc_content``."""
        raise NotImplementedError()

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        file_type: str | None = None,
        tier: Tier | None = None,
        min_tier: Tier | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResponse:
        """Async version of ``DoclibInterface.search``."""
        raise NotImplementedError()

    @abstractmethod
    async def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> FindResponse:
        """Async version of ``DoclibInterface.find``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_file_by_path(self, path: str) -> FileInfoResponse:
        """Async version of ``DoclibInterface.get_file_by_path``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_config(self) -> ConfigResponse:
        """Async version of ``DoclibInterface.get_config``."""
        raise NotImplementedError()

    @abstractmethod
    async def get_config_key(self, key: str) -> ConfigValueResponse:
        """Async version of ``DoclibInterface.get_config_key``."""
        raise NotImplementedError()

    @abstractmethod
    async def set_config(self, key: str, request: ConfigSetRequest) -> ConfigSetResponse:
        """Async version of ``DoclibInterface.set_config``."""
        raise NotImplementedError()

    @abstractmethod
    async def unset_config(self, key: str) -> ConfigUnsetResponse:
        """Async version of ``DoclibInterface.unset_config``."""
        raise NotImplementedError()

    @abstractmethod
    async def add_watch(self, request: WatchRequest) -> WatchInfo:
        """Async version of ``DoclibInterface.add_watch``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_watches(self) -> WatchListResponse:
        """Async version of ``DoclibInterface.list_watches``."""
        raise NotImplementedError()

    @abstractmethod
    async def remove_watch(self, watch_id: int) -> RemoveWatchResponse:
        """Async version of ``DoclibInterface.remove_watch``."""
        raise NotImplementedError()

    @abstractmethod
    async def add_exclude_rule(self, request: ExcludeRuleRequest) -> ExcludeRuleInfo:
        """Async version of ``DoclibInterface.add_exclude_rule``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_exclude_rules(self) -> ExcludeRuleListResponse:
        """Async version of ``DoclibInterface.list_exclude_rules``."""
        raise NotImplementedError()

    @abstractmethod
    async def remove_exclude_rule(self, rule_id: int) -> RemoveExcludeRuleResponse:
        """Async version of ``DoclibInterface.remove_exclude_rule``."""
        raise NotImplementedError()

    @abstractmethod
    async def add_parsing_rule(self, request: ParsingRuleRequest) -> ParsingRuleInfo:
        """Async version of ``DoclibInterface.add_parsing_rule``."""
        raise NotImplementedError()

    @abstractmethod
    async def list_parsing_rules(self) -> ParsingRuleListResponse:
        """Async version of ``DoclibInterface.list_parsing_rules``."""
        raise NotImplementedError()

    @abstractmethod
    async def remove_parsing_rule(self, rule_id: int) -> RemoveParsingRuleResponse:
        """Async version of ``DoclibInterface.remove_parsing_rule``."""
        raise NotImplementedError()

    @abstractmethod
    async def cleanup_deleted_files(self, request: CleanupDeletedRequest) -> CleanupDeletedResponse:
        """Async version of ``DoclibInterface.cleanup_deleted_files``."""
        raise NotImplementedError()

    @abstractmethod
    async def cleanup_orphan_docs(self, request: CleanupOrphansRequest) -> CleanupOrphansResponse:
        """Async version of ``DoclibInterface.cleanup_orphan_docs``."""
        raise NotImplementedError()

    @abstractmethod
    async def cleanup_temp_files(self, request: CleanupTempRequest) -> CleanupTempResponse:
        """Async version of ``DoclibInterface.cleanup_temp_files``."""
        raise NotImplementedError()
