"""Doclib service layer: parse, scan, config, search, and cleanup."""

from .cleanup_svc import CleanupService
from .config_svc import ConfigService
from .content_read_svc import ContentReadService
from .ingest_svc import (
    FileRefreshResult,
    FileRefreshStatus,
    IngestService,
    ensure_doc_record,
)
from ...parser.page_range import expand_page_range, parse_page_range_set
from .parse_errors import ParseFailure
from .parse_policy import default_parse_range, page_range_covered, page_range_uncovered
from .parse_queue_svc import ParseQueueService
from .parse_execution_svc import ParseExecutionService
from .scan_svc import ScanService
from .search_svc import SearchService

__all__ = [
    "CleanupService",
    "ConfigService",
    "ContentReadService",
    "FileRefreshResult",
    "FileRefreshStatus",
    "IngestService",
    "ParseFailure",
    "ParseQueueService",
    "ParseExecutionService",
    "ScanService",
    "SearchService",
    "default_parse_range",
    "ensure_doc_record",
    "expand_page_range",
    "page_range_covered",
    "page_range_uncovered",
    "parse_page_range_set",
]
