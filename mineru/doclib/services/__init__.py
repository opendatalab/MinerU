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
from .parse_policy import default_parse_range
from .parse_svc import (
    ParseFailure,
    ParseService,
    expand_page_range,
    page_range_covered,
    page_range_uncovered,
    parse_page_range_set,
)
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
    "ParseService",
    "ScanService",
    "SearchService",
    "default_parse_range",
    "ensure_doc_record",
    "expand_page_range",
    "page_range_covered",
    "page_range_uncovered",
    "parse_page_range_set",
]
