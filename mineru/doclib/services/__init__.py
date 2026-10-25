"""Doclib service layer: parse, scan, config, search, and cleanup."""

from .cleanup_svc import CleanupService
from .config_svc import ConfigService
from .content_read_svc import ContentReadService
from .parse_svc import (
    FileRefreshResult,
    FileRefreshStatus,
    ParseFailure,
    ParseService,
    default_parse_range,
    ensure_doc_record,
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
