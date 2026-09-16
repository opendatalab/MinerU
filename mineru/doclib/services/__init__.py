"""Doclib service layer: parse, scan, config, search, and cleanup."""

from .cleanup_svc import CleanupService
from .config_svc import ConfigService
from .parse_svc import (
    FileRefreshResult,
    FileRefreshStatus,
    ParseFailure,
    ParseService,
    default_parse_range,
    ensure_doc_record,
    expand_page_range,
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    page_range_covered,
    page_range_uncovered,
    parse_batch_json_path,
    parse_page_range_set,
)
from .scan_svc import ScanService
from .search_svc import SearchService

__all__ = [
    "CleanupService",
    "ConfigService",
    "FileRefreshResult",
    "FileRefreshStatus",
    "ParseFailure",
    "ParseService",
    "ScanService",
    "SearchService",
    "default_parse_range",
    "ensure_doc_record",
    "expand_page_range",
    "filter_pages_by_user_range",
    "load_pages_from_done_batches",
    "page_range_covered",
    "page_range_uncovered",
    "parse_batch_json_path",
    "parse_page_range_set",
]
