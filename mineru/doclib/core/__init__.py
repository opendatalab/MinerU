"""Core data layer: database, file I/O, and full-text search."""

from .db import DatabaseManager
from .file_io import FileStat, MetadataExtractionError, compute_sha256, extract_metadata, get_file_stat
from .fts import FTSManager, strip_sep, tokenize_for_index, tokenize_for_query
from .middle_json import read_cached_middle_json

from .parse_artifacts import (
    accessible_file_for_sha256,
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    parse_artifact_directory,
    parse_batch_is_readable,
    parse_batch_json_path,
    parse_coverage,
    write_parse_payload,
)

__all__ = [
    "DatabaseManager",
    "accessible_file_for_sha256",
    "filter_pages_by_user_range",
    "load_pages_from_done_batches",
    "parse_artifact_directory",
    "parse_batch_is_readable",
    "parse_batch_json_path",
    "parse_coverage",
    "read_cached_middle_json",
    "write_parse_payload",
    "FTSManager",
    "FileStat",
    "MetadataExtractionError",
    "compute_sha256",
    "extract_metadata",
    "get_file_stat",
    "strip_sep",
    "tokenize_for_index",
    "tokenize_for_query",
]
