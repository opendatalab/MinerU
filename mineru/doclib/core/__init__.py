"""Core data layer: database, file I/O, and full-text search."""

from .db import DatabaseManager
from .file_io import FileStat, compute_sha256, extract_metadata, get_file_stat
from .fts import FTSManager, strip_sep, tokenize_for_index, tokenize_for_query

__all__ = [
    "DatabaseManager",
    "FTSManager",
    "FileStat",
    "compute_sha256",
    "extract_metadata",
    "get_file_stat",
    "strip_sep",
    "tokenize_for_index",
    "tokenize_for_query",
]
