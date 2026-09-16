"""Path normalization helpers for CLI user input."""

from __future__ import annotations

from ..doclib.utils.path_utils import normalize_doclib_path


def normalize_cli_path(path: str) -> str:
    """Normalize a local filesystem path provided to the CLI."""
    return normalize_doclib_path(path)
