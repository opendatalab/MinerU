"""Path normalization helpers for CLI user input."""

from __future__ import annotations

import os


def normalize_cli_path(path: str) -> str:
    """Normalize a local filesystem path provided to the CLI."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))
