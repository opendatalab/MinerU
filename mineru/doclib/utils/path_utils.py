"""Path normalization helpers for doclib."""

from __future__ import annotations

import os
from pathlib import Path


def normalize_doclib_path(path: str) -> str:
    """Normalize a user-facing doclib path without resolving symlinks."""
    stripped = path.strip()
    if not stripped:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(stripped)))


def rebase_watch_event_path(event_path: str, watch_root: str) -> str:
    """Map a watcher event back into the stored watch-root namespace."""
    normalized_event = normalize_doclib_path(event_path)
    normalized_root = normalize_doclib_path(watch_root)
    if not normalized_event or not normalized_root:
        return normalized_event

    event_real = Path(normalized_event).resolve(strict=False)
    root_real = Path(normalized_root).resolve(strict=False)
    try:
        relative_event = event_real.relative_to(root_real)
    except ValueError:
        return normalized_event
    return normalize_doclib_path(str(Path(normalized_root, relative_event)))
