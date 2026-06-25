"""Shared file type whitelists and constants."""

from __future__ import annotations

from pathlib import Path

# ── file types ─────────────────────────────────────────────────────

OFFICE_EXTENSIONS: set[str] = {
    "docx",
    "pptx",
    "xlsx",
}

TEXT_EXTENSIONS: set[str] = {
    "csv",
    "md",
    "markdown",
    "rst",
    "tex",
    "txt",
}

ALLOWED_EXTENSIONS: set[str] = {
    "pdf",
    *OFFICE_EXTENSIONS,
    "html",
    "htm",
    *TEXT_EXTENSIONS,
    # Unsupported legacy Office formats:
    # "doc",
    # "xls",
    # "ppt",
    # Unsupported document/e-book/archive-like formats:
    # "epub",
    # "key",
    # "mobi",
    # "numbers",
    # "ods",
    # "odt",
    # "pages",
    # "rtf",
    # Unsupported mail formats:
    # "eml",
    # "mbox",
}


def is_office_temp_lock_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.name.startswith("~$") and file_path.suffix.lower().lstrip(".") in OFFICE_EXTENSIONS

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "*/Library/*",
    "*/.git/*",
    "*/node_modules/*",
    "*/vendor/*",
    "*/go/pkg/*",
    "*/__pycache__/*",
    "*/.venv/*",
    "*/miniconda3/*",
    "*/.nvm/*",
    "*/.docker/*",
    "*/target/*",
    "*/dist/*",
    "*/build/*",
]
