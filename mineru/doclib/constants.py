"""Shared file type whitelists and constants."""

from __future__ import annotations

# ── file types ─────────────────────────────────────────────────────

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
    "docx",
    "pptx",
    "xlsx",
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
