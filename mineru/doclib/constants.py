"""Doclib configuration defaults and constants."""

from __future__ import annotations

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
