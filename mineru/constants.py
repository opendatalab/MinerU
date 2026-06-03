"""Shared enums, file type whitelists, and constants."""

from __future__ import annotations

from enum import StrEnum

# ── paths ──────────────────────────────────────────────────────────

SOCKET_PATH = "/tmp/mineru.sock"
DATA_DIR = "~/MinerU"

# ── file types ─────────────────────────────────────────────────────

ALLOWED_EXTENSIONS: set[str] = {
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "rtf", "odt", "ods",
    "epub", "mobi",
    "pages", "key", "numbers",
    "txt", "md", "markdown", "rst", "tex",
    "html", "htm",
    "eml", "mbox",
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

# ── enums ──────────────────────────────────────────────────────────


class Tier(StrEnum):
    FLASH = "flash"
    STANDARD = "standard"
    PRO = "pro"


TIER_ORDER: dict[str, int] = {
    Tier.FLASH: 0,
    Tier.STANDARD: 1,
    Tier.PRO: 2,
}


class ScanStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"
    UNREACHABLE = "unreachable"


class ParseStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    DONE = "done"
    FAILED = "failed"


class RuleType(StrEnum):
    EXCLUDE = "exclude"
    PARSING_RULE = "parsing_rule"


class WatchStatus(StrEnum):
    ACTIVE = "active"
    UNREACHABLE = "unreachable"
