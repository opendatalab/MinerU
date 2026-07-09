from __future__ import annotations

from pathlib import Path

from .types import QUALITY_TIER_SELECTION_ORDER, QUALITY_TIERS, Tier, validate_tier

# ── file types ─────────────────────────────────────────────────────

OFFICE_EXTENSIONS: set[str] = {
    "docx",
    "pptx",
    "xlsx",
}

LEGACY_OFFICE_EXTENSION_UPGRADES: dict[str, str] = {
    "doc": "docx",
    "ppt": "pptx",
    "xls": "xlsx",
}

IMAGE_EXTENSIONS: set[str] = {
    "bmp",
    "gif",
    "jp2",
    "jpeg",
    "jpg",
    "png",
    "tiff",
    "webp",
}

TEXT_EXTENSIONS: set[str] = {
    "csv",
    "md",
    "markdown",
    "rst",
    "tex",
    "txt",
}

HTML_EXTENSIONS: set[str] = {
    "html",
    "htm",
}

DISCOVERABLE_EXTENSIONS: set[str] = {
    "pdf",
    *OFFICE_EXTENSIONS,
    *HTML_EXTENSIONS,
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

PARSEABLE_EXTENSIONS: set[str] = {
    *DISCOVERABLE_EXTENSIONS,
    *IMAGE_EXTENSIONS,
}

TIERED_PARSE_EXTENSIONS: frozenset[str] = frozenset({"pdf", *IMAGE_EXTENSIONS})
FLASH_ONLY_PARSE_EXTENSIONS: frozenset[str] = frozenset(
    {
        *OFFICE_EXTENSIONS,
        *HTML_EXTENSIONS,
        *TEXT_EXTENSIONS,
    }
)
PARSING_RULE_TIER_SELECTION_ORDER: tuple[Tier, ...] = (*QUALITY_TIER_SELECTION_ORDER, "flash")


def normalize_parse_extension(path_or_ext: str | Path) -> str:
    text = str(path_or_ext)
    if text.startswith("."):
        return text.lower().lstrip(".")
    suffix = Path(text).suffix
    if suffix:
        return suffix.lower().lstrip(".")
    return text.lower().lstrip(".")


def is_tiered_parse_extension(path_or_ext: str | Path) -> bool:
    return normalize_parse_extension(path_or_ext) in TIERED_PARSE_EXTENSIONS


def is_flash_only_parse_extension(path_or_ext: str | Path) -> bool:
    return normalize_parse_extension(path_or_ext) in FLASH_ONLY_PARSE_EXTENSIONS


def ensure_tier_supported_for_parse_extension(tier: Tier | None, path_or_ext: str | Path) -> None:
    if tier not in QUALITY_TIERS or is_tiered_parse_extension(path_or_ext):
        return
    ext = normalize_parse_extension(path_or_ext)
    raise ValueError(f"Tier '{tier}' is only supported for PDF and image files; '{ext}' files use tier 'flash'.")


def batch_effective_parse_tier(tier: Tier, path_or_ext: str | Path) -> Tier:
    if is_flash_only_parse_extension(path_or_ext):
        return "flash"
    return validate_tier(tier)


def select_parsing_rule_tier(available_tiers: list[Tier] | tuple[Tier, ...] | set[Tier] | None = None) -> Tier:
    available = set(available_tiers or PARSING_RULE_TIER_SELECTION_ORDER)
    for candidate in PARSING_RULE_TIER_SELECTION_ORDER:
        if candidate in available:
            return candidate
    return "flash"


def is_office_temp_lock_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.name.startswith("~$") and file_path.suffix.lower().lstrip(".") in OFFICE_EXTENSIONS
