from __future__ import annotations

from pathlib import Path

from .types import QUALITY_TIERS, Tier, validate_tier

# ── file types ─────────────────────────────────────────────────────

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

PDF_EXTENSIONS: frozenset[str] = frozenset({"pdf"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "jp2"})

OFFICE_EXTENSIONS: frozenset[str] = frozenset({"docx", "pptx", "xlsx"})

HTML_EXTENSIONS: frozenset[str] = frozenset({"html", "htm"})

TEXT_EXTENSIONS: frozenset[str] = frozenset({"txt", "md", "markdown", "csv", "rst", "tex"})

TIERED_PARSE_EXTENSIONS: frozenset[str] = PDF_EXTENSIONS | IMAGE_EXTENSIONS

FLASH_ONLY_PARSE_EXTENSIONS: frozenset[str] = OFFICE_EXTENSIONS | HTML_EXTENSIONS

PARSEABLE_EXTENSIONS: frozenset[str] = TIERED_PARSE_EXTENSIONS | FLASH_ONLY_PARSE_EXTENSIONS

# Acceptable for doclib ingest
INGESTIBLE_EXTENSIONS: frozenset[str] = PARSEABLE_EXTENSIONS | TEXT_EXTENSIONS

# Used for doclib scan and watch
DISCOVERABLE_EXTENSIONS: frozenset[str] = INGESTIBLE_EXTENSIONS - IMAGE_EXTENSIONS

FILE_TYPE_BY_EXTENSION: dict[str, str] = {
    **dict.fromkeys(PDF_EXTENSIONS, "pdf"),
    **dict.fromkeys(IMAGE_EXTENSIONS, "image"),
    **{ext: ext for ext in OFFICE_EXTENSIONS},
    **dict.fromkeys(HTML_EXTENSIONS, "html"),
    "txt": "text",
    "md": "markdown",
    "markdown": "markdown",
    "csv": "csv",
    "rst": "rst",
    "tex": "tex",
}

MIME_TYPE_BY_EXTENSION: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "htm": "text/html",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "jp2": "image/jp2",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}

LEGACY_OFFICE_EXTENSION_UPGRADES: dict[str, str] = {
    "doc": "docx",
    "ppt": "pptx",
    "xls": "xlsx",
}


def normalize_parse_extension(path_or_ext: str | Path) -> str:
    text = str(path_or_ext)
    if text.startswith("."):
        return text.lower().lstrip(".")
    suffix = Path(text).suffix
    if suffix:
        return suffix.lower().lstrip(".")
    return text.lower().lstrip(".")


def file_type_for_extension(path_or_ext: str | Path) -> str:
    ext = normalize_parse_extension(path_or_ext)
    return FILE_TYPE_BY_EXTENSION.get(ext, ext or "unknown")


def mime_type_for_extension(path_or_ext: str | Path, *, default: str = "application/octet-stream") -> str:
    return MIME_TYPE_BY_EXTENSION.get(normalize_parse_extension(path_or_ext), default)


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


def is_office_temp_lock_file(path: str | Path) -> bool:
    file_path = Path(path)
    return file_path.name.startswith("~$") and file_path.suffix.lower().lstrip(".") in OFFICE_EXTENSIONS
