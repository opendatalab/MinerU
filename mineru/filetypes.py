from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from .types import QUALITY_TIERS, Tier, validate_tier

# ── file types ─────────────────────────────────────────────────────

# Legacy Office binary formats and RTF are natively parsed by model.flash.office converters.
# Unsupported document/e-book/archive-like formats:
# "key",
# "mobi",
# "numbers",
# "pages",
# Unsupported mail formats:
# "eml",
# "mbox",
# Other doc formats:
# "djvu/djv"
# "epdf",
# "xps",
# "ps",
# "eps/epsi/epi"
# "ept/ept2/ept3"

PDF_EXTENSIONS: frozenset[str] = frozenset({"pdf"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "jp2"})

ODF_EXTENSIONS: frozenset[str] = frozenset({"odt", "ods", "odp"})

EPUB_EXTENSIONS: frozenset[str] = frozenset({"epub"})

OFD_EXTENSIONS: frozenset[str] = frozenset({"ofd"})

OFFICE_EXTENSIONS: frozenset[str] = frozenset({"doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf"}) | ODF_EXTENSIONS

HTML_EXTENSIONS: frozenset[str] = frozenset({"html", "htm", "shtml"})

CSV_EXTENSIONS: frozenset[str] = frozenset({"csv", "tsv"})

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {"txt", "text", "ftxt", "md", "markdown", "mdx", "rst", "tex", "latex", "adoc", "asciidoc"}
)

TIERED_PARSE_EXTENSIONS: frozenset[str] = PDF_EXTENSIONS | IMAGE_EXTENSIONS

PAGE_RANGE_PARSE_EXTENSIONS: frozenset[str] = PDF_EXTENSIONS

FLASH_ONLY_PARSE_EXTENSIONS: frozenset[str] = (
    OFFICE_EXTENSIONS | HTML_EXTENSIONS | CSV_EXTENSIONS | EPUB_EXTENSIONS | OFD_EXTENSIONS
)

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
    **dict.fromkeys(CSV_EXTENSIONS, "csv"),
    **dict.fromkeys(EPUB_EXTENSIONS, "epub"),
    **dict.fromkeys(OFD_EXTENSIONS, "ofd"),
    "txt": "text",
    "text": "text",
    "ftxt": "text",
    "md": "markdown",
    "markdown": "markdown",
    "mdx": "markdown",
    "rst": "rst",
    "tex": "tex",
    "latex": "tex",
    "adoc": "asciidoc",
    "asciidoc": "asciidoc",
}

TEXT_FILE_TYPES: frozenset[str] = frozenset(FILE_TYPE_BY_EXTENSION[ext] for ext in TEXT_EXTENSIONS)

MIME_TYPE_BY_EXTENSION: dict[str, str] = {
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "rtf": "application/rtf",
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "epub": "application/epub+zip",
    # OFD 尚无 IANA 注册 subtype；使用生态中通行的项目级 MIME 映射。
    "ofd": "application/ofd",
    "html": "text/html",
    "htm": "text/html",
    "shtml": "text/html",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "jp2": "image/jp2",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "txt": "text/plain",
    "text": "text/plain",
    "ftxt": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "mdx": "text/markdown",
    "rst": "text/x-rst",
    "tex": "text/x-tex",
    "latex": "application/x-latex",
    "adoc": "text/plain",
    "asciidoc": "text/plain",
}

_RTF_HEADER_RE: Final = re.compile(
    rb"\A(?:\xef\xbb\xbf)?[ \t\r\n]{0,64}\{\\rtf(?=[0-9])",
    re.IGNORECASE,
)


def rtf_header_offset(file_bytes: bytes) -> int | None:
    """返回 RTF 根组左花括号偏移；不接受带任意前缀的伪装文本。"""
    match = _RTF_HEADER_RE.match(file_bytes)
    if match is None:
        return None
    return file_bytes.find(b"{", 0, match.end())


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


def is_page_range_parse_extension(path_or_ext: str | Path) -> bool:
    """仅允许 PDF 输入使用局部页范围解析。"""
    return normalize_parse_extension(path_or_ext) in PAGE_RANGE_PARSE_EXTENSIONS


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
