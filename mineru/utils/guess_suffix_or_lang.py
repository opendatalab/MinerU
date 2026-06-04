# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from loguru import logger
from magika import Magika

DEFAULT_LANG = "txt"
PDF_SIG_BYTES = b"%PDF"
OOXML_ROOT_RELS = "_rels/.rels"
OOXML_CONTENT_TYPES = "[Content_Types].xml"
OOXML_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OOXML_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
OOXML_OFFICE_DOCUMENT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
OOXML_MAIN_CONTENT_TYPES = {
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"): "docx",
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"): "pptx",
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"): "xlsx",
}
magika = Magika()


def _normalize_text_for_language_guess(code: str) -> str:
    if not code:
        return ""

    normalized = []
    index = 0
    while index < len(code):
        current_char = code[index]
        current_ord = ord(current_char)

        if 0xD800 <= current_ord <= 0xDBFF:
            if index + 1 < len(code):
                next_char = code[index + 1]
                next_ord = ord(next_char)
                if 0xDC00 <= next_ord <= 0xDFFF:
                    pair = current_char + next_char
                    normalized.append(pair.encode("utf-16", "surrogatepass").decode("utf-16"))
                    index += 2
                    continue
            index += 1
            continue

        if 0xDC00 <= current_ord <= 0xDFFF:
            index += 1
            continue

        normalized.append(current_char)
        index += 1

    return "".join(normalized)


def guess_language_by_text(code: str) -> str:
    normalized_code = _normalize_text_for_language_guess(code)
    if not normalized_code:
        return DEFAULT_LANG

    try:
        codebytes = normalized_code.encode("utf-8", errors="replace")
        lang = magika.identify_bytes(codebytes).prediction.output.label
    except Exception:
        return DEFAULT_LANG

    return lang if lang != "unknown" else DEFAULT_LANG


def _strip_package_part_name(part_name: str | None) -> str:
    """规范化 OPC part 路径，方便匹配 Content_Types 中的 PartName。"""
    if not part_name:
        return ""
    return part_name.replace("\\", "/").lstrip("/")


def _ooxml_relationship_targets(root: ElementTree.Element) -> list[str]:
    """从根关系文件中提取 Office 主文档关系目标。"""
    targets = []
    for relationship in root:
        if relationship.tag not in {
            f"{{{OOXML_PACKAGE_REL_NS}}}Relationship",
            "Relationship",
        }:
            continue
        if relationship.get("TargetMode") == "External":
            continue
        if relationship.get("Type") != OOXML_OFFICE_DOCUMENT_REL:
            continue
        target = _strip_package_part_name(relationship.get("Target"))
        if target:
            targets.append(target)
    return targets


def _ooxml_content_type_overrides(root: ElementTree.Element) -> dict[str, str]:
    """读取 Content_Types 中每个显式 part 的 ContentType 映射。"""
    overrides = {}
    for override in root:
        if override.tag not in {
            f"{{{OOXML_CONTENT_TYPES_NS}}}Override",
            "Override",
        }:
            continue
        part_name = _strip_package_part_name(override.get("PartName"))
        content_type = override.get("ContentType")
        if part_name and content_type:
            overrides[part_name] = content_type
    return overrides


def _guess_ooxml_suffix_from_zip(package: ZipFile) -> str | None:
    """根据 OOXML 包内标准主文档关系和主内容类型判断 Office 子类型。"""
    rels_root = ElementTree.fromstring(package.read(OOXML_ROOT_RELS))
    content_types_root = ElementTree.fromstring(package.read(OOXML_CONTENT_TYPES))

    overrides = _ooxml_content_type_overrides(content_types_root)
    for target in _ooxml_relationship_targets(rels_root):
        suffix = OOXML_MAIN_CONTENT_TYPES.get(overrides.get(target, ""))
        if suffix:
            return suffix
    return None


def _guess_ooxml_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """优先用 OOXML 包结构识别 docx/pptx/xlsx，避免 Magika 被内嵌对象误导。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as package:
            return _guess_ooxml_suffix_from_zip(package)
    except (
        BadZipFile,
        KeyError,
        ElementTree.ParseError,
        RuntimeError,
        OSError,
        ValueError,
    ):
        return None


def _guess_ooxml_suffix_by_path(file_path: Path) -> str | None:
    """从文件路径读取 OOXML 包结构；失败时交给 Magika 原有逻辑兜底。"""
    try:
        with ZipFile(file_path) as package:
            return _guess_ooxml_suffix_from_zip(package)
    except (
        BadZipFile,
        KeyError,
        ElementTree.ParseError,
        RuntimeError,
        OSError,
        ValueError,
    ):
        return None


def guess_suffix_by_bytes(file_bytes: bytes, file_path: str | None = None) -> str:
    ooxml_suffix = _guess_ooxml_suffix_by_bytes(file_bytes)
    if ooxml_suffix:
        return ooxml_suffix

    suffix = magika.identify_bytes(file_bytes).prediction.output.label
    if (
        file_path
        and suffix in ["ai", "html"]
        and Path(file_path).suffix.lower() in [".pdf"]
        and file_bytes[:4] == PDF_SIG_BYTES
    ):
        suffix = "pdf"
    return suffix


def guess_suffix_by_path(file_path: str | Path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    ooxml_suffix = _guess_ooxml_suffix_by_path(file_path)
    if ooxml_suffix:
        return ooxml_suffix

    suffix = magika.identify_path(file_path).prediction.output.label
    if suffix in ["ai", "html"] and file_path.suffix.lower() in [".pdf"]:
        try:
            with open(file_path, "rb") as f:
                if f.read(4) == PDF_SIG_BYTES:
                    suffix = "pdf"
        except Exception as e:
            logger.warning(f"Failed to read file {file_path} for PDF signature check: {e}")
    return suffix
