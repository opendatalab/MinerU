# Copyright (c) Opendatalab. All rights reserved.
"""根据文件内容和容器结构识别 MinerU 支持的输入后缀。"""

from io import BytesIO
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from loguru import logger
from magika import Magika

from ..filetypes import IMAGE_EXTENSIONS, rtf_header_offset

PDF_SIG_BYTES = b"%PDF"
OLE2_SIG_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
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
ODF_MIMETYPE_SUFFIXES = {
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
}
ODF_MANIFEST_PATH = "META-INF/manifest.xml"
ODF_MANIFEST_NS = "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
# OLE2 compound file 内部 stream 名 → 旧 Office 格式后缀
# doc: WordDocument stream；xls: Workbook 或 Book stream；ppt: PowerPoint Document stream
OLE2_STREAM_SUFFIX_MAP: dict[str, str] = {
    "WordDocument": "doc",
    "Workbook": "xls",
    "Book": "xls",
    "PowerPoint Document": "ppt",
}
_STRONG_CONTENT_SUFFIXES = frozenset(
    {
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "rtf",
        "epub",
        "ofd",
        "odt",
        "ods",
        "odp",
        *IMAGE_EXTENSIONS,
    }
)


@lru_cache(maxsize=1)
def _magika() -> Magika:
    """惰性创建文件类型识别器，避免导入 parser 时加载模型。"""
    return Magika()


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


def _guess_odf_suffix_from_zip(package: ZipFile) -> str | None:
    """按 ODF mimetype、manifest 根条目依次识别 odt/ods/odp。"""
    try:
        mimetype_info = package.getinfo("mimetype")
        if mimetype_info.file_size <= 256:
            mimetype = package.read(mimetype_info).decode("ascii", errors="strict").strip()
            if suffix := ODF_MIMETYPE_SUFFIXES.get(mimetype):
                return suffix
    except (KeyError, UnicodeDecodeError, RuntimeError, OSError, ValueError):
        pass
    try:
        manifest_info = package.getinfo(ODF_MANIFEST_PATH)
        if manifest_info.file_size > 1024 * 1024:
            return None
        root = ElementTree.fromstring(package.read(manifest_info))
    except (KeyError, ElementTree.ParseError, RuntimeError, OSError, ValueError):
        return None
    for entry in root.iter(f"{{{ODF_MANIFEST_NS}}}file-entry"):
        if entry.get(f"{{{ODF_MANIFEST_NS}}}full-path") != "/":
            continue
        media_type = entry.get(f"{{{ODF_MANIFEST_NS}}}media-type", "").strip()
        return ODF_MIMETYPE_SUFFIXES.get(media_type)
    return None


def _guess_odf_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """从内存 ZIP 包识别 ODF，失败时不影响后续 OLE/Magika/CSV 路由。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as package:
            return _guess_odf_suffix_from_zip(package)
    except (BadZipFile, RuntimeError, OSError, ValueError):
        return None


def _guess_odf_suffix_by_path(file_path: Path) -> str | None:
    """从路径 ZIP 包识别 ODF，保持现有 OOXML 检测优先级。"""
    try:
        with ZipFile(file_path) as package:
            return _guess_odf_suffix_from_zip(package)
    except (BadZipFile, RuntimeError, OSError, ValueError):
        return None


def _guess_epub_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """从内存 ZIP 包验证 EPUB 强内容身份。"""
    from ..model.flash.epub import detect_epub

    return "epub" if detect_epub(file_bytes) else None


def _guess_epub_suffix_by_path(file_path: Path) -> str | None:
    """从路径 ZIP 包验证 EPUB 强内容身份。"""
    from ..model.flash.epub import detect_epub_path

    return "epub" if detect_epub_path(file_path) else None


def _guess_ofd_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """从内存 ZIP 包验证 OFD 强内容身份。"""
    from ..model.flash.ofd import detect_ofd

    return "ofd" if detect_ofd(file_bytes) else None


def _guess_ofd_suffix_by_path(file_path: Path) -> str | None:
    """从路径 ZIP 包验证 OFD 强内容身份。"""
    from ..model.flash.ofd import detect_ofd_path

    return "ofd" if detect_ofd_path(file_path) else None


def _guess_ole2_suffix_by_bytes(file_bytes: bytes) -> str | None:
    """用 OLE2 magic + olefile 内部 stream 区分 doc/xls/ppt。

    olefile 是纯 Python 库且已是核心依赖（mineru.model.flash.office.legacy 使用）。
    在 OOXML 识别失败后、Magika 兜底前插入此层，避免 Magika 对 OLE2 返回 unknown。
    """
    if len(file_bytes) < 8 or file_bytes[:8] != OLE2_SIG_BYTES:
        return None
    try:
        import olefile  # type: ignore[import-untyped]

        with olefile.OleFileIO(BytesIO(file_bytes)) as ole:
            for stream_name in ole.listdir(streams=True):
                name = "/".join(stream_name)
                suffix = OLE2_STREAM_SUFFIX_MAP.get(name)
                if suffix:
                    return suffix
    except Exception:
        return None
    return None


def _guess_ole2_suffix_by_path(file_path: Path) -> str | None:
    """从文件路径读取 OLE2 容器并识别旧 Office 格式。"""
    try:
        with open(file_path, "rb") as f:
            return _guess_ole2_suffix_by_bytes(f.read())
    except OSError:
        return None


def _has_pdf_signature_by_path(file_path: Path) -> bool:
    """读取文件头判断路径指向的内容是否具有 PDF 强签名。"""
    try:
        with open(file_path, "rb") as file:
            return file.read(len(PDF_SIG_BYTES)) == PDF_SIG_BYTES
    except OSError:
        return False


def _has_rtf_signature_by_path(file_path: Path) -> bool:
    """读取有限文件头并按共享规则识别 RTF 根组。"""
    try:
        with open(file_path, "rb") as file:
            return rtf_header_offset(file.read(128)) is not None
    except OSError:
        return False


def _resolve_signatureless_csv_suffix(detected_suffix: str, file_path: str | Path | None) -> str:
    """仅以 .csv 扩展名兜底无签名文本，并保留强内容类型的优先级。"""
    extension = Path(file_path).suffix.lower().lstrip(".") if file_path else ""
    if extension == "csv":
        if detected_suffix in _STRONG_CONTENT_SUFFIXES:
            return detected_suffix
        return "csv"
    if detected_suffix == "csv":
        if extension in ODF_MIMETYPE_SUFFIXES.values():
            return "txt"
        return extension or "txt"
    return detected_suffix


def _resolve_signatureless_html_suffix(detected_suffix: str, file_path: str | Path | None) -> str:
    """用 .html/.htm 兜底短文本，并把 Magika 的 HTML 结果统一规范为 html。"""
    extension = Path(file_path).suffix.lower().lstrip(".") if file_path else ""
    if extension in {"html", "htm"} and detected_suffix not in _STRONG_CONTENT_SUFFIXES:
        return "html"
    return "html" if detected_suffix == "html" else detected_suffix


def _reject_unverified_package_suffix(detected_suffix: str) -> str:
    """拒绝未通过包身份验证、仅由启发式工具猜出的 ODF/EPUB/OFD 类型。"""
    package_suffixes = {*ODF_MIMETYPE_SUFFIXES.values(), "epub", "ofd"}
    return "unknown" if detected_suffix in package_suffixes else detected_suffix


def guess_suffix_by_bytes(file_bytes: bytes, file_path: str | None = None) -> str:
    if file_bytes[: len(PDF_SIG_BYTES)] == PDF_SIG_BYTES:
        return "pdf"
    if rtf_header_offset(file_bytes[:128]) is not None:
        return "rtf"

    ofd_suffix = _guess_ofd_suffix_by_bytes(file_bytes)
    if ofd_suffix:
        return ofd_suffix

    epub_suffix = _guess_epub_suffix_by_bytes(file_bytes)
    if epub_suffix:
        return epub_suffix

    ooxml_suffix = _guess_ooxml_suffix_by_bytes(file_bytes)
    if ooxml_suffix:
        return ooxml_suffix

    odf_suffix = _guess_odf_suffix_by_bytes(file_bytes)
    if odf_suffix:
        return odf_suffix

    ole2_suffix = _guess_ole2_suffix_by_bytes(file_bytes)
    if ole2_suffix:
        return ole2_suffix

    suffix = _magika().identify_bytes(file_bytes).prediction.output.label
    if (
        file_path
        and suffix in ["ai", "html"]
        and Path(file_path).suffix.lower() in [".pdf"]
        and file_bytes[:4] == PDF_SIG_BYTES
    ):
        suffix = "pdf"
    suffix = _resolve_signatureless_csv_suffix(_reject_unverified_package_suffix(suffix), file_path)
    return _resolve_signatureless_html_suffix(suffix, file_path)


def guess_suffix_by_path(file_path: str | Path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if _has_rtf_signature_by_path(file_path):
        return "rtf"

    ofd_suffix = _guess_ofd_suffix_by_path(file_path)
    if ofd_suffix:
        return ofd_suffix

    epub_suffix = _guess_epub_suffix_by_path(file_path)
    if epub_suffix:
        return epub_suffix

    ooxml_suffix = _guess_ooxml_suffix_by_path(file_path)
    if ooxml_suffix:
        return ooxml_suffix

    odf_suffix = _guess_odf_suffix_by_path(file_path)
    if odf_suffix:
        return odf_suffix

    ole2_suffix = _guess_ole2_suffix_by_path(file_path)
    if ole2_suffix:
        return ole2_suffix

    if _has_pdf_signature_by_path(file_path):
        return "pdf"

    suffix = _magika().identify_path(file_path).prediction.output.label
    if suffix in ["ai", "html"] and file_path.suffix.lower() in [".pdf"]:
        try:
            with open(file_path, "rb") as f:
                if f.read(4) == PDF_SIG_BYTES:
                    suffix = "pdf"
        except Exception as e:
            logger.warning(f"Failed to read file {file_path} for PDF signature check: {e}")
    suffix = _resolve_signatureless_csv_suffix(_reject_unverified_package_suffix(suffix), file_path)
    return _resolve_signatureless_html_suffix(suffix, file_path)


__all__ = ["guess_suffix_by_bytes", "guess_suffix_by_path"]
