# Copyright (c) Opendatalab. All rights reserved.
import posixpath
import re
from io import BytesIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from loguru import logger
from lxml import etree

LEGACY_PPT_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

WORDPROCESSINGML_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MARKUP_COMPATIBILITY_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
PRESENTATIONML_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
PACKAGE_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)

CONTENT_PART_TAG = f"{{{PRESENTATIONML_NS}}}contentPart"
RELATIONSHIP_TAG = f"{{{PACKAGE_RELATIONSHIPS_NS}}}Relationship"
PPTX_SHAPE_TAGS = {
    f"{{{PRESENTATIONML_NS}}}sp",
    f"{{{PRESENTATIONML_NS}}}grpSp",
    f"{{{PRESENTATIONML_NS}}}graphicFrame",
    f"{{{PRESENTATIONML_NS}}}cxnSp",
    f"{{{PRESENTATIONML_NS}}}pic",
}

ROOT_TAG_PATTERN = re.compile(
    br"<(?![?!])(?:[A-Za-z_][\w.-]*:)?[A-Za-z_][\w.-]*(?=\s|/?>)"
)

STRICT_OOXML_REPLACEMENTS = (
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/metadata/thumbnail",
        b"http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/customProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships/extendedProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/relationships",
        b"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    ),
    (
        b"http://purl.oclc.org/ooxml/drawingml/main",
        b"http://schemas.openxmlformats.org/drawingml/2006/main",
    ),
    (
        b"http://purl.oclc.org/ooxml/drawingml/chart",
        b"http://schemas.openxmlformats.org/drawingml/2006/chart",
    ),
    (
        b"http://purl.oclc.org/ooxml/presentationml/main",
        b"http://schemas.openxmlformats.org/presentationml/2006/main",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/math",
        b"http://schemas.openxmlformats.org/officeDocument/2006/math",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/customProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/extendedProperties",
        b"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes",
        b"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    ),
    (
        b"http://purl.oclc.org/ooxml/officeDocument/oleObject",
        b"http://schemas.openxmlformats.org/officeDocument/2006/oleObject",
    ),
)

KNOWN_NAMESPACE_DECLARATIONS = {
    b"w": f'xmlns:w="{WORDPROCESSINGML_NS}"'.encode("utf-8"),
}


def normalize_pptx_package(file_bytes: bytes) -> bytes:
    """在进入 python-pptx 前修复常见包级兼容问题，避免修复逻辑散落到形状解析阶段。"""
    if file_bytes.startswith(LEGACY_PPT_MAGIC):
        raise ValueError(
            "Legacy binary PPT files are not supported; convert the file to PPTX before parsing."
        )

    try:
        with ZipFile(BytesIO(file_bytes)) as source:
            loaded_members: list[tuple[ZipInfo, bytes]] = []
            rewritten_members: list[tuple[ZipInfo, bytes]] = []
            skipped_members: set[str] = set()
            changed = False

            for info in source.infolist():
                member_data = _read_member_best_effort(source, info)
                if member_data is None:
                    skipped_members.add(info.filename)
                    changed = True
                    continue
                loaded_members.append((info, member_data))

            for info, member_data in loaded_members:
                normalized_data = _normalize_member_xml(
                    info.filename,
                    member_data,
                    skipped_members,
                )
                if normalized_data != member_data:
                    changed = True
                rewritten_members.append((info, normalized_data))
    except BadZipFile as exc:
        raise ValueError("Invalid PPTX package: file is not a ZIP archive.") from exc

    if not changed:
        return file_bytes

    return _write_package(rewritten_members)


def _read_member_best_effort(source: ZipFile, info: ZipInfo) -> bytes | None:
    """读取 ZIP 成员；损坏的媒体资源可跳过，关键 XML/关系文件仍保持失败。"""
    try:
        return source.read(info.filename)
    except BadZipFile as exc:
        if _is_skippable_corrupt_member(info.filename):
            logger.warning(
                f"Skipping corrupt non-critical PPTX media member {info.filename}: {exc}"
            )
            return None
        raise


def _is_skippable_corrupt_member(filename: str) -> bool:
    """判断损坏成员是否属于可降级丢弃的媒体资源。"""
    return filename.startswith("ppt/media/")


def _normalize_member_xml(
    filename: str,
    member_data: bytes,
    skipped_members: set[str] | None = None,
) -> bytes:
    """仅对 XML/关系成员做文本级和结构级规范化，二进制资源保持原样。"""
    if not (filename.endswith(".xml") or filename.endswith(".rels")):
        return member_data

    normalized = _translate_strict_ooxml_uris(member_data)
    if filename.endswith(".rels"):
        normalized = _remove_relationships_to_skipped_members(
            filename,
            normalized,
            skipped_members or set(),
        )
    if filename.endswith(".xml"):
        normalized = _add_missing_known_namespaces(normalized)
        normalized = _replace_content_part_alternate_content_with_fallback(normalized)
    return normalized


def _remove_relationships_to_skipped_members(
    rels_filename: str,
    rels_xml: bytes,
    skipped_members: set[str],
) -> bytes:
    """删除指向已跳过媒体成员的内部关系，避免归一化包保留悬空引用。"""
    if not skipped_members:
        return rels_xml

    try:
        parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
        root = etree.fromstring(rels_xml, parser)
    except etree.XMLSyntaxError:
        return rels_xml

    removed_count = 0
    for relationship in list(root):
        if (
            relationship.tag != RELATIONSHIP_TAG
            and etree.QName(relationship).localname != "Relationship"
        ):
            continue
        if relationship.get("TargetMode") == "External":
            continue

        target = relationship.get("Target")
        if not target:
            continue

        resolved_target = _resolve_relationship_target(rels_filename, target)
        if resolved_target in skipped_members:
            root.remove(relationship)
            removed_count += 1

    if removed_count == 0:
        return rels_xml

    return etree.tostring(
        root,
        xml_declaration=rels_xml.lstrip().startswith(b"<?xml"),
        encoding="UTF-8",
        standalone=True,
    )


def _resolve_relationship_target(rels_filename: str, target: str) -> str:
    """把关系文件中的 Target 解析成 ZIP 包内的规范成员路径。"""
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")

    base_dir = _relationship_source_base_dir(rels_filename)
    if not base_dir:
        return posixpath.normpath(target)
    return posixpath.normpath(posixpath.join(base_dir, target))


def _relationship_source_base_dir(rels_filename: str) -> str:
    """根据 .rels 成员路径推导源 part 的基础目录。"""
    if rels_filename == "_rels/.rels":
        return ""

    marker = "/_rels/"
    if marker not in rels_filename:
        return posixpath.dirname(rels_filename)

    prefix, rels_basename = rels_filename.rsplit(marker, 1)
    if not rels_basename.endswith(".rels"):
        return prefix

    source_part_name = rels_basename[: -len(".rels")]
    source_part_path = posixpath.normpath(posixpath.join(prefix, source_part_name))
    return posixpath.dirname(source_part_path)


def _translate_strict_ooxml_uris(xml_bytes: bytes) -> bytes:
    """把 Strict OOXML URI 转为 python-pptx 能识别的 Transitional URI。"""
    normalized = xml_bytes
    for strict_uri, transitional_uri in STRICT_OOXML_REPLACEMENTS:
        normalized = normalized.replace(strict_uri, transitional_uri)
    return normalized


def _add_missing_known_namespaces(xml_bytes: bytes) -> bytes:
    """为实际使用但未声明的已知前缀补齐命名空间，修复轻微损坏的 XML。"""
    declarations = []
    for prefix, declaration in KNOWN_NAMESPACE_DECLARATIONS.items():
        if prefix + b":" in xml_bytes and b"xmlns:" + prefix + b"=" not in xml_bytes:
            declarations.append(declaration)

    if not declarations:
        return xml_bytes

    root_match = ROOT_TAG_PATTERN.search(xml_bytes)
    if root_match is None:
        return xml_bytes

    close_index = xml_bytes.find(b">", root_match.end())
    if close_index < 0:
        return xml_bytes

    insert_at = close_index
    if insert_at > 0 and xml_bytes[insert_at - 1 : insert_at] == b"/":
        insert_at -= 1

    namespace_attrs = b" " + b" ".join(declarations)
    return xml_bytes[:insert_at] + namespace_attrs + xml_bytes[insert_at:]


def _replace_content_part_alternate_content_with_fallback(xml_bytes: bytes) -> bytes:
    """将 python-pptx 不支持的 p:contentPart 优先分支替换为可解析的 fallback 图形。"""
    if b"AlternateContent" not in xml_bytes or b"contentPart" not in xml_bytes:
        return xml_bytes

    try:
        parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
        root = etree.fromstring(xml_bytes, parser)
    except etree.XMLSyntaxError:
        return xml_bytes

    replaced_count = 0
    alternate_content_nodes = root.findall(
        f".//{{{MARKUP_COMPATIBILITY_NS}}}AlternateContent"
    )
    for alternate_content in alternate_content_nodes:
        if _replace_single_alternate_content(alternate_content):
            replaced_count += 1

    if replaced_count == 0:
        return xml_bytes

    return etree.tostring(
        root,
        xml_declaration=xml_bytes.lstrip().startswith(b"<?xml"),
        encoding="UTF-8",
        standalone=True,
    )


def _replace_single_alternate_content(alternate_content: etree._Element) -> bool:
    """替换单个 AlternateContent 节点，优先保留 fallback 中的可见图形。"""
    choice = alternate_content.find(f"{{{MARKUP_COMPATIBILITY_NS}}}Choice")
    fallback = alternate_content.find(f"{{{MARKUP_COMPATIBILITY_NS}}}Fallback")
    if choice is None or fallback is None:
        return False

    if not any(child.tag == CONTENT_PART_TAG for child in choice):
        return False

    fallback_shapes = [child for child in list(fallback) if child.tag in PPTX_SHAPE_TAGS]
    if not fallback_shapes:
        return False

    parent = alternate_content.getparent()
    if parent is None:
        return False

    insert_index = parent.index(alternate_content)
    parent.remove(alternate_content)
    for fallback_shape in fallback_shapes:
        fallback.remove(fallback_shape)
        parent.insert(insert_index, fallback_shape)
        insert_index += 1
    return True


def _write_package(members: list[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 PPTX ZIP 包，并重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()
