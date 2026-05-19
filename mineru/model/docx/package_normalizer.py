# Copyright (c) Opendatalab. All rights reserved.
import posixpath
import zlib
from io import BytesIO
from typing import Iterator
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

from loguru import logger
from lxml import etree


PACKAGE_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
RELATIONSHIP_TAG = f"{{{PACKAGE_RELATIONSHIPS_NS}}}Relationship"
ZIP_MEMBER_READ_ERRORS = (BadZipFile, RuntimeError, NotImplementedError, zlib.error)
DOCX_EMBEDDED_OFFICE_PREFIX = "word/embeddings/"


def normalize_docx_package(file_bytes: bytes) -> bytes:
    """在进入 python-docx 前修复 DOCX 包级容错问题。"""
    with ZipFile(BytesIO(file_bytes)) as source:
        package_members = {info.filename for info in source.infolist()}
        reachable_members, relationship_graph_complete = (
            _collect_relationship_reachable_members(source, package_members)
        )
        trusted_reachable_members = (
            reachable_members if relationship_graph_complete else None
        )
        loaded_members: list[tuple[ZipInfo, bytes]] = []
        skipped_members: set[str] = set()
        changed = False

        for info in source.infolist():
            member_data = _read_member_best_effort(
                source,
                info,
                trusted_reachable_members,
            )
            if member_data is None:
                skipped_members.add(info.filename)
                changed = True
                continue
            loaded_members.append((info, member_data))

        rewritten_members: list[tuple[ZipInfo, bytes]] = []
        for info, member_data in loaded_members:
            normalized_data = member_data
            if info.filename.endswith(".rels"):
                normalized_data = _remove_missing_internal_relationships(
                    info.filename,
                    member_data,
                    package_members,
                    skipped_members,
                )
                if normalized_data != member_data:
                    changed = True
            rewritten_members.append((info, normalized_data))

    if not changed:
        return file_bytes

    return _write_package(rewritten_members)


def _collect_relationship_reachable_members(
    source: ZipFile,
    package_members: set[str],
) -> tuple[set[str], bool]:
    """按 OPC relationship 图收集 python-docx/mammoth 可能访问的包成员。"""
    reachable_members = {"[Content_Types].xml"}
    root_rels = "_rels/.rels"
    if root_rels not in package_members:
        return reachable_members, False

    relationship_queue = [root_rels]
    processed_relationships: set[str] = set()
    graph_complete = True

    while relationship_queue:
        rels_filename = relationship_queue.pop()
        if rels_filename in processed_relationships:
            continue

        processed_relationships.add(rels_filename)
        reachable_members.add(rels_filename)

        rels_xml = source.read(rels_filename)
        try:
            targets = list(
                _iter_internal_relationship_targets(rels_filename, rels_xml)
            )
        except etree.XMLSyntaxError:
            graph_complete = False
            continue

        for target in targets:
            if target not in package_members:
                continue
            reachable_members.add(target)

            target_rels = _relationship_part_rels_filename(target)
            if (
                target_rels is not None
                and target_rels in package_members
                and target_rels not in processed_relationships
            ):
                relationship_queue.append(target_rels)

    return reachable_members, graph_complete


def _iter_internal_relationship_targets(
    rels_filename: str,
    rels_xml: bytes,
) -> Iterator[str]:
    """解析 .rels 文件，逐个返回有效的内部 relationship 目标路径。"""
    parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
    root = etree.fromstring(rels_xml, parser)
    for relationship in root:
        if not _is_relationship_element(relationship):
            continue
        if relationship.get("TargetMode") == "External":
            continue

        resolved_target = _resolve_internal_relationship_target(
            rels_filename,
            relationship.get("Target"),
        )
        if resolved_target is not None:
            yield resolved_target


def _relationship_part_rels_filename(part_name: str) -> str | None:
    """根据包内 part 路径推导它对应的 relationship 成员路径。"""
    normalized_part_name = part_name.replace("\\", "/")
    if normalized_part_name in {"", "."} or normalized_part_name.startswith("../"):
        return None

    part_dir, part_basename = posixpath.split(normalized_part_name)
    if not part_basename:
        return None
    if part_dir:
        return f"{part_dir}/_rels/{part_basename}.rels"
    return f"_rels/{part_basename}.rels"


def _read_member_best_effort(
    source: ZipFile,
    info: ZipInfo,
    reachable_members: set[str] | None,
) -> bytes | None:
    """读取 ZIP 成员；仅跳过不可达坏成员或可降级媒体，关键成员继续失败。"""
    try:
        return source.read(info.filename)
    except ZIP_MEMBER_READ_ERRORS as exc:
        if _is_skippable_corrupt_member(info.filename, reachable_members):
            logger.warning(
                "Skipping corrupt non-critical DOCX member {}: {}",
                info.filename,
                exc,
            )
            return None
        raise


def _is_skippable_corrupt_member(
    filename: str,
    reachable_members: set[str] | None,
) -> bool:
    """判断损坏成员是否可安全丢弃，避免吞掉正文结构损坏。"""
    if filename.startswith("word/media/"):
        return True
    if _is_docx_embedded_office_member(filename):
        return True
    return reachable_members is not None and filename not in reachable_members


def _is_docx_embedded_office_member(filename: str) -> bool:
    """判断成员是否为 Word 内嵌 Office/OLE 对象载荷，解析正文时可降级跳过。"""
    return filename.replace("\\", "/").startswith(DOCX_EMBEDDED_OFFICE_PREFIX)


def _is_relationship_element(element: etree._Element) -> bool:
    """判断 XML 节点是否为 Relationship 元素，兼容缺省命名空间。"""
    if element.tag == RELATIONSHIP_TAG:
        return True
    try:
        return etree.QName(element).localname == "Relationship"
    except ValueError:
        return False


def _remove_missing_internal_relationships(
    rels_filename: str,
    rels_xml: bytes,
    package_members: set[str],
    skipped_members: set[str],
) -> bytes:
    """删除指向缺失、非法或已跳过成员的关系，避免 python-docx 加载时崩溃。"""
    try:
        parser = etree.XMLParser(resolve_entities=False, remove_blank_text=False)
        root = etree.fromstring(rels_xml, parser)
    except etree.XMLSyntaxError:
        return rels_xml

    removed_count = 0
    for relationship in list(root):
        if not _is_relationship_element(relationship):
            continue
        if relationship.get("TargetMode") == "External":
            continue

        resolved_target = _resolve_internal_relationship_target(
            rels_filename,
            relationship.get("Target"),
        )
        if (
            resolved_target is not None
            and resolved_target in package_members
            and resolved_target not in skipped_members
        ):
            continue

        root.remove(relationship)
        removed_count += 1

    if removed_count == 0:
        return rels_xml

    logger.debug(
        "Removed {} broken internal DOCX relationships from {}",
        removed_count,
        rels_filename,
    )
    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone="yes",
    )


def _resolve_internal_relationship_target(
    rels_filename: str,
    target: str | None,
) -> str | None:
    """把内部 relationship Target 解析成 ZIP 包内成员路径。"""
    if not target:
        return None

    target = target.replace("\\", "/")
    if target.startswith("/"):
        resolved = posixpath.normpath(target.lstrip("/"))
    else:
        base_dir = _relationship_source_base_dir(rels_filename)
        if base_dir is None:
            return None
        resolved = posixpath.normpath(posixpath.join(base_dir, target))

    if resolved in {"", "."} or resolved.startswith("../"):
        return None
    return resolved


def _relationship_source_base_dir(rels_filename: str) -> str | None:
    """根据 .rels 路径推导源 part 所在目录。"""
    rels_filename = rels_filename.replace("\\", "/")
    if rels_filename == "_rels/.rels":
        return ""

    marker = "/_rels/"
    if marker not in rels_filename:
        return None

    prefix, rels_basename = rels_filename.rsplit(marker, 1)
    if not rels_basename.endswith(".rels"):
        return None

    source_part_name = rels_basename[: -len(".rels")]
    source_part_path = posixpath.normpath(posixpath.join(prefix, source_part_name))
    return posixpath.dirname(source_part_path)


def _write_package(members: list[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 DOCX ZIP 包，并重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()
