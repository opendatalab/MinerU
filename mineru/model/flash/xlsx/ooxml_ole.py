# Copyright (c) Opendatalab. All rights reserved.

"""读取 XLSX worksheet 中的 Equation.3 OLE 对象、anchor 与预览。"""

from __future__ import annotations

from dataclasses import dataclass
import posixpath
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from loguru import logger

from mineru.model.flash.legacy_office.errors import LegacyOfficeResourceLimitError
from mineru.model.flash.legacy_office.limits import MAX_ENTRY_BYTES
from mineru.model.flash.office.image import serialize_office_image
from mineru.model.flash.office.ooxml_equation import (
    OoxmlEquationDecoder,
    is_equation_3_prog_id,
)

WORKBOOK_PART = "xl/workbook.xml"
RELATIONSHIP_NS_SUFFIX = "/relationships"


@dataclass(frozen=True, slots=True)
class XlsxOleEquationArtifact:
    """一个已绑定 worksheet anchor 的 Equation.3 公式或预览。"""

    row: int | None
    col: int | None
    latex: str | None
    preview_base64: str | None
    order: int
    shape_id: str | None


@dataclass(frozen=True, slots=True)
class _Relationship:
    """一个经过安全路径规范化的 OPC relationship。"""

    target: str | None
    reltype: str
    external: bool


def _local_name(value: object) -> str:
    """返回 XML QName 或普通属性名的本地名称。"""

    text = str(value)
    return text.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _relationship_id(element: ET.Element) -> str | None:
    """读取 r:id，避免把普通 shape id 误当 relationship。"""

    for attribute, value in element.attrib.items():
        if not attribute.endswith("}id"):
            continue
        namespace = attribute[1:].split("}", 1)[0] if attribute.startswith("{") else ""
        if namespace.rstrip("/").endswith(RELATIONSHIP_NS_SUFFIX):
            return value
    return None


def _relationship_embed_id(element: ET.Element) -> str | None:
    """读取 DrawingML a:blip 的 r:embed relationship id。"""

    for attribute, value in element.attrib.items():
        if not attribute.endswith("}embed"):
            continue
        namespace = attribute[1:].split("}", 1)[0] if attribute.startswith("{") else ""
        if namespace.rstrip("/").endswith(RELATIONSHIP_NS_SUFFIX):
            return value
    return None


def _relationship_part_name(source_part: str) -> str:
    """由源 part 生成同目录的 .rels part 名称。"""

    directory, basename = posixpath.split(source_part)
    return posixpath.join(directory, "_rels", f"{basename}.rels")


def _resolve_internal_target(source_part: str, target: str) -> str | None:
    """解析 OPC 内部 target，并拒绝反斜杠和越界路径。"""

    candidate = target.strip()
    if not candidate or "\\" in candidate:
        return None
    if candidate.startswith("/"):
        normalized = posixpath.normpath(candidate.lstrip("/"))
    else:
        normalized = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_part), candidate)
        )
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        return None
    return normalized


def _read_xml(source: ZipFile, part_name: str) -> ET.Element | None:
    """尽力读取一个 XML part，坏可选 part 返回空。"""

    if part_name not in source.namelist():
        return None
    try:
        return ET.fromstring(source.read(part_name))
    except (ET.ParseError, KeyError, OSError, RuntimeError):
        return None


def _relationships(source: ZipFile, source_part: str) -> dict[str, _Relationship]:
    """读取源 part 的 relationships，并规范化所有内部 target。"""

    root = _read_xml(source, _relationship_part_name(source_part))
    if root is None:
        return {}
    result: dict[str, _Relationship] = {}
    for element in root.iter():
        if _local_name(element.tag) != "Relationship":
            continue
        relationship_id = element.get("Id")
        if not relationship_id:
            continue
        external = (element.get("TargetMode") or "").casefold() == "external"
        target = None
        if not external:
            target = _resolve_internal_target(
                source_part,
                element.get("Target") or "",
            )
        result[relationship_id] = _Relationship(
            target=target,
            reltype=element.get("Type") or "",
            external=external,
        )
    return result


def _read_member_bounded(source: ZipFile, part_name: str | None) -> bytes | None:
    """按共享单 part 上限读取 XLSX ZIP 成员。"""

    if part_name is None:
        return None
    try:
        info = source.getinfo(part_name)
    except KeyError:
        return None
    if info.file_size > MAX_ENTRY_BYTES:
        raise LegacyOfficeResourceLimitError(
            f"XLSX embedded part exceeds max_entry_bytes={MAX_ENTRY_BYTES}: {part_name}"
        )
    payload = source.read(info)
    if len(payload) > MAX_ENTRY_BYTES:
        raise LegacyOfficeResourceLimitError(
            f"XLSX embedded part exceeds max_entry_bytes={MAX_ENTRY_BYTES}: {part_name}"
        )
    return payload


def workbook_sheet_parts(source: ZipFile) -> dict[str, str]:
    """按 workbook sheet 名称建立 worksheet part 映射。"""

    workbook = _read_xml(source, WORKBOOK_PART)
    if workbook is None:
        return {}
    relationships = _relationships(source, WORKBOOK_PART)
    result: dict[str, str] = {}
    for sheet in workbook.iter():
        if _local_name(sheet.tag) != "sheet":
            continue
        name = sheet.get("name")
        relationship_id = _relationship_id(sheet)
        relationship = relationships.get(relationship_id or "")
        if (
            not name
            or relationship is None
            or relationship.external
            or relationship.target is None
            or not relationship.reltype.rstrip("/").casefold().endswith("/worksheet")
        ):
            continue
        result[name] = relationship.target
    return result


def package_has_sheet_ole_objects(
    source: ZipFile,
    worksheet_parts: dict[str, str],
) -> bool:
    """判断任一 worksheet 是否包含 openpyxl 无法安全忽略的 oleObjects。"""

    for part_name in worksheet_parts.values():
        root = _read_xml(source, part_name)
        if root is None:
            continue
        if any(_local_name(node.tag) == "oleObjects" for node in root):
            return True
    return False


def _child_int(element: ET.Element, name: str) -> int | None:
    """读取指定本地名称的首个整数子元素。"""

    for child in element.iter():
        if _local_name(child.tag) != name or child.text is None:
            continue
        try:
            return int(child.text.strip())
        except ValueError:
            return None
    return None


def _anchor_from_object_properties(ole_object: ET.Element) -> tuple[int, int] | None:
    """优先从 objectPr/anchor/from 读取零基 row/col。"""

    for object_properties in ole_object:
        if _local_name(object_properties.tag) != "objectPr":
            continue
        for anchor in object_properties:
            if _local_name(anchor.tag) != "anchor":
                continue
            from_marker = next(
                (
                    child
                    for child in anchor
                    if _local_name(child.tag) == "from"
                ),
                None,
            )
            if from_marker is None:
                continue
            row = _child_int(from_marker, "row")
            col = _child_int(from_marker, "col")
            if row is not None and col is not None and row >= 0 and col >= 0:
                return row, col
    return None


def _object_preview_relationship_id(ole_object: ET.Element) -> str | None:
    """读取 objectPr 指向缓存预览图片的 relationship id。"""

    for child in ole_object:
        if _local_name(child.tag) == "objectPr":
            return _relationship_id(child)
    return None


def _shape_id_matches(raw_id: str | None, shape_id: str) -> bool:
    """兼容 DrawingML 数字 id 和 VML `_x0000_sNNN` 写法。"""

    if raw_id is None:
        return False
    if raw_id == shape_id:
        return True
    match = re.search(r"(?:^|[_s])(\d+)$", raw_id)
    return bool(match and match.group(1) == shape_id)


def _drawing_shape_info(
    source: ZipFile,
    drawing_part: str,
    shape_id: str,
) -> tuple[tuple[int, int] | None, str | None]:
    """从 DrawingML anchor 按 cNvPr id 查找坐标和预览 part。"""

    root = _read_xml(source, drawing_part)
    if root is None:
        return None, None
    relationships = _relationships(source, drawing_part)
    for anchor in root:
        if _local_name(anchor.tag) not in {"oneCellAnchor", "twoCellAnchor", "absoluteAnchor"}:
            continue
        if not any(
            _local_name(node.tag) == "cNvPr"
            and _shape_id_matches(node.get("id"), shape_id)
            for node in anchor.iter()
        ):
            continue
        from_marker = next(
            (
                child
                for child in anchor
                if _local_name(child.tag) == "from"
            ),
            None,
        )
        coordinate = None
        if from_marker is not None:
            row = _child_int(from_marker, "row")
            col = _child_int(from_marker, "col")
            if row is not None and col is not None and row >= 0 and col >= 0:
                coordinate = (row, col)
        for node in anchor.iter():
            if _local_name(node.tag) != "blip":
                continue
            relationship = relationships.get(_relationship_embed_id(node) or "")
            if relationship is not None and not relationship.external:
                return coordinate, relationship.target
        return coordinate, None
    return None, None


def _vml_shape_info(
    source: ZipFile,
    drawing_part: str,
    shape_id: str,
) -> tuple[tuple[int, int] | None, str | None]:
    """从 VML ClientData/Anchor 按 shape id 查找坐标和预览 part。"""

    root = _read_xml(source, drawing_part)
    if root is None:
        return None, None
    relationships = _relationships(source, drawing_part)
    for shape in root.iter():
        if _local_name(shape.tag) != "shape" or not _shape_id_matches(
            shape.get("id"),
            shape_id,
        ):
            continue
        coordinate = None
        client_data = next(
            (node for node in shape.iter() if _local_name(node.tag) == "ClientData"),
            None,
        )
        if client_data is not None:
            anchor_text = next(
                (
                    node.text
                    for node in client_data
                    if _local_name(node.tag) == "Anchor" and node.text
                ),
                None,
            )
            if anchor_text:
                try:
                    values = [int(value.strip()) for value in anchor_text.split(",")]
                except ValueError:
                    values = []
                if len(values) >= 4 and values[0] >= 0 and values[2] >= 0:
                    coordinate = (values[2], values[0])
            if coordinate is None:
                row = _child_int(client_data, "Row")
                col = _child_int(client_data, "Column")
                if row is not None and col is not None and row >= 0 and col >= 0:
                    coordinate = (row, col)
        for node in shape.iter():
            if _local_name(node.tag) != "imagedata":
                continue
            relationship = relationships.get(_relationship_id(node) or "")
            if relationship is not None and not relationship.external:
                return coordinate, relationship.target
        return coordinate, None
    return None, None


def _shape_anchor_and_preview(
    source: ZipFile,
    worksheet_relationships: dict[str, _Relationship],
    shape_id: str,
) -> tuple[tuple[int, int] | None, str | None]:
    """按 DrawingML 后 VML 的顺序恢复 shape anchor 与预览。"""

    candidates = [
        relationship
        for relationship in worksheet_relationships.values()
        if not relationship.external and relationship.target is not None
    ]
    for relationship in candidates:
        reltype = relationship.reltype.rstrip("/").casefold()
        if reltype.endswith("/drawing"):
            coordinate, preview = _drawing_shape_info(
                source,
                relationship.target or "",
                shape_id,
            )
            if coordinate is not None or preview is not None:
                return coordinate, preview
    for relationship in candidates:
        reltype = relationship.reltype.rstrip("/").casefold()
        if reltype.endswith("/vmldrawing"):
            coordinate, preview = _vml_shape_info(
                source,
                relationship.target or "",
                shape_id,
            )
            if coordinate is not None or preview is not None:
                return coordinate, preview
    return None, None


def _preview_data_uri(source: ZipFile, part_name: str | None) -> str | None:
    """读取并按现有 Office 图片策略序列化 OLE 缓存预览。"""

    payload = _read_member_bounded(source, part_name)
    if payload is None:
        return None
    return serialize_office_image(
        payload,
        part_name=part_name,
        content_type=None,
    )


def read_sheet_equation_artifacts(
    source: ZipFile,
    worksheet_part: str,
    decoder: OoxmlEquationDecoder,
) -> list[XlsxOleEquationArtifact]:
    """读取一个 worksheet 的 Equation.3 对象并绑定公式或图片回退。"""

    worksheet = _read_xml(source, worksheet_part)
    if worksheet is None:
        return []
    relationships = _relationships(source, worksheet_part)
    artifacts: list[XlsxOleEquationArtifact] = []
    for order, ole_object in enumerate(
        node for node in worksheet.iter() if _local_name(node.tag) == "oleObject"
    ):
        prog_id = ole_object.get("progId") or ole_object.get("ProgID")
        if not is_equation_3_prog_id(prog_id):
            continue
        relationship = relationships.get(_relationship_id(ole_object) or "")
        is_linked = bool(ole_object.get("link")) or (
            relationship is not None and relationship.external
        )
        draw_aspect = (ole_object.get("dvAspect") or "").casefold()
        show_as_icon = "icon" in draw_aspect

        latex = None
        if (
            not is_linked
            and relationship is not None
            and relationship.target is not None
            and relationship.reltype.rstrip("/").casefold().endswith("/oleobject")
        ):
            blob = _read_member_bounded(source, relationship.target)
            latex = decoder.decode(
                blob,
                prog_id=prog_id,
                show_as_icon=show_as_icon,
            )
            if latex is None and not show_as_icon:
                logger.warning(
                    "XLSX_MTEF_FALLBACK: worksheet={!r}, relationship={!r} has an invalid or unsupported Equation.3 object",
                    worksheet_part,
                    _relationship_id(ole_object),
                )

        coordinate = _anchor_from_object_properties(ole_object)
        preview_relationship = relationships.get(
            _object_preview_relationship_id(ole_object) or ""
        )
        preview_part = (
            preview_relationship.target
            if preview_relationship is not None and not preview_relationship.external
            else None
        )
        shape_coordinate, shape_preview = _shape_anchor_and_preview(
            source,
            relationships,
            ole_object.get("shapeId") or "",
        )
        coordinate = coordinate or shape_coordinate
        preview_part = preview_part or shape_preview
        preview = _preview_data_uri(source, preview_part)

        if latex is None and preview is None:
            continue
        artifacts.append(
            XlsxOleEquationArtifact(
                row=coordinate[0] if coordinate is not None else None,
                col=coordinate[1] if coordinate is not None else None,
                latex=latex,
                preview_base64=preview,
                order=order,
                shape_id=ole_object.get("shapeId"),
            )
        )
    return artifacts
