# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import xml.etree.ElementTree as ET

SPREADSHEETML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
SHARED_STRINGS_PATH = "xl/sharedStrings.xml"
UNDERLINE_TAG = f"{{{SPREADSHEETML_NS}}}u"
UNDERLINE_VAL_ATTRS = ("val", f"{{{SPREADSHEETML_NS}}}val")
VALID_UNDERLINE_VALUES = {
    "single",
    "double",
    "singleAccounting",
    "doubleAccounting",
    "none",
}


def normalize_xlsx_package(file_bytes: bytes) -> bytes:
    """在进入 openpyxl 前修复常见 XLSX 包级兼容问题。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as source:
            if SHARED_STRINGS_PATH not in source.namelist():
                return file_bytes

            rewritten_members: list[tuple[ZipInfo, bytes]] = []
            changed = False

            for info in source.infolist():
                member_data = source.read(info.filename)
                if info.filename == SHARED_STRINGS_PATH:
                    normalized_data = _normalize_shared_strings_xml(member_data)
                    if normalized_data != member_data:
                        changed = True
                    member_data = normalized_data
                rewritten_members.append((info, member_data))
    except BadZipFile as exc:
        raise ValueError("Invalid XLSX package: file is not a ZIP archive.") from exc

    if not changed:
        return file_bytes

    return _write_package(rewritten_members)


def _normalize_shared_strings_xml(xml_bytes: bytes) -> bytes:
    """规范化共享字符串 XML 中 openpyxl 无法接受的富文本属性。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return xml_bytes

    changed = False
    for underline in root.findall(f".//{UNDERLINE_TAG}"):
        if _drop_blank_underline_value(underline):
            changed = True

    if not changed:
        return xml_bytes

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _drop_blank_underline_value(underline: ET.Element) -> bool:
    """删除空白 underline val，保留 <u/> 的默认单下划线语义。"""
    for attr_name in UNDERLINE_VAL_ATTRS:
        attr_value = underline.attrib.get(attr_name)
        if attr_value is None:
            continue
        if attr_value.strip() == "":
            del underline.attrib[attr_name]
            return True
        if attr_value not in VALID_UNDERLINE_VALUES:
            return False
    return False


def _write_package(members: list[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 XLSX ZIP 包，并重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()
