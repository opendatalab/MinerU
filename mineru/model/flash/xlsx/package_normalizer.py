# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO
import re
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import xml.etree.ElementTree as ET

SPREADSHEETML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
SHARED_STRINGS_PATH = "xl/sharedStrings.xml"
STYLES_PATH = "xl/styles.xml"
WORKSHEET_PREFIX = "xl/worksheets/"
WORKSHEET_SUFFIX = ".xml"
UNDERLINE_TAG = f"{{{SPREADSHEETML_NS}}}u"
FILLS_TAG = f"{{{SPREADSHEETML_NS}}}fills"
FILL_TAG = f"{{{SPREADSHEETML_NS}}}fill"
PATTERN_FILL_TAG = f"{{{SPREADSHEETML_NS}}}patternFill"
AUTOFILTER_TAG = f"{{{SPREADSHEETML_NS}}}autoFilter"
DIMENSION_TAG = f"{{{SPREADSHEETML_NS}}}dimension"
SHEET_DATA_TAG = f"{{{SPREADSHEETML_NS}}}sheetData"
CELL_TAG = f"{{{SPREADSHEETML_NS}}}c"
UNDERLINE_VAL_ATTRS = ("val", f"{{{SPREADSHEETML_NS}}}val")
VALID_UNDERLINE_VALUES = {
    "single",
    "double",
    "singleAccounting",
    "doubleAccounting",
    "none",
}
ROW_ONLY_RANGE_RE = re.compile(r"^\$?([1-9][0-9]*):\$?([1-9][0-9]*)$")
CELL_REF_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?[1-9][0-9]*$")
CELL_RANGE_RE = re.compile(
    r"^\$?([A-Za-z]{1,3})\$?[1-9][0-9]*:\$?([A-Za-z]{1,3})\$?[1-9][0-9]*$"
)
WHOLE_COLUMN_RANGE_RE = re.compile(r"^\$?([A-Za-z]{1,3}):\$?([A-Za-z]{1,3})$")
MAX_EXCEL_COLUMN = "XFD"


def normalize_xlsx_package(file_bytes: bytes) -> bytes:
    """在进入 openpyxl 前修复常见 XLSX 包级兼容问题。"""
    try:
        with ZipFile(BytesIO(file_bytes)) as source:
            rewritten_members: list[tuple[ZipInfo, bytes]] = []
            changed = False

            for info in source.infolist():
                member_data = source.read(info.filename)
                normalized_data = _normalize_xlsx_member(info.filename, member_data)
                if normalized_data != member_data:
                    changed = True
                member_data = normalized_data
                rewritten_members.append((info, member_data))
    except BadZipFile as exc:
        raise ValueError("Invalid XLSX package: file is not a ZIP archive.") from exc

    if not changed:
        return file_bytes

    return _write_package(rewritten_members)


def _normalize_xlsx_member(member_name: str, member_data: bytes) -> bytes:
    """根据 XLSX 包内成员路径分发 XML 兼容性规范化逻辑。"""
    if member_name == SHARED_STRINGS_PATH:
        return _normalize_shared_strings_xml(member_data)
    if member_name == STYLES_PATH:
        return _normalize_styles_xml(member_data)
    if _is_worksheet_xml(member_name):
        return _normalize_worksheet_xml(member_data)
    return member_data


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


def _normalize_styles_xml(xml_bytes: bytes) -> bytes:
    """规范化 styles.xml 中 openpyxl 无法接受的空 fill 节点。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return xml_bytes

    changed = False
    fills = root.find(FILLS_TAG)
    if fills is not None:
        for fill in fills.findall(FILL_TAG):
            if _ensure_fill_has_pattern(fill):
                changed = True

    if not changed:
        return xml_bytes

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _normalize_worksheet_xml(xml_bytes: bytes) -> bytes:
    """规范化 worksheet XML 中会阻断 openpyxl 加载的行范围筛选器。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return xml_bytes

    changed = False
    for auto_filter in root.findall(f".//{AUTOFILTER_TAG}"):
        if _normalize_auto_filter_ref(root, auto_filter):
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


def _ensure_fill_has_pattern(fill: ET.Element) -> bool:
    """为空 fill 补一个空 patternFill，保留原 fillId 顺序和样式引用。"""
    if len(fill):
        return False

    fill.append(ET.Element(PATTERN_FILL_TAG))
    return True


def _normalize_auto_filter_ref(root: ET.Element, auto_filter: ET.Element) -> bool:
    """将 autoFilter 的纯行范围 ref 扩展为 openpyxl 可接受的单元格范围。"""
    ref = auto_filter.attrib.get("ref")
    if not ref:
        return False

    rows = _parse_row_only_range(ref)
    if rows is None:
        return False

    min_col, max_col = _resolve_worksheet_column_bounds(root)
    auto_filter.set("ref", f"{min_col}{rows[0]}:{max_col}{rows[1]}")
    return True


def _parse_row_only_range(ref: str) -> tuple[int, int] | None:
    """解析 Excel 行范围写法，仅接受正整数且起止顺序有效的范围。"""
    match = ROW_ONLY_RANGE_RE.match(ref)
    if match is None:
        return None

    min_row, max_row = int(match.group(1)), int(match.group(2))
    if min_row > max_row:
        return None
    return min_row, max_row


def _resolve_worksheet_column_bounds(root: ET.Element) -> tuple[str, str]:
    """优先从 dimension 获取列边界，失败时退回扫描 sheetData。"""
    dimension = root.find(DIMENSION_TAG)
    if dimension is not None:
        columns = _column_bounds_from_ref(dimension.attrib.get("ref", ""))
        if columns is not None:
            return columns

    columns = _column_bounds_from_sheet_data(root)
    if columns is not None:
        return columns

    return "A", MAX_EXCEL_COLUMN


def _column_bounds_from_ref(ref: str) -> tuple[str, str] | None:
    """从 dimension/ref 这类范围字符串中解析起止列。"""
    ref = ref.strip()
    if not ref:
        return None

    range_match = CELL_RANGE_RE.match(ref)
    if range_match is not None:
        return (
            range_match.group(1).upper(),
            range_match.group(2).upper(),
        )

    cell_match = CELL_REF_RE.match(ref)
    if cell_match is not None:
        column = cell_match.group(1).upper()
        return column, column

    whole_column_match = WHOLE_COLUMN_RANGE_RE.match(ref)
    if whole_column_match is not None:
        return (
            whole_column_match.group(1).upper(),
            whole_column_match.group(2).upper(),
        )

    return None


def _column_bounds_from_sheet_data(root: ET.Element) -> tuple[str, str] | None:
    """扫描 sheetData 里的单元格引用，推导工作表真实列范围。"""
    sheet_data = root.find(SHEET_DATA_TAG)
    if sheet_data is None:
        return None

    column_indexes = []
    for cell in sheet_data.findall(f".//{CELL_TAG}"):
        column = _column_from_cell_ref(cell.attrib.get("r", ""))
        if column is not None:
            column_indexes.append(_column_to_index(column))

    if not column_indexes:
        return None

    return (
        _index_to_column(min(column_indexes)),
        _index_to_column(max(column_indexes)),
    )


def _column_from_cell_ref(cell_ref: str) -> str | None:
    """从 A1 形式的单元格引用中提取列名。"""
    match = CELL_REF_RE.match(cell_ref)
    if match is None:
        return None
    return match.group(1).upper()


def _column_to_index(column: str) -> int:
    """将 Excel 列名转换为 1-based 数字索引。"""
    index = 0
    for char in column.upper():
        if not "A" <= char <= "Z":
            return 0
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def _index_to_column(index: int) -> str:
    """将 1-based 数字索引转换为 Excel 列名。"""
    column = []
    while index:
        index, remainder = divmod(index - 1, 26)
        column.append(chr(ord("A") + remainder))
    return "".join(reversed(column))


def _is_worksheet_xml(member_name: str) -> bool:
    """判断包内成员是否为普通 worksheet XML。"""
    return (
        member_name.startswith(WORKSHEET_PREFIX)
        and member_name.endswith(WORKSHEET_SUFFIX)
        and not member_name.startswith(f"{WORKSHEET_PREFIX}_rels/")
    )


def _write_package(members: list[tuple[ZipInfo, bytes]]) -> bytes:
    """把规范化后的成员重新写成 XLSX ZIP 包，并重新计算 CRC。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for info, member_data in members:
            target.writestr(info, member_data)
    return output.getvalue()
