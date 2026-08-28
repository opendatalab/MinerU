from __future__ import annotations

import struct
import zlib

from bs4 import BeautifulSoup

from mineru.backend.analyze import doc_analyze
from mineru.model.flash.office.ppt import parser as ppt_parser
from mineru.model.flash.office.ppt.records import PptRecord
from mineru.model.flash.office.xls.embedded_chart import extract_embedded_chart_html
from mineru.types import BlockType, ChartBlock

from _legacy_ppt_test_utils import _build_cfb
from _legacy_xls_test_utils import biff_bof, biff_record, label_cell, number_cell
from _span_test_utils import inline_text


def _boundsheet(offset: int, name: str, sheet_type: int, *, visible: bool = True) -> bytes:
    """构造带指定工作表类型的 BoundSheet8。"""

    encoded = name.encode("utf-16le")
    payload = struct.pack("<IBB", offset, 0 if visible else 1, sheet_type)
    payload += struct.pack("<BB", len(encoded) // 2, 1) + encoded
    return biff_record(0x0085, payload)


def _chart_brai(row_last: int, col_last: int) -> bytes:
    """构造引用第二个 BoundSheet 数据范围的 PtgArea3d BRAI。"""

    tokens = struct.pack("<BH4H", 0x3B, 0, 0, row_last, 0, col_last)
    payload = struct.pack("<BBHHH", 1, 2, 0, 0, len(tokens)) + tokens
    return biff_record(0x1051, payload)


def _excel_chart_workbook() -> bytes:
    """构造一个独立 Chart1 引用 Sheet1!A1:B3 的 BIFF8 Workbook。"""

    prefix = biff_bof(0x0005)
    prefix += biff_record(0x0042, struct.pack("<H", 1200))
    prefix += biff_record(0x01AE, struct.pack("<HH", 2, 0x0401))
    prefix += biff_record(0x0017, struct.pack("<H3H", 1, 0, 1, 1))
    window = bytearray(18)
    struct.pack_into("<H", window, 10, 0)
    prefix += biff_record(0x003D, bytes(window))

    chart_stream = biff_bof(0x0020) + _chart_brai(2, 1) + biff_record(0x000A)
    data_records = (
        label_cell(0, 0, "类别")
        + label_cell(0, 1, "系列")
        + label_cell(1, 0, "A")
        + number_cell(1, 1, 1.5)
        + label_cell(2, 0, "B")
        + number_cell(2, 1, 2.5)
    )
    data_stream = biff_bof(0x0010) + data_records + biff_record(0x000A)
    placeholders = _boundsheet(0, "Chart1", 0x02) + _boundsheet(0, "Sheet1", 0x00)
    globals_size = len(prefix) + len(placeholders) + len(biff_record(0x000A))
    chart_offset = globals_size
    data_offset = chart_offset + len(chart_stream)
    directory = _boundsheet(chart_offset, "Chart1", 0x02) + _boundsheet(
        data_offset,
        "Sheet1",
        0x00,
    )
    return prefix + directory + biff_record(0x000A) + chart_stream + data_stream


def _graph_label(row: int, col: int, text: str) -> bytes:
    """构造 MS-OGRAPH Label datasheet 单元格。"""

    encoded = text.encode("utf-16le")
    payload = struct.pack("<HHBHBB", row, col, 0, 0, len(text), 1) + encoded
    return biff_record(0x0204, payload)


def _graph_number(row: int, col: int, value: float) -> bytes:
    """构造 MS-OGRAPH Number datasheet 单元格。"""

    return biff_record(0x0003, struct.pack("<HHBHd", row, col, 0, 0, value))


def _graph_chart_workbook(*, exclude_last_row: bool = False) -> bytes:
    """构造一个含两行两列 datasheet 的 MS-OGRAPH Workbook。"""

    globals_stream = biff_bof(0x0005, version=0x0680) + biff_record(0x000A)
    datasheet = (
        biff_record(0x1052, b"\x00" * 4)
        + _graph_label(0, 0, "类别")
        + _graph_label(0, 1, "系列")
        + _graph_label(1, 0, "A")
        + _graph_number(1, 1, 3.5)
    )
    if exclude_last_row:
        datasheet += _graph_label(2, 0, "B") + _graph_number(2, 1, 4.5)
        datasheet += biff_record(0x1053, struct.pack("<2H", 0, 2))
    chart_stream = biff_bof(0x8000, version=0x0680) + datasheet + biff_record(0x000A)
    return globals_stream + chart_stream


def _table_rows(content: str) -> list[list[str]]:
    """把 chart HTML 归一化为二维文本矩阵。"""

    soup = BeautifulSoup(content, "html.parser")
    return [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)] for row in soup.find_all("tr")
    ]


def test_excel_chart_workbook_uses_chart_sheet_brai_selection() -> None:
    """验证 Excel.Chart 独立 chart sheet 精确恢复引用范围。"""

    content = extract_embedded_chart_html(_excel_chart_workbook())

    assert content is not None
    assert _table_rows(content) == [["类别", "系列"], ["A", "1.5"], ["B", "2.5"]]


def test_msgraph_chart_workbook_recovers_embedded_datasheet() -> None:
    """验证 MS-OGRAPH 专用 Label/Number 记录恢复为 HTML 表格。"""

    content = extract_embedded_chart_html(_graph_chart_workbook())

    assert content is not None
    assert _table_rows(content) == [["类别", "系列"], ["A", "3.5"]]


def test_msgraph_chart_applies_excluded_row_boundaries() -> None:
    """验证 MS-OGRAPH ExcludeRows 不把未参与 chart 的 datasheet 行输出。"""

    content = extract_embedded_chart_html(_graph_chart_workbook(exclude_last_row=True))

    assert content is not None
    assert _table_rows(content) == [["类别", "系列"], ["A", "3.5"]]


def test_xls_standalone_chart_sheet_keeps_workbook_order() -> None:
    """验证独立 Chart Sheet 按 BoundSheet 顺序输出单独逻辑页。"""

    middle, model = doc_analyze(
        _build_cfb([("Workbook", _excel_chart_workbook())]),
        file_suffix="xls",
    )

    assert len(model.pages) == len(middle.pages) == 2
    assert [block.get("type") for block in model.pages[0]] == [
        BlockType.PARAGRAPH_TITLE,
        BlockType.CHART,
    ]
    assert isinstance(middle.pages[0].blocks[1], ChartBlock)
    assert _table_rows(middle.pages[0].blocks[1].content[0].content) == [
        ["类别", "系列"],
        ["A", "1.5"],
        ["B", "2.5"],
    ]
    assert inline_text(model.pages[1][0]["content"]) == "Sheet1"


def test_excel_chart_single_visible_sheet_fallback_uses_nonempty_extent() -> None:
    """验证没有可解析 chart sheet 时仅对唯一可见 worksheet 使用安全回退。"""

    workbook = _excel_chart_workbook().replace(_chart_brai(2, 1), biff_record(0x1051, b"\x00" * 8))
    content = extract_embedded_chart_html(workbook)

    assert content is not None
    assert _table_rows(content) == [["类别", "系列"], ["A", "1.5"], ["B", "2.5"]]


def test_ppt_sync_flush_chart_storage_is_accepted_only_at_declared_cfb_size() -> None:
    """验证真实 Office 风格无 trailer 压缩流在严格 CFB 门下恢复。"""

    storage = _build_cfb([("Workbook", _excel_chart_workbook())])
    compressor = zlib.compressobj()
    compressed = compressor.compress(storage) + compressor.flush(zlib.Z_SYNC_FLUSH)
    record = PptRecord(
        offset=0,
        version=0,
        instance=1,
        record_type=ppt_parser.RT_EXTERNAL_OLE_OBJECT_STG,
        payload=struct.pack("<I", len(storage)) + compressed,
    )

    assert ppt_parser._decompress_ole_storage(record) == storage

    truncated = PptRecord(
        offset=0,
        version=0,
        instance=1,
        record_type=ppt_parser.RT_EXTERNAL_OLE_OBJECT_STG,
        payload=struct.pack("<I", len(storage) - 1) + compressed,
    )
    assert ppt_parser._decompress_ole_storage(truncated) is None
