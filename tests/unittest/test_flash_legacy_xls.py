from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
import struct

from bs4 import BeautifulSoup
import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import XlsModel
from mineru.model.flash.office.legacy import (
    LegacyOfficeEncryptedError,
    LegacyOfficeMissingPartError,
    LegacyOfficeResourceLimitError,
)
from mineru.model.flash.office.legacy.limits import MAX_RECORDS
from mineru.model.flash.office.xls import xls_converter as xls_converter_module
from mineru.model.flash.office.xls import parser as xls_parser
from mineru.model.flash.office.xls.number_format import format_number, format_text
from mineru.model.flash.office.xls.records import RecordBudget
from mineru.parser import parse
from mineru.types import BlockType, ImageBlock, MiddleJson, ModelJson, TableBlock

from _legacy_xls_test_utils import (
    SheetFixture,
    biff_record,
    build_biff5_xls,
    build_xls,
    continued_rich_sst,
    font_record,
    formula_number_cell,
    formula_string_cell,
    label_cell,
    labelsst_cell,
    merged_cells,
    rich_sst,
    url_hyperlink,
)
from _legacy_ppt_test_utils import _build_cfb


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_XLS = _PROJECT_ROOT / "demo" / "office_docs" / "xlsx_01.xls"


def test_xls_model_preserves_visible_empty_pages_and_skips_hidden_sheet() -> None:
    """验证可见空表保留页位而隐藏 sheet 不输出。"""

    file_bytes = build_xls(
        [
            SheetFixture("Visible", label_cell(0, 0, "visible")),
            SheetFixture("Empty"),
            SheetFixture("Secret", label_cell(0, 0, "secret"), visible=False),
        ]
    )
    stream = BytesIO(file_bytes)

    pages = XlsModel().predict(stream)

    assert not stream.closed
    assert pages == [[{"type": BlockType.TEXT, "content": "visible"}], []]


def test_backend_analyze_accepts_xls_and_async_contract() -> None:
    """验证同步与异步 Analyze 都返回严格 XLS 文档契约。"""

    file_bytes = build_xls([SheetFixture("Data", label_cell(0, 0, "value"))])
    middle_json, model_json = doc_analyze(file_bytes, file_suffix="xls")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="xls"))

    assert isinstance(model_json, ModelJson)
    assert isinstance(middle_json, MiddleJson)
    assert model_json.file_suffix == middle_json.file_suffix == "xls"
    assert model_json.effort == middle_json.effort == "flash"
    assert model_json.parse_mode == middle_json.parse_mode == "txt"
    assert model_json.is_full_document is middle_json.is_full_document is True
    assert async_model == model_json
    assert async_middle == middle_json


def test_xls_formula_cache_merge_and_hyperlink_flow_into_table() -> None:
    """验证缓存公式、合并结构和安全链接进入同一 HTML 表格。"""

    records = (
        label_cell(0, 0, "header")
        + label_cell(0, 1, "link")
        + formula_number_cell(1, 0, 21.0)
        + formula_string_cell(1, 1, "#NAME?")
        + url_hyperlink(0, 1, "link", "https://example.test/path")
        + merged_cells((2, 0, 2, 1))
        + label_cell(2, 0, "merged")
    )

    pages = XlsModel().predict(BytesIO(build_xls([SheetFixture("Data", records)])))
    table = next(block for block in pages[0] if block["type"] == BlockType.TABLE)
    soup = BeautifulSoup(table["content"], "html.parser")

    assert soup.get_text(" ", strip=True).split() == [
        "header",
        "link",
        "21",
        "#NAME?",
        "merged",
    ]
    assert soup.find("a")["href"] == "https://example.test/path"
    assert soup.find(string="merged").find_parent(["th", "td"])["colspan"] == "2"


def test_xls_rich_sst_uses_utf16_ranges_across_non_bmp_text() -> None:
    """验证非 BMP 字符的 UTF-16 rich run 边界不会偏移。"""

    globals_records = font_record(bold=True) + font_record(italic=True) + rich_sst([("A😀B", [(0, 1), (3, 2)])])
    pages = XlsModel().predict(
        BytesIO(
            build_xls(
                [SheetFixture("Rich", labelsst_cell(0, 0, 0))],
                globals_records=globals_records,
            )
        )
    )

    table = next(block for block in pages[0] if block["type"] == BlockType.TABLE)
    assert "<strong>A😀</strong>" in table["content"]
    assert "<em>B</em>" in table["content"]


def test_xls_sst_character_data_can_cross_continue_records() -> None:
    """验证 SST 在字符中间切入 CONTINUE 后重读压缩标志并继续 rich runs。"""

    globals_records = font_record(bold=True) + font_record(italic=True) + continued_rich_sst("A😀BC", [(0, 1), (3, 2)])
    pages = XlsModel().predict(
        BytesIO(
            build_xls(
                [SheetFixture("Rich", labelsst_cell(0, 0, 0))],
                globals_records=globals_records,
            )
        )
    )

    table = next(block for block in pages[0] if block["type"] == BlockType.TABLE)
    assert "<strong>A😀</strong>" in table["content"]
    assert "<em>BC</em>" in table["content"]


@pytest.mark.parametrize(
    ("format_code", "value", "expected"),
    [
        ("0.0%", 0.075, "7.5%"),
        ("#,##0.00", 1234.5, "1,234.50"),
        ('"$"#,##0.00', 1234.5, "$1,234.50"),
        ("0.00;(0.00)", -3.5, "(3.50)"),
        ("0.00E+00", 12345.0, "1.23E+04"),
        ("# ?/?", 5.25, "5 1/4"),
        ("# ?/8", 5.25, "5 2/8"),
        ("0.0,,", 12_345_678.0, "12.3"),
        ("0.00_);(0.00)", 3.5, "3.50 "),
        ('0.0\\ "m/s"', 3.51, "3.5 m/s"),
        ('"~"General" kg"', 1234.5, "~1234.5 kg"),
        ("yyyy-mm-dd", 46096.0, "2026-03-15"),
        ("[h]:mm:ss", 1.5, "36:00:00"),
    ],
)
def test_xls_number_formats_match_expected_visible_semantics(
    format_code: str,
    value: float,
    expected: str,
) -> None:
    """验证关键数值格式符合预期的稳定显示语义。"""

    assert format_number(value, format_code, date1904=False) == expected


def test_xls_text_format_and_unsafe_hyperlink_fallback() -> None:
    """验证文本 section 生效且危险链接降级为普通文本。"""

    assert format_text("hi", '0;0;0;"* "@" *"') == "* hi *"
    records = label_cell(0, 0, "unsafe") + url_hyperlink(
        0,
        0,
        "unsafe",
        "javascript:alert(1)",
    )
    pages = XlsModel().predict(BytesIO(build_xls([SheetFixture("Data", records)])))
    assert pages == [[{"type": BlockType.TEXT, "content": "unsafe"}]]
    assert xls_parser._sanitize_hyperlink_target("mailto:user@example.test") == ("mailto:user@example.test")
    assert xls_parser._sanitize_hyperlink_target("#Sheet2!A1") == "#Sheet2!A1"
    assert xls_parser._sanitize_hyperlink_target("../relative/path") == "../relative/path"
    assert xls_parser._sanitize_hyperlink_target("file:///tmp/local.xls") is None
    assert xls_parser._sanitize_hyperlink_target("custom:payload") is None


def test_xls_filepass_and_broken_boundsheet_behaviors() -> None:
    """验证加密硬失败，而损坏 sheet offset 可按 worksheet BOF 恢复。"""

    with pytest.raises(LegacyOfficeEncryptedError, match="password-protected"):
        XlsModel().predict(BytesIO(build_xls([SheetFixture("Data")], encrypted=True)))

    pages = XlsModel().predict(
        BytesIO(
            build_xls(
                [SheetFixture("Recovered", label_cell(0, 0, "ok"))],
                corrupt_first_offset=True,
            )
        )
    )
    assert pages == [[{"type": BlockType.TEXT, "content": "ok"}]]


def test_xls_encrypted_ooxml_marker_and_missing_workbook_are_stable_errors() -> None:
    """验证 OLE 加密包不会误判成 BIFF，缺失 Workbook/Book 返回稳定错误。"""

    encrypted = _build_cfb(
        [
            ("EncryptionInfo", b"marker"),
            ("EncryptedPackage", b"payload"),
        ]
    )
    with pytest.raises(LegacyOfficeEncryptedError, match="password-protected"):
        XlsModel().predict(BytesIO(encrypted))

    with pytest.raises(LegacyOfficeMissingPartError, match="Book"):
        XlsModel().predict(BytesIO(_build_cfb([("Other", b"payload")])))


def test_xls_biff5_codepage_and_hidden_rows_columns_are_preserved() -> None:
    """验证 BIFF5 codepage 降级及用户选择的隐藏行列保留策略。"""

    assert XlsModel().predict(BytesIO(build_biff5_xls("légacy"))) == [[{"type": BlockType.TEXT, "content": "légacy"}]]

    row = bytearray(16)
    struct.pack_into("<H", row, 0, 0)
    row[12] = 0x20
    colinfo = bytearray(12)
    struct.pack_into("<HH", colinfo, 0, 0, 0)
    colinfo[8] = 0x01
    records = biff_record(0x0208, bytes(row)) + biff_record(0x007D, bytes(colinfo)) + label_cell(0, 0, "still visible")

    assert XlsModel().predict(BytesIO(build_xls([SheetFixture("Data", records)]))) == [
        [{"type": BlockType.TEXT, "content": "still visible"}]
    ]


def test_xls_record_and_grid_limits_are_hard_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证记录访问与工作簿网格预算均不可静默绕过。"""

    budget = RecordBudget(count=MAX_RECORDS)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_records"):
        budget.charge()

    monkeypatch.setattr(xls_converter_module, "MAX_GRID_SLOTS", 0)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_grid_slots"):
        XlsModel().predict(BytesIO(build_xls([SheetFixture("Data", label_cell(0, 0, "value"))])))


def test_xls_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 XLS。"""

    path = tmp_path / "sample.xls"
    path.write_bytes(build_xls([SheetFixture("Data", label_cell(0, 0, "value"))]))

    result = parse(path, tier="flash")

    assert result.middle_json.file_suffix == "xls"
    assert result.pages[0].blocks[0].content == "value"


@pytest.mark.skipif(not _REAL_XLS.exists(), reason="real Office roundtrip fixture is local-only")
def test_real_xls_recovers_tables_charts_image_link_and_exports(tmp_path: Path) -> None:
    """验证真实 XLS 的三页结构、图表、图片、链接与 sidecar 闭包。"""

    middle_json, model_json = doc_analyze(_REAL_XLS.read_bytes(), file_suffix="xls")

    assert len(model_json.pages) == len(middle_json.pages) == 3
    assert model_json.file_suffix == middle_json.file_suffix == "xls"
    assert [[block.get("type") for block in page] for page in model_json.pages] == [
        [BlockType.PARAGRAPH_TITLE, BlockType.TABLE],
        [
            BlockType.PARAGRAPH_TITLE,
            BlockType.TABLE,
            BlockType.TABLE,
            BlockType.CHART,
            BlockType.TABLE,
            BlockType.CHART,
        ],
        [BlockType.PARAGRAPH_TITLE, BlockType.TABLE, BlockType.TABLE, BlockType.IMAGE],
    ]

    page_one_table = next(block for block in middle_json.pages[0].blocks if isinstance(block, TableBlock))
    page_one_soup = BeautifulSoup(page_one_table.content[0].content, "html.parser")
    page_one_rows = page_one_soup.find_all("tr")
    assert len(page_one_rows) == 9
    assert all(len(row.find_all(["th", "td"], recursive=False)) == 3 for row in page_one_rows)
    assert page_one_soup.find("a")["href"] == "http://www.baidu.com/"
    assert "21" in page_one_soup.get_text(" ", strip=True)
    assert "#NAME?" in page_one_soup.get_text(" ", strip=True)
    assert "(x+a)^n" in page_one_soup.get_text(" ", strip=True)

    chart_tables = [block.content[0].content for block in middle_json.pages[1].blocks if block.type == BlockType.CHART]
    assert [
        (
            len(BeautifulSoup(content, "html.parser").find_all("tr")),
            max(
                len(row.find_all(["th", "td"], recursive=False)) for row in BeautifulSoup(content, "html.parser").find_all("tr")
            ),
        )
        for content in chart_tables
    ] == [(5, 2), (9, 4)]

    page_three_tables = [block for block in middle_json.pages[2].blocks if isinstance(block, TableBlock)]
    assert len(page_three_tables) == 2
    assert all(
        len(
            [
                cell
                for cell in BeautifulSoup(table.content[0].content, "html.parser").find_all(["th", "td"])
                if cell.has_attr("rowspan") or cell.has_attr("colspan")
            ]
        )
        == 4
        for table in page_three_tables
    )
    assert sum(isinstance(block, ImageBlock) for block in middle_json.pages[2].blocks) == 1

    export_result = middle_json.export(tmp_path / "export")
    exported = export_result.middle_json.model_dump(mode="json", exclude_none=True)
    assert export_result.json_path.exists()
    assert len(export_result.image_paths) == 1
    assert export_result.image_paths[0].exists()
    assert "image_base64" not in str(exported)
