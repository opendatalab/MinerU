from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
import struct

from bs4 import BeautifulSoup
import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import PptModel
from mineru.model.flash._shared.hyperlink import sanitize_hyperlink_target
from mineru.model.flash.office.errors import LegacyOfficeEncryptedError, LegacyOfficeResourceLimitError
from mineru.model.flash.office.ppt import parser as ppt_parser
from mineru.model.flash.office.ppt.models import PptPresentation, PptSlide
from mineru.model.flash.office.ppt.ppt_converter import PptConverter
from mineru.model.flash.office.ppt.records import PptRecord, RecordBudget
from mineru.model.flash.office.ppt.style_text import CharacterRun, StyleRuns
from mineru.parser import parse
from mineru.types import BlockType, ChartBlock, ImageBlock, MiddleJson, ModelJson, TableBlock

from _legacy_ppt_test_utils import build_deep_nested_ppt, build_multimaster_ppt, build_sparse_notes_ppt
from _span_test_utils import inline, inline_text


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_PPT = _PROJECT_ROOT / "demo" / "office_docs" / "pptx_01.ppt"


def test_ppt_model_preserves_slide_pages_and_sparse_notes() -> None:
    """验证旧版 PPT 始终逐 slide 分页，并按 slide id 绑定稀疏备注。"""

    pages = PptModel().predict(BytesIO(build_sparse_notes_ppt()))

    assert len(pages) == 2
    assert pages[0] == [{"type": BlockType.TEXT, "content": inline("First slide text")}]
    assert pages[1] == [
        {"type": BlockType.TEXT, "content": inline("Second slide text")},
        {"type": BlockType.PAGE_FOOTNOTE, "content": inline("Notes for the second slide")},
    ]


def test_ppt_converter_preserves_empty_slide_positions() -> None:
    """验证空白或隐藏 slide 仍在 model-list 中保留对应空页。"""

    presentation = PptPresentation(slides=[PptSlide(slide_id=1), PptSlide(slide_id=2, hidden=True)])

    assert PptConverter._presentation_to_pages(presentation) == [[], []]


def test_backend_analyze_accepts_ppt_and_async_contract() -> None:
    """验证同步与异步 Backend Analyze 均返回严格 PPT 文档契约。"""

    file_bytes = build_sparse_notes_ppt()
    middle_json, model_json = doc_analyze(file_bytes, file_suffix="ppt")
    async_middle_json, async_model_json = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="ppt"))

    assert isinstance(model_json, ModelJson)
    assert isinstance(middle_json, MiddleJson)
    assert model_json.file_suffix == "ppt"
    assert middle_json.file_suffix == "ppt"
    assert model_json.effort == middle_json.effort == "flash"
    assert model_json.parse_mode == middle_json.parse_mode == "txt"
    assert model_json.is_full_document is middle_json.is_full_document is True
    assert [page.page_idx for page in middle_json.pages] == [0, 1]
    assert async_middle_json == middle_json
    assert async_model_json == model_json


def test_ppt_model_applies_per_slide_master_styles() -> None:
    """验证每页按自身 master id 继承 bullet、粗体和斜体，而不是固定首个母版。"""

    pages = PptModel().predict(BytesIO(build_multimaster_ppt()))

    assert pages == [
        [
            {
                "type": BlockType.LIST,
                "attribute": "unordered",
                "ilevel": 0,
                "content": [
                    {
                        "type": BlockType.TEXT,
                        "content": inline("Alpha master body text", styles=["bold"]),
                    }
                ],
            }
        ],
        [
            {
                "type": BlockType.TEXT,
                "content": inline("Beta master body text", styles=["italic"]),
            }
        ],
    ]


def test_ppt_record_depth_is_a_hard_resource_limit() -> None:
    """验证恶意深层 PPT records 在递归前触发固定资源限制。"""

    with pytest.raises(LegacyOfficeResourceLimitError, match="max_record_depth"):
        PptModel().predict(BytesIO(build_deep_nested_ppt()))


def test_ppt_encryption_marker_is_rejected_before_record_parsing() -> None:
    """验证 Current User 加密标志在读取文档记录前返回稳定错误。"""

    current_user = bytearray(20)
    struct.pack_into("<H", current_user, 2, 0x0FF6)
    struct.pack_into("<I", current_user, 12, 0xF3D1_C4DF)

    with pytest.raises(LegacyOfficeEncryptedError, match="password-protected"):
        ppt_parser.parse_ppt_document(b"not parsed", current_user=bytes(current_user))


def test_ppt_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 PPT。"""

    path = tmp_path / "sample.ppt"
    path.write_bytes(build_sparse_notes_ppt())

    result = parse(path, tier="flash")

    assert result.middle_json.file_suffix == "ppt"
    assert len(result.pages) == 2


def test_safe_ppt_hyperlink_schemes_are_explicit() -> None:
    """验证外链白名单保留 Web/邮件链接并拒绝本地或脚本目标。"""

    assert sanitize_hyperlink_target("https://example.com/a", allowed_schemes=ppt_parser._ALLOWED_LINK_SCHEMES) == (
        "https://example.com/a"
    )
    assert sanitize_hyperlink_target("mailto:user@example.com", allowed_schemes=ppt_parser._ALLOWED_LINK_SCHEMES) == (
        "mailto:user@example.com"
    )
    assert sanitize_hyperlink_target("file:///tmp/a", allowed_schemes=ppt_parser._ALLOWED_LINK_SCHEMES) is None
    assert sanitize_hyperlink_target("javascript:alert(1)", allowed_schemes=ppt_parser._ALLOWED_LINK_SCHEMES) is None


def test_ppt_hyperlink_range_splits_utf16_and_style_boundaries() -> None:
    """验证非 BMP 字符的 UTF-16 链接范围可跨字符样式边界准确拆分。"""

    interactive_atom = struct.pack("<II8x", 0, 7)
    container_payload = struct.pack("<HHI", 0, ppt_parser.RT_INTERACTIVE_INFO_ATOM, len(interactive_atom)) + interactive_atom
    records = [
        PptRecord(
            offset=0,
            version=0xF,
            instance=0,
            record_type=ppt_parser.RT_INTERACTIVE_INFO,
            payload=container_payload,
        ),
        PptRecord(
            offset=0,
            version=0,
            instance=0,
            record_type=ppt_parser.RT_TEXT_INTERACTIVE_INFO_ATOM,
            payload=struct.pack("<II", 1, 4),
        ),
    ]
    spans = ppt_parser._interactive_spans(
        records,
        {7: "https://example.com"},
        RecordBudget(),
    )
    paragraphs = ppt_parser._build_paragraphs(
        "A😀BC",
        StyleRuns(
            characters=[
                CharacterRun(count=3, bold=True),
                CharacterRun(count=2, italic=True),
            ]
        ),
        [],
        spans,
    )

    assert [(run.text, run.bold, run.italic, run.hyperlink) for run in paragraphs[0].runs] == [
        ("A", True, False, None),
        ("😀", True, False, "https://example.com"),
        ("B", False, True, "https://example.com"),
        ("C", False, True, None),
    ]


@pytest.mark.skipif(not _REAL_PPT.exists(), reason="real Office roundtrip fixture is local-only")
def test_real_ppt_recovers_table_notes_images_and_exports(tmp_path: Path) -> None:
    """验证真实六页 PPT 的合并表格、备注、图片及 sidecar 完整闭包。"""

    middle_json, model_json = doc_analyze(_REAL_PPT.read_bytes(), file_suffix="ppt")

    assert len(model_json.pages) == len(middle_json.pages) == 6
    assert model_json.file_suffix == middle_json.file_suffix == "ppt"
    table = next(block for block in middle_json.pages[0].blocks if isinstance(block, TableBlock))
    table_html = table.content[0].content
    soup = BeautifulSoup(table_html, "html.parser")
    rows = soup.find_all("tr")
    assert len(rows) == 9
    assert max(sum(int(cell.get("colspan", 1)) for cell in row.find_all("td")) for row in rows) == 7
    merged = sorted(
        (int(cell.get("rowspan", 1)), int(cell.get("colspan", 1)), cell.get_text(strip=True))
        for cell in soup.find_all("td")
        if cell.has_attr("rowspan") or cell.has_attr("colspan")
    )
    assert merged == sorted(
        [
            (1, 3, "Class1"),
            (1, 3, "Class2"),
            (1, 2, "A merged with B"),
            (2, 1, "R3"),
            (3, 1, "R4"),
        ]
    )
    assert [inline_text(block.content) for block in middle_json.pages[1].blocks if block.type == BlockType.PAGE_FOOTNOTE] == [
        "Some notes on the second slide."
    ]
    assert [inline_text(block.content) for block in middle_json.pages[2].blocks if block.type == BlockType.PAGE_FOOTNOTE] == [
        "Final notes on the third slide.",
        "Second line of notes.",
    ]
    assert [sum(isinstance(block, ImageBlock) for block in page.blocks) for page in middle_json.pages] == [
        0,
        0,
        0,
        2,
        0,
        0,
    ]
    chart = next(block for block in middle_json.pages[4].blocks if isinstance(block, ChartBlock))
    chart_soup = BeautifulSoup(chart.content[0].content, "html.parser")
    assert [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
        for row in chart_soup.find_all("tr")
    ] == [
        ["", "系列 1", "系列 2", "系列 3"],
        ["类别 1", "4.3", "2.4", "2"],
        ["类别 2", "2.5", "4.4", "2"],
        ["类别 3", "3.5", "1.8", "3"],
        ["类别 4", "4.5", "2.8", "5"],
    ]
    assert chart.content[0].image_base64 is not None

    page_six_lists = [block for block in model_json.pages[5] if block.get("type") == BlockType.LIST]
    assert page_six_lists[0]["attribute"] == "ordered"
    assert page_six_lists[-1]["attribute"] == "ordered"
    assert page_six_lists[-1]["start"] == 3

    export_result = middle_json.export(tmp_path / "export")
    exported_payload = export_result.middle_json.model_dump(mode="json", exclude_none=True)
    assert export_result.json_path.exists()
    assert export_result.image_paths
    assert all(path.exists() and path.stat().st_size > 0 for path in export_result.image_paths)
    assert "image_base64" not in str(exported_payload)
