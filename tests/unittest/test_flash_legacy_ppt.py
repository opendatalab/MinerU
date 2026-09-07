from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
import struct

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from docvortex.analyzers.native import PptModel
from docvortex.analyzers.native._shared.hyperlink import sanitize_hyperlink_target
from docvortex.analyzers.native.office.errors import LegacyOfficeEncryptedError
from docvortex.analyzers.native.office.errors import LegacyOfficeResourceLimitError
from docvortex.analyzers.native.office.ppt import parser as ppt_parser
from docvortex.analyzers.native.office.ppt.models import PptPresentation
from docvortex.analyzers.native.office.ppt.models import PptSlide
from docvortex.analyzers.native.office.ppt.ppt_converter import PptConverter
from docvortex.analyzers.native.office.ppt.records import PptRecord
from docvortex.analyzers.native.office.ppt.records import RecordBudget
from docvortex.analyzers.native.office.ppt.style_text import CharacterRun
from docvortex.analyzers.native.office.ppt.style_text import StyleRuns
from mineru.parser import parse
from mineru.types import BlockType, MiddleJson, ModelJson

from _legacy_ppt_test_utils import build_deep_nested_ppt, build_multimaster_ppt, build_sparse_notes_ppt
from _span_test_utils import inline


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
    assert model_json.extensions["mineru"]["effort"] == middle_json.extensions["mineru"]["effort"] == "flash"
    assert model_json.extensions["mineru"]["parse_mode"] == middle_json.extensions["mineru"]["parse_mode"] == "txt"
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
