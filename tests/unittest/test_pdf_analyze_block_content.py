from __future__ import annotations

from pathlib import Path

import pytest

from mineru.backend import analyze
from mineru.model.flash import PdfModel
from mineru.types import BlockType, ContentType, Line, Span
from mineru.utils.pdf_document import PDFDocument


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NPU_PDF_PATH = _PROJECT_ROOT / "demo" / "pdfs" / "NPU_开发环境部署_参考指南.pdf"


def _build_text_lines(*contents: str) -> list[Line]:
    """按输入文本构造稳定的多行文本结构，供 block content 拼接测试复用。"""
    return [
        Line(
            bbox=(0.0, float(line_idx), 100.0, float(line_idx + 1)),
            spans=[
                Span(
                    type=ContentType.TEXT,
                    bbox=(0.0, float(line_idx), 100.0, float(line_idx + 1)),
                    content=content,
                )
            ],
        )
        for line_idx, content in enumerate(contents)
    ]


def test_empty_index_content_preserves_grouped_line_breaks() -> None:
    """验证空目录块由组行结果回填时保留每个目录项的物理换行。"""
    block = {
        "type": BlockType.INDEX,
        "bbox": [0.1, 0.2, 0.9, 0.8],
        "angle": 0,
        "content": "",
    }

    analyze._apply_block_content_and_line_metadata(
        [block],
        {0: _build_text_lines("第一行", "第二行")},
        (100.0, 100.0),
    )

    assert block == {
        "type": BlockType.INDEX,
        "bbox": [0.1, 0.2, 0.9, 0.8],
        "angle": 0,
        "content": "第一行\n第二行",
    }


def test_text_content_keeps_natural_paragraph_joining() -> None:
    """验证普通中文正文仍沿用自然段跨行连接规则，不受目录修复影响。"""
    content = analyze._lines_to_block_content(
        _build_text_lines("第一行", "第二行"),
        BlockType.TEXT,
    )

    assert content == "第一行第二行"


@pytest.mark.parametrize("block_type", [BlockType.CODE, BlockType.ALGORITHM])
def test_code_content_keeps_existing_line_breaks(block_type: str) -> None:
    """验证 code/algorithm 原有的逐行拼接语义保持不变。"""
    content = analyze._lines_to_block_content(
        _build_text_lines("line one", "line two"),
        block_type,
    )

    assert content == "line one\nline two"


def test_existing_index_content_is_not_overwritten() -> None:
    """验证 VLM 等上游已经提供的目录 content 不会被 PDF/OCR 组行结果覆盖。"""
    block = {
        "type": BlockType.INDEX,
        "bbox": [0.1, 0.2, 0.9, 0.8],
        "angle": 0,
        "content": "已有目录一\n已有目录二",
    }

    analyze._apply_block_content_and_line_metadata(
        [block],
        {0: _build_text_lines("回填目录一", "回填目录二")},
        (100.0, 100.0),
    )

    assert block["content"] == "已有目录一\n已有目录二"
    assert "_lines" not in block


def test_npu_page_three_medium_index_matches_flash_line_structure() -> None:
    """验证真实 NPU 目录页的 Medium 回填结果与 Flash 一样保留 24 个目录行。"""
    document = PDFDocument(_NPU_PDF_PATH.read_bytes())
    try:
        page = document[2]
        page_size = tuple(float(value) for value in page.size)
        index_block = {
            "type": BlockType.INDEX,
            "bbox": [0.064, 0.17, 0.943, 0.689],
            "angle": 0,
            "content": "",
        }
        block_lines = analyze._group_page_spans_by_block(
            [index_block],
            analyze._build_pdf_text_line_spans(page),
            page_size,
            {BlockType.INDEX},
        )

        assert len(block_lines[0]) == 24
        analyze._apply_block_content_and_line_metadata(
            [index_block],
            block_lines,
            page_size,
        )

        flash_index_block = next(block for block in PdfModel().predict(document)[2] if block.get("type") == BlockType.INDEX)
        assert index_block["content"] == flash_index_block["content"]
        assert len(str(index_block["content"]).splitlines()) == 24
        assert str(index_block["content"]).splitlines()[0] == "1 前言 1"
        assert str(index_block["content"]).splitlines()[-1] == "5 附录： 19"
        assert index_block["type"] == BlockType.INDEX
        assert index_block["bbox"] == [0.064, 0.17, 0.943, 0.689]
        assert index_block["angle"] == 0
    finally:
        document.close()
