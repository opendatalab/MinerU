# Copyright (c) Opendatalab. All rights reserved.
"""覆盖 Issue #5565：行级几何缺失不能删除已有正文和块框的文本。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest

from mineru.backend.analysis.pdf import normalization
from mineru.types import RAW_CAPTION, RAW_FOOTNOTE, RAW_PHONETIC, BlockType


@pytest.mark.parametrize(
    "block_type",
    [BlockType.TEXT, BlockType.REF_TEXT, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE, RAW_CAPTION, RAW_FOOTNOTE],
)
@pytest.mark.parametrize(
    "invalid_lines",
    [
        None,
        [],
        "invalid",
        {"bbox": [0.1, 0.2, 0.8, 0.4]},
        [None],
        [{}],
        [{"bbox": [0.1, 0.2, 0.8]}],
        [{"bbox": [True, 0.2, 0.8, 0.4]}],
        [{"bbox": [0.1, 0.2, float("nan"), 0.4]}],
        [{"bbox": [0.1, 0.2, float("inf"), 0.4]}],
        [{"bbox": [-0.1, 0.2, 0.8, 0.4]}],
        [{"bbox": [0.1, 0.2, 1.1, 0.4]}],
        [{"bbox": [0.8, 0.2, 0.8, 0.4]}],
        [{"bbox": [0.1, 0.4, 0.8, 0.2]}],
        [{"bbox": [0.1, 0.2, 0.8, 0.3]}, {"bbox": [0.1, 0.4, 0.1, 0.5]}],
    ],
)
def test_normalization_repairs_invalid_lines_from_block_bbox(
    block_type: str, invalid_lines: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证六类文本的缺失或非法行框均可由合法块框兜底，且不覆盖内容。"""
    warning = MagicMock()
    monkeypatch.setattr(normalization.logger, "warning", warning)
    bbox = [0.1, 0.2, 0.8, 0.4]
    block = {"type": block_type, "bbox": bbox, "content": "已识别的正文", "lines": invalid_lines}
    page = [block]
    pages = [page]

    normalization._normalize_pdf_model_list(pages)

    assert pages[0] is page
    assert page == [block]
    assert page[0] is block
    assert block["content"] == [{"type": "text", "content": "已识别的正文"}]
    assert block["lines"] == [{"bbox": bbox}]
    assert block["lines"][0]["bbox"] is not bbox
    warning.assert_called_once_with("PDF text block normalization: page_idx={}, bbox_fallback={}, dropped={}", 0, 1, 0)


@pytest.mark.parametrize("bbox", [[0, 0, 1, 1], (0.1, 0.2, 0.8, 0.4)])
def test_normalization_repairs_missing_lines_and_phonetic_type(bbox: object) -> None:
    """验证未提供 lines 的文本与转换后的音标块均能使用边界合法的块框。"""
    block = {"type": RAW_PHONETIC, "bbox": bbox, "content": "phonetic"}
    pages = [[block]]

    normalization._normalize_pdf_model_list(pages)

    assert pages == [[block]]
    assert block["type"] == BlockType.TEXT
    assert block["lines"] == [{"bbox": list(bbox)}]


@pytest.mark.parametrize(
    "invalid_bbox",
    [
        None,
        [],
        "0 0 1 1",
        [0, 0, 1],
        [0, 0, 1, 1, 1],
        [False, 0, 1, 1],
        [0, 0, "1", 1],
        [0, 0, float("nan"), 1],
        [0, 0, float("inf"), 1],
        [0, float("-inf"), 1, 1],
        [0, 0, 10**400, 1],
        [-0.1, 0, 1, 1],
        [0, 0, 1.1, 1],
        [0.5, 0, 0.5, 1],
        [0, 0.5, 1, 0.5],
        [0.8, 0, 0.2, 1],
        [0, 0.8, 1, 0.2],
    ],
)
def test_normalization_still_discards_text_without_usable_geometry(
    invalid_bbox: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证没有合法行框或块框时仍删除文本，并按页报告丢弃数量。"""
    warning = MagicMock()
    monkeypatch.setattr(normalization.logger, "warning", warning)
    pages = [[{"type": BlockType.TEXT, "content": "正文", "bbox": invalid_bbox, "lines": []}]]

    normalization._normalize_pdf_model_list(pages)

    assert pages == [[]]
    warning.assert_called_once_with("PDF text block normalization: page_idx={}, bbox_fallback={}, dropped={}", 0, 0, 1)


@pytest.mark.parametrize(
    "content",
    [None, "", " \n ", [], ["invalid"], [{"type": "text", "content": "  "}], [{"type": "equation_inline", "content": ""}]],
)
def test_normalization_does_not_rescue_empty_content(content: object) -> None:
    """验证合法块框不能保留空正文或无可见内容的 Span。"""
    pages = [[{"type": BlockType.TEXT, "bbox": [0, 0, 1, 1], "content": content, "lines": []}]]

    normalization._normalize_pdf_model_list(pages)

    assert pages == [[]]


@pytest.mark.parametrize("bbox", [None, [0.8, 0.2, 0.1, 0.4], [0.1, 0.2, 0.8, 0.4]])
def test_normalization_keeps_valid_lines_unchanged(bbox: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证已有完整合法行框时不重建行框，也不额外改变原有块框校验行为。"""
    warning = MagicMock()
    monkeypatch.setattr(normalization.logger, "warning", warning)
    lines = [{"bbox": [0.1, 0.2, 0.8, 0.3]}, {"bbox": (0.1, 0.3, 0.8, 0.4)}]
    block = {"type": BlockType.TEXT, "content": "正文", "bbox": bbox, "lines": lines}
    pages = [[block]]

    normalization._normalize_pdf_model_list(pages)

    assert pages == [[block]]
    assert block["lines"] is lines
    warning.assert_not_called()


def test_normalization_preserves_equation_dense_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """复现十六个编号公式块仅两个有行框的页面，确保顺序、正文与对象均不丢失。"""
    warning = MagicMock()
    monkeypatch.setattr(normalization.logger, "warning", warning)
    texts: list[dict[str, Any]] = []
    for index, number in enumerate(range(62, 78)):
        bbox = [0.1, 0.05 + index * 0.05, 0.8, 0.09 + index * 0.05]
        texts.append(
            {
                "type": BlockType.TEXT,
                "bbox": bbox,
                "content": f"{number}. \\(x^2 + {number} = 0\\)",
                "lines": [{"bbox": list(bbox)}] if number in {63, 77} else [],
            }
        )
    header = {"type": BlockType.HEADER, "content": "习题"}
    container = {"type": BlockType.LIST, "bbox": [0.1, 0.05, 0.8, 0.14]}
    page_number = {"type": BlockType.PAGE_NUMBER, "content": "11"}
    page = [header, container, *texts, page_number]
    pages = [page]

    normalization._normalize_pdf_model_list(pages)

    assert pages[0] is page
    assert len(page) == 19
    assert page == [header, container, *texts, page_number]
    for number, block in zip(range(62, 78), texts, strict=True):
        assert block["content"] == [
            {"type": "text", "content": f"{number}. "},
            {"type": "equation_inline", "content": f"x^2 + {number} = 0"},
        ]
        assert block["lines"] == [{"bbox": block["bbox"]}]
    warning.assert_called_once_with("PDF text block normalization: page_idx={}, bbox_fallback={}, dropped={}", 0, 14, 0)

    snapshot = deepcopy(pages)
    warning.reset_mock()
    normalization._normalize_pdf_model_list(pages)
    assert pages == snapshot
    warning.assert_not_called()


def test_normalization_preserves_formula_spans_and_summarizes_each_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证纯公式 Span 不是空正文，非文本块不参与兜底，异常按所在页汇总。"""
    warning = MagicMock()
    monkeypatch.setattr(normalization.logger, "warning", warning)
    content = [{"type": "equation_inline", "content": "x^2"}]
    recovered = {"type": BlockType.TEXT, "content": content, "bbox": [0, 0, 1, 1]}
    equation = {"type": BlockType.EQUATION, "content": "", "lines": []}
    pages = [[equation], [recovered, {"type": BlockType.TEXT, "content": "", "bbox": [0, 0, 1, 1]}]]

    normalization._normalize_pdf_model_list(pages)

    assert pages == [[equation], [recovered]]
    assert recovered["content"] == content
    assert "lines" not in pages[0][0] or pages[0][0]["lines"] == []
    warning.assert_called_once_with("PDF text block normalization: page_idx={}, bbox_fallback={}, dropped={}", 1, 1, 1)
