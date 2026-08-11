from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mineru.backend import analyze
from mineru.backend.utils.span_pre_proc import (
    POST_OCR_FALLBACK_CONTENT_KEY,
    POST_OCR_FALLBACK_SCORE_KEY,
)
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


def _build_post_ocr_page_block_lines(
    *page_crop_values: int | tuple[int, ...],
) -> tuple[list[dict[int, list[Line]]], list[Span]]:
    """按页构造携带待 OCR 裁图的 span，便于验证窗口级批处理和回填顺序。"""
    page_block_lines_list: list[dict[int, list[Line]]] = []
    spans: list[Span] = []
    for page_idx, crop_values in enumerate(page_crop_values):
        normalized_crop_values = (crop_values,) if isinstance(crop_values, int) else crop_values
        page_spans: list[Span] = []
        for span_idx, crop_value in enumerate(normalized_crop_values):
            bbox = (float(span_idx), float(page_idx), float(span_idx + 1), float(page_idx + 1))
            span = Span(
                type=ContentType.TEXT,
                bbox=bbox,
                _np_img=np.full((4, 8, 3), crop_value, dtype=np.uint8),
            )
            spans.append(span)
            page_spans.append(span)
        line_bbox = (0.0, float(page_idx), float(len(page_spans)), float(page_idx + 1))
        page_block_lines_list.append({0: [Line(bbox=line_bbox, spans=page_spans)]})
    return page_block_lines_list, spans


def test_xhigh_vlm_blocks_normalize_visual_annotation_types() -> None:
    """验证 xhigh VLM 的细分视觉标题和脚注会统一成通用类型。"""
    page_blocks = [
        {"type": BlockType.TABLE_CAPTION, "content": "table caption"},
        {"type": BlockType.IMAGE_CAPTION, "content": "image caption"},
        {"type": BlockType.CODE_CAPTION, "content": "code caption"},
        {"type": BlockType.TABLE_FOOTNOTE, "content": "table footnote"},
        {"type": BlockType.IMAGE_FOOTNOTE, "content": "image footnote"},
        {"type": BlockType.TEXT, "content": "text"},
    ]

    analyze._normalize_xhigh_vlm_blocks([page_blocks])

    assert [block["type"] for block in page_blocks] == [
        BlockType.CAPTION,
        BlockType.CAPTION,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
        BlockType.FOOTNOTE,
        BlockType.TEXT,
    ]
    assert [block["content"] for block in page_blocks] == [
        "table caption",
        "image caption",
        "code caption",
        "table footnote",
        "image footnote",
        "text",
    ]


def test_xhigh_vlm_blocks_remove_merge_prev_from_all_text_blocks() -> None:
    """验证 xhigh VLM 正文无论 merge_prev 取值如何都移除该字段。"""
    page_blocks = [
        {"type": BlockType.TEXT, "content": "merge", "merge_prev": True},
        {"type": BlockType.TEXT, "content": "do not merge", "merge_prev": False},
        {"type": BlockType.TEXT, "content": "without hint"},
        {"type": BlockType.CAPTION, "content": "caption"},
    ]

    analyze._normalize_xhigh_vlm_blocks([page_blocks])

    assert all("merge_prev" not in block for block in page_blocks if block["type"] == BlockType.TEXT)
    assert page_blocks[-1] == {"type": BlockType.CAPTION, "content": "caption"}


def test_caption_and_footnote_keep_normalized_line_metadata() -> None:
    """验证通用 caption/footnote 与 text 一样保留归一化 _lines。"""
    assert analyze.LINE_METADATA_BLOCK_TYPES == {
        BlockType.TEXT,
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
    }
    text_block = {"type": BlockType.TEXT, "content": "text"}
    caption_block = {"type": BlockType.CAPTION, "content": "caption"}
    footnote_block = {"type": BlockType.FOOTNOTE, "content": "footnote"}
    unrelated_block = {"type": BlockType.TITLE, "content": "title", "_lines": [{"bbox": [0, 0, 1, 1]}]}
    page_blocks = [text_block, caption_block, footnote_block, unrelated_block]

    analyze._apply_block_content_and_line_metadata(
        page_blocks,
        {
            0: _build_text_lines("text"),
            1: _build_text_lines("caption line 1", "caption line 2"),
            2: _build_text_lines("footnote"),
            3: _build_text_lines("title"),
        },
        (200.0, 100.0),
    )

    assert text_block["_lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]
    assert caption_block["_lines"] == [
        {"bbox": [0.0, 0.0, 0.5, 0.01]},
        {"bbox": [0.0, 0.01, 0.5, 0.02]},
    ]
    assert footnote_block["_lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]
    assert "_lines" not in unrelated_block


def test_window_post_ocr_batches_all_pages_once() -> None:
    """验证窗口内多页待识别 span 合并为一次 OCR-rec，并按原顺序回填。"""
    page_block_lines_list, spans = _build_post_ocr_page_block_lines((11, 12), (21, 22))
    local_model_context = MagicMock()
    local_model_context.ocr_model.ocr.return_value = [
        [("第一页一", 0.91), ("第一页二", 0.82), ("第二页一", 0.73), ("第二页二", 0.64)]
    ]

    analyze._apply_window_post_ocr(local_model_context, page_block_lines_list)

    local_model_context.ocr_model.ocr.assert_called_once()
    call_args = local_model_context.ocr_model.ocr.call_args
    assert [int(crop[0, 0, 0]) for crop in call_args.args[0]] == [11, 12, 21, 22]
    assert call_args.kwargs == {"det": False, "tqdm_enable": True}
    assert [span.content for span in spans] == ["第一页一", "第一页二", "第二页一", "第二页二"]
    assert [span.score for span in spans] == [0.91, 0.82, 0.73, 0.64]
    assert all(span._np_img is None for span in spans)


def test_window_post_ocr_skips_empty_window() -> None:
    """验证窗口内没有待识别裁图时不会调用 OCR-rec。"""
    local_model_context = MagicMock()

    analyze._apply_window_post_ocr(local_model_context, [{}, {0: []}])

    local_model_context.ocr_model.ocr.assert_not_called()


def test_window_post_ocr_rejects_result_count_mismatch() -> None:
    """验证窗口级 OCR-rec 返回数量异常时继续抛出明确错误。"""
    page_block_lines_list, _ = _build_post_ocr_page_block_lines(11, 22)
    local_model_context = MagicMock()
    local_model_context.ocr_model.ocr.return_value = [[("仅一个结果", 0.9)]]

    with pytest.raises(ValueError, match="ocr_res_list=1, need_ocr_spans=2"):
        analyze._apply_window_post_ocr(local_model_context, page_block_lines_list)


def test_window_post_ocr_keeps_low_confidence_fallback_semantics() -> None:
    """验证低置信结果仍恢复原生文本兜底，无兜底文本的 span 继续置空。"""
    page_block_lines_list, spans = _build_post_ocr_page_block_lines(11, 22)
    spans[0].content = "原生文本"
    spans[0].score = 0.88
    spans[0]._extra[POST_OCR_FALLBACK_CONTENT_KEY] = spans[0].content
    spans[0]._extra[POST_OCR_FALLBACK_SCORE_KEY] = spans[0].score
    local_model_context = MagicMock()
    local_model_context.ocr_model.ocr.return_value = [[("低置信文本", 0.0), ("低置信文本", 0.0)]]

    analyze._apply_window_post_ocr(local_model_context, page_block_lines_list)

    assert spans[0].content == "原生文本"
    assert spans[0].score == 0.88
    assert spans[0]._extra == {}
    assert spans[1].content == ""
    assert spans[1].score == 0.0


def test_fill_window_batches_txt_post_ocr_before_page_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 TXT 窗口先统一 OCR-rec，再按页生成最终 block content。"""
    page_block_lines_list, _ = _build_post_ocr_page_block_lines(11, 22)
    grouped_page_lines = iter(page_block_lines_list)
    monkeypatch.setattr(analyze, "_build_page_text_formula_spans", lambda *_args: [])
    monkeypatch.setattr(analyze, "_fill_native_pdf_text_spans", lambda _page, spans, *_args: spans)
    monkeypatch.setattr(analyze, "_group_page_spans_by_block", lambda *_args: next(grouped_page_lines))
    local_model_context = MagicMock()
    local_model_context.ocr_model.ocr.return_value = [[("窗口第一页", 0.9), ("窗口第二页", 0.8)]]
    model_list = [
        [{"type": BlockType.TEXT, "bbox": [0.0, 0.0, 1.0, 1.0], "content": ""}],
        [{"type": BlockType.TEXT, "bbox": [0.0, 0.0, 1.0, 1.0], "content": ""}],
    ]
    pdf_pages = [MagicMock(size=(100.0, 100.0)), MagicMock(size=(100.0, 100.0))]

    result = analyze._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}, {"img_pil": object(), "scale": 1.0}],
        pdf_pages,
        model_list,
        [[], []],
        [[], []],
        "txt",
        {BlockType.TEXT},
        local_model_context,
    )

    assert result is model_list
    local_model_context.ocr_model.ocr.assert_called_once()
    assert [page[0]["content"] for page in model_list] == ["窗口第一页", "窗口第二页"]


def test_fill_window_ocr_mode_does_not_run_txt_post_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 OCR 模式不进入 TXT 的窗口级 post-OCR 分支。"""
    page_block_lines_list, spans = _build_post_ocr_page_block_lines(11)
    spans[0].content = "已有 OCR 文本"
    monkeypatch.setattr(analyze, "_build_page_text_formula_spans", lambda *_args: [])
    monkeypatch.setattr(analyze, "_group_page_spans_by_block", lambda *_args: page_block_lines_list[0])
    local_model_context = MagicMock()
    model_list = [[{"type": BlockType.TEXT, "bbox": [0.0, 0.0, 1.0, 1.0], "content": ""}]]

    analyze._fill_window_block_content_and_lines(
        [{"img_pil": object(), "scale": 1.0}],
        [MagicMock(size=(100.0, 100.0))],
        model_list,
        [[]],
        [[]],
        "ocr",
        {BlockType.TEXT},
        local_model_context,
    )

    local_model_context.ocr_model.ocr.assert_not_called()
    assert model_list[0][0]["content"] == "已有 OCR 文本"


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
