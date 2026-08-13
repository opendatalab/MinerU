from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from mineru_vl_utils.structs import ContentBlock as VlmContentBlock
from mineru_vl_utils.structs import ExtractResult

from mineru.backend import analyze
from mineru.backend.utils.analyze_draft import _AnalyzeLine, _AnalyzeSpan
from mineru.backend.utils.raw_block_types import RAW_ALGORITHM, RAW_CAPTION, RAW_FOOTNOTE
from mineru.backend.utils.span_pre_proc import (
    POST_OCR_FALLBACK_CONTENT_KEY,
    POST_OCR_FALLBACK_SCORE_KEY,
)
from mineru.model.flash import PdfModel
from mineru.types import BlockType, ContentType, MiddleJson
from mineru.utils.pdf_document import PDFDocument


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NPU_PDF_PATH = _PROJECT_ROOT / "demo" / "pdfs" / "NPU_开发环境部署_参考指南.pdf"


def _build_text_lines(*contents: str) -> list[_AnalyzeLine]:
    """按输入文本构造稳定的多行文本结构，供 block content 拼接测试复用。"""
    return [
        _AnalyzeLine(
            bbox=(0.0, float(line_idx), 100.0, float(line_idx + 1)),
            spans=[
                _AnalyzeSpan(
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
) -> tuple[list[dict[int, list[_AnalyzeLine]]], list[_AnalyzeSpan]]:
    """按页构造携带待 OCR 裁图的 span，便于验证窗口级批处理和回填顺序。"""
    page_block_lines_list: list[dict[int, list[_AnalyzeLine]]] = []
    spans: list[_AnalyzeSpan] = []
    for page_idx, crop_values in enumerate(page_crop_values):
        normalized_crop_values = (crop_values,) if isinstance(crop_values, int) else crop_values
        page_spans: list[_AnalyzeSpan] = []
        for span_idx, crop_value in enumerate(normalized_crop_values):
            bbox = (float(span_idx), float(page_idx), float(span_idx + 1), float(page_idx + 1))
            span = _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=bbox,
                image=np.full((4, 8, 3), crop_value, dtype=np.uint8),
            )
            spans.append(span)
            page_spans.append(span)
        line_bbox = (0.0, float(page_idx), float(len(page_spans)), float(page_idx + 1))
        page_block_lines_list.append({0: [_AnalyzeLine(bbox=line_bbox, spans=page_spans)]})
    return page_block_lines_list, spans


def test_convert_vlm_results_to_model_list_uses_builtin_containers() -> None:
    """验证 VLM 容器子类会转换为原生 list/dict，且不会修改源块。"""
    source_block = VlmContentBlock(
        type=BlockType.TEXT,
        bbox=[0.1, 0.2, 0.8, 0.4],
        angle=0,
        content="原始文本",
        merge_prev=True,
    )
    source_page = ExtractResult([source_block], layout_scored=object())

    model_list = analyze._convert_vlm_results_to_model_list([source_page, ExtractResult()])

    assert type(model_list) is list
    assert all(type(page) is list for page in model_list)
    assert type(model_list[0][0]) is dict
    assert model_list[0][0] == {
        "type": BlockType.TEXT,
        "bbox": [0.1, 0.2, 0.8, 0.4],
        "angle": 0,
        "content": "原始文本",
    }
    assert model_list[1] == []
    assert model_list[0] is not source_page
    assert model_list[0][0] is not source_block

    model_list[0][0]["content"] = "转换后文本"
    assert source_block["content"] == "原始文本"


def test_convert_vlm_results_deep_copies_projected_mutable_fields() -> None:
    """验证 VLM 投影后的可变字段不会与上游对象共享引用。"""
    source_cell_merge = [1, 0]
    source_block = {
        "type": BlockType.TABLE,
        "bbox": [0.1, 0.2, 0.8, 0.4],
        "content": "<table></table>",
        "cell_merge": source_cell_merge,
    }

    model_list = analyze._convert_vlm_results_to_model_list([[source_block]])
    model_list[0][0]["cell_merge"].append(1)

    assert source_cell_merge == [1, 0]


@pytest.mark.parametrize(
    ("effort", "parse_mode", "vlm_method_name"),
    [
        ("high", "txt", "batch_extract_with_layout"),
        ("high", "ocr", "batch_extract_with_layout"),
        ("xhigh", "txt", "batch_two_step_extract"),
        ("xhigh", "ocr", "batch_two_step_extract"),
    ],
)
def test_doc_analyze_converts_vlm_results_before_downstream_processing(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    parse_mode: str,
    vlm_method_name: str,
) -> None:
    """验证 high/xhigh 的 TXT/OCR 路径都只向后续阶段传递原生 list/dict。"""
    source_block = VlmContentBlock(
        type=BlockType.PAGE_NUMBER,
        bbox=[0.45, 0.9, 0.55, 0.95],
        angle=0,
        content="1",
    )
    source_page = ExtractResult([source_block], layout_scored=object())
    fake_document = MagicMock()
    fake_document.page_count = 1
    fake_document.__getitem__.return_value = MagicMock(size=(612.0, 792.0))

    hybrid_model = MagicMock()
    hybrid_model.device = "cpu"
    hybrid_model.layout_model.batch_predict.return_value = [[]]
    hybrid_singleton = MagicMock()
    hybrid_singleton.get_model.return_value = hybrid_model

    vlm_predictor = MagicMock()
    getattr(vlm_predictor, vlm_method_name).return_value = [source_page]
    vlm_singleton = MagicMock()
    vlm_singleton.get_model.return_value = vlm_predictor
    serial_wrapper = MagicMock(return_value=vlm_predictor)
    page_image = Image.new("RGB", (64, 96), "white")

    def fake_process_text_and_formulas(
        _images_list: object,
        _pdf_pages: object,
        window_model_list: list[list[dict[str, object]]],
        _parse_mode: object,
        _effort: object,
        _hybrid_model: object,
        _images_layout_res: object,
    ) -> list[list[dict[str, object]]]:
        """原样返回窗口结果，并在后处理入口校验精确容器类型。"""
        assert type(window_model_list) is list
        assert all(type(page) is list for page in window_model_list)
        assert all(type(block) is dict for page in window_model_list for block in page)
        return window_model_list

    xhigh_normalizer = MagicMock(wraps=analyze._normalize_xhigh_vlm_blocks)
    monkeypatch.setattr(analyze, "PDFDocument", MagicMock(return_value=fake_document))
    monkeypatch.setattr(analyze, "HybridLocalModelContextSingleton", MagicMock(return_value=hybrid_singleton))
    monkeypatch.setattr(
        analyze,
        "_load_vlm_runtime",
        MagicMock(
            return_value={
                "ModelSingleton": MagicMock(return_value=vlm_singleton),
                "_maybe_enable_serial_execution": serial_wrapper,
            }
        ),
    )
    monkeypatch.setattr(analyze, "get_vlm_engine", MagicMock(return_value="transformers"))
    monkeypatch.setattr(
        analyze,
        "load_images_from_pdf_bytes_range",
        MagicMock(return_value=[{"img_pil": page_image, "scale": 1.0}]),
    )
    monkeypatch.setattr(analyze, "_process_text_and_formulas", fake_process_text_and_formulas)
    monkeypatch.setattr(analyze, "_normalize_xhigh_vlm_blocks", xhigh_normalizer)
    monkeypatch.setattr(analyze, "_apply_seal_ocr", MagicMock())
    monkeypatch.setattr(analyze, "_supplement_missing_image_block_containers", MagicMock())
    monkeypatch.setattr(analyze, "_attach_visual_block_images", MagicMock())
    monkeypatch.setattr(analyze, "model_list_to_pages", MagicMock(return_value=[]))
    monkeypatch.setattr(analyze, "clean_memory", MagicMock())

    middle_json, model_list = analyze.doc_analyze(
        b"fake-pdf",
        effort=effort,  # type: ignore[arg-type]
        parse_mode=parse_mode,  # type: ignore[arg-type]
    )

    assert type(model_list) is list
    assert type(model_list[0]) is list
    assert type(model_list[0][0]) is dict
    assert model_list == [[{"type": BlockType.PAGE_NUMBER, "bbox": [0.45, 0.9, 0.55, 0.95], "content": "1"}]]
    assert isinstance(middle_json, MiddleJson)
    assert middle_json.pages == []
    assert middle_json.effort == effort
    assert middle_json.parse_mode == parse_mode
    assert type(source_page) is ExtractResult
    assert type(source_block) is VlmContentBlock
    assert source_block["angle"] == 0

    if effort == "xhigh":
        xhigh_normalizer.assert_called_once()
        normalized_model_list = xhigh_normalizer.call_args.args[0]
        assert all(type(page) is list for page in normalized_model_list)
        assert all(type(block) is dict for page in normalized_model_list for block in page)
        vlm_predictor.batch_two_step_extract.assert_called_once()
        vlm_predictor.batch_extract_with_layout.assert_not_called()
    else:
        xhigh_normalizer.assert_not_called()
        vlm_predictor.batch_extract_with_layout.assert_called_once()
        vlm_predictor.batch_two_step_extract.assert_not_called()


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
        RAW_CAPTION,
        RAW_CAPTION,
        RAW_CAPTION,
        RAW_FOOTNOTE,
        RAW_FOOTNOTE,
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


def test_five_pdf_text_types_keep_normalized_line_metadata() -> None:
    """验证五类 PDF 文本块统一保留公开的归一化 lines。"""
    assert analyze.LINE_METADATA_BLOCK_TYPES == {
        BlockType.TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    }
    text_block = {"type": BlockType.TEXT, "content": "text"}
    doc_title_block = {"type": BlockType.DOC_TITLE, "content": "doc title"}
    paragraph_title_block = {
        "type": BlockType.PARAGRAPH_TITLE,
        "content": "paragraph title",
    }
    caption_block = {"type": RAW_CAPTION, "content": "caption"}
    footnote_block = {"type": RAW_FOOTNOTE, "content": "footnote"}
    unrelated_block = {
        "type": BlockType.EQUATION,
        "content": "x=1",
        "lines": [{"bbox": [0, 0, 1, 1]}],
    }
    page_blocks = [
        text_block,
        doc_title_block,
        paragraph_title_block,
        caption_block,
        footnote_block,
        unrelated_block,
    ]

    analyze._apply_block_content_and_line_metadata(
        page_blocks,
        {
            0: _build_text_lines("text"),
            1: _build_text_lines("doc title"),
            2: _build_text_lines("paragraph title"),
            3: _build_text_lines("caption line 1", "caption line 2"),
            4: _build_text_lines("footnote"),
            5: _build_text_lines("x=1"),
        },
        (200.0, 100.0),
    )

    assert text_block["lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]
    assert doc_title_block["lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]
    assert paragraph_title_block["lines"] == [
        {"bbox": [0.0, 0.0, 0.5, 0.01]}
    ]
    assert caption_block["lines"] == [
        {"bbox": [0.0, 0.0, 0.5, 0.01]},
        {"bbox": [0.0, 0.01, 0.5, 0.02]},
    ]
    assert footnote_block["lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]
    assert "lines" not in unrelated_block


@pytest.mark.parametrize("effort", ["high", "xhigh"])
def test_high_ocr_detects_lines_for_all_five_pdf_text_types(
    effort: str,
) -> None:
    """验证 High/XHigh OCR 对标题、正文、caption 和 footnote 都执行行检测。"""

    ocr_det_type, mfr_enabled = analyze._build_ocr_det_type_and_mfr_enable(
        "ocr",
        effort,
    )

    assert ocr_det_type == analyze.LINE_METADATA_BLOCK_TYPES
    assert mfr_enabled is False


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
    assert all(span.image is None for span in spans)


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
    spans[0].metadata[POST_OCR_FALLBACK_CONTENT_KEY] = spans[0].content
    spans[0].metadata[POST_OCR_FALLBACK_SCORE_KEY] = spans[0].score
    local_model_context = MagicMock()
    local_model_context.ocr_model.ocr.return_value = [[("低置信文本", 0.0), ("低置信文本", 0.0)]]

    analyze._apply_window_post_ocr(local_model_context, page_block_lines_list)

    assert spans[0].content == "原生文本"
    assert spans[0].score == 0.88
    assert spans[0].metadata == {}
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


@pytest.mark.parametrize("block_type", [BlockType.CODE, RAW_ALGORITHM])
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
    assert "lines" not in block


def test_low_txt_visual_runs_split_distant_text_before_layout_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Low/TXT 在版面匹配前按 Flash 字符间隙拆开同一粗行。"""

    chars = [
        {"char": "L", "bbox": (5.0, 10.0, 9.0, 20.0)},
        {"char": "e", "bbox": (9.0, 10.0, 13.0, 20.0)},
        {"char": "f", "bbox": (13.0, 10.0, 17.0, 20.0)},
        {"char": "t", "bbox": (17.0, 10.0, 21.0, 20.0)},
        {"char": " ", "bbox": (21.0, 10.0, 25.0, 20.0)},
        {"char": "R", "bbox": (70.0, 10.0, 74.0, 20.0)},
        {"char": "i", "bbox": (74.0, 10.0, 78.0, 20.0)},
        {"char": "g", "bbox": (78.0, 10.0, 82.0, 20.0)},
        {"char": "h", "bbox": (82.0, 10.0, 86.0, 20.0)},
        {"char": "t", "bbox": (86.0, 10.0, 90.0, 20.0)},
    ]
    pdf_lines = [
        {
            "bbox": (5.0, 10.0, 90.0, 20.0),
            "rotation": 0.0,
            "spans": [
                {
                    "text": "Left Right",
                    "bbox": (5.0, 10.0, 90.0, 20.0),
                    "rotation": 0.0,
                    "chars": chars,
                }
            ],
        }
    ]
    pdf_page = MagicMock()
    pdf_page.size = (100.0, 100.0)
    pdf_page.rotation = 0
    pdf_page.get_chars.return_value = []
    monkeypatch.setattr(analyze, "get_lines_from_chars", lambda _chars: pdf_lines)

    spans = analyze._build_pdf_text_visual_run_spans(pdf_page)

    assert [(span.content, span.bbox) for span in spans] == [
        ("Left", (5.0, 10.0, 21.0, 20.0)),
        ("Right", (70.0, 10.0, 90.0, 20.0)),
    ]


def test_existing_text_content_is_preserved_while_lines_are_refreshed() -> None:
    """验证 Low/TXT 只补空正文，已有 VLM 内容不覆盖但仍写入真实行框。"""

    block = {"type": BlockType.TEXT, "content": "existing content"}

    analyze._apply_block_content_and_line_metadata(
        [block],
        {0: _build_text_lines("native content")},
        (200.0, 100.0),
    )

    assert block["content"] == "existing content"
    assert block["lines"] == [{"bbox": [0.0, 0.0, 0.5, 0.01]}]


def test_npu_page_three_low_visual_runs_keep_index_line_structure() -> None:
    """验证 Low/TXT 的 Flash visual run 拆分不会破坏 NPU 的 24 行目录。"""
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
            analyze._build_pdf_text_visual_run_spans(page),
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
