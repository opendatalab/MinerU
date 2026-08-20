# Copyright (c) Opendatalab. All rights reserved.
"""文本/公式 Span 构造、组行、post-OCR 与 block content 回填。"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image
from pdftext.schema import Char

from mineru.backend.local_model_runtime import HybridLocalModelContext, run_ocr_inference
from mineru.types import BBox, BlockType, ContentType
from mineru.utils.language import detect_lang
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.utils.pdf_document import PDFPage, get_lines_from_chars
from mineru.utils.pdf_text_styles import (
    PDFTextStyleLine,
    apply_pdf_strikethrough_styles,
)
from mineru.utils.text_utils import resolve_text_line_boundary

from ..constants import (
    CODE_CONTENT_BLOCK_TYPES,
    LINE_METADATA_BLOCK_TYPES,
    TITLE_BLOCK_TYPES,
)
from ..geometry import _bbox_to_pixel_bbox, _sidecar_bbox_to_page_bbox
from .lines import group_spans_to_lines
from .models import _AnalyzeLine, _AnalyzeSpan
from .native import (
    MAX_NATIVE_TEXT_CHARS_PER_PAGE,
    SpanBlockMatcher,
    __replace_ligatures,
    __replace_unicode,
    _clear_post_ocr_fallback,
    _is_supported_rotation,
    _restore_post_ocr_fallback,
    txt_spans_extract,
)
from .styles import build_pdf_native_visual_lines_and_styles


def _validate_text_formula_window_inputs(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    images_layout_res: list[list[dict[str, Any]]],
) -> None:
    """校验文本公式处理所需的窗口分页数据，避免 zip 静默截断。"""
    page_counts = {
        "images": len(images_list),
        "pdf_pages": len(pdf_pages),
        "model_list": len(model_list),
        "layout": len(images_layout_res),
    }
    if len(set(page_counts.values())) != 1:
        raise ValueError(f"Hybrid text/formula window page count mismatch: {page_counts}")

    for page_idx, image_dict in enumerate(images_list):
        if image_dict.get("img_pil") is None:
            raise ValueError(f"Hybrid text/formula window image is missing img_pil: page_idx={page_idx}")
        scale = float(image_dict.get("scale", 0) or 0)
        if scale <= 0:
            raise ValueError(f"Hybrid text/formula window image scale must be positive: page_idx={page_idx}")


def _build_pdf_text_line_spans(pdf_page: PDFPage) -> list[_AnalyzeSpan]:
    """将标准方向的 pdftext line 转为私有 span，并复用现有字符清洗规则。"""
    page_spans: list[_AnalyzeSpan] = []
    for pdf_line in get_lines_from_chars(pdf_page.get_chars()):
        try:
            rotation = float(pdf_line.get("rotation", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not _is_supported_rotation(rotation):
            continue

        raw_bbox = pdf_line.get("bbox")
        bbox = getattr(raw_bbox, "bbox", raw_bbox)
        try:
            if bbox is None or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        content = "".join(str(pdf_span.get("text", "") or "") for pdf_span in pdf_line.get("spans", []))
        content = __replace_unicode(content)
        content = __replace_ligatures(content).strip()
        if not content:
            continue

        page_spans.append(
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=(x0, y0, x1, y1),
                content=content,
                score=1.0,
            )
        )
    return page_spans


def _build_page_text_formula_spans(
    page_inline_formula_list: list[dict[str, Any]],
    page_ocr_res_list: list[dict[str, Any]],
    page_size: tuple[float, float],
    render_scale: float,
) -> list[_AnalyzeSpan]:
    """将当前页行内公式和 OCR 结果转换为私有 span，供正文与公式共同组行。"""
    page_spans: list[_AnalyzeSpan] = []
    for formula in page_inline_formula_list:
        bbox = _sidecar_bbox_to_page_bbox(formula.get("bbox"), page_size, render_scale)
        if bbox is None:
            continue
        page_spans.append(
            _AnalyzeSpan(
                type=ContentType.INLINE_EQUATION,
                bbox=bbox,
                content=str(formula.get("latex", "") or "").strip(),
                score=float(formula.get("score", 0.0) or 0.0),
            )
        )

    for ocr_res in page_ocr_res_list:
        bbox = _sidecar_bbox_to_page_bbox(ocr_res.get("bbox"), page_size, render_scale)
        if bbox is None:
            continue
        page_spans.append(
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=bbox,
                content=str(ocr_res.get("text", "") or ""),
                score=float(ocr_res.get("score", 0.0) or 0.0),
            )
        )
    return page_spans


def _fill_native_pdf_text_spans(
    pdf_page: PDFPage,
    page_spans: list[_AnalyzeSpan],
    page_pil_image: Image.Image,
    render_scale: float,
    page_size: tuple[float, float],
    *,
    page_chars: list[Char] | None = None,
) -> list[_AnalyzeSpan]:
    """复用原生 PDF 字符回填逻辑，并允许共享同页删除线检测读取的字符。"""
    page_width, page_height = page_size
    virtual_block = (0, 0, page_width, page_height, None, None, None, BlockType.TEXT)
    return txt_spans_extract(
        pdf_page,
        page_spans,
        page_pil_image,
        render_scale,
        [virtual_block],
        [],
        page_chars=page_chars,
    )


def _group_page_spans_by_block(
    page_model_list: list[dict[str, Any]],
    page_spans: list[_AnalyzeSpan],
    page_size: tuple[float, float],
    target_block_types: set[str],
) -> dict[int, list[_AnalyzeLine]]:
    """按 block 原始顺序消费 span，并使用现有文本修复逻辑形成真实行。"""
    span_matcher = SpanBlockMatcher(page_spans)
    block_lines: dict[int, list[_AnalyzeLine]] = {}
    for block_idx, block_item in enumerate(page_model_list):
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        if block_type not in target_block_types:
            continue
        block_bbox = _bbox_to_pixel_bbox(block_item.get("bbox"), page_size)
        if block_bbox is None:
            block_lines[block_idx] = []
            continue

        block_lines[block_idx] = group_spans_to_lines(span_matcher.collect_for_block(block_bbox))
    return block_lines


def _apply_window_post_ocr(
    local_model_context: HybridLocalModelContext,
    page_block_lines_list: list[dict[int, list[_AnalyzeLine]]],
) -> None:
    """在当前窗口内识别原生字符不足的 span，保持 finalize 后置 OCR 的回退语义。"""
    need_ocr_spans: list[_AnalyzeSpan] = []
    img_crop_list: list[np.ndarray] = []
    for block_lines in page_block_lines_list:
        for lines in block_lines.values():
            for line in lines:
                for span in line.spans:
                    if span.image is None:
                        continue
                    need_ocr_spans.append(span)
                    img_crop_list.append(rotate_vertical_crop_if_needed(span.image))
                    span.image = None

    if not img_crop_list:
        return
    ocr_res_list = run_ocr_inference(
        local_model_context.ocr_model.ocr,
        img_crop_list,
        det=False,
        tqdm_enable=True,
    )[0]
    if len(ocr_res_list) != len(need_ocr_spans):
        raise ValueError(
            f"Hybrid post-OCR result count mismatch: ocr_res_list={len(ocr_res_list)}, need_ocr_spans={len(need_ocr_spans)}"
        )

    for span, ocr_res in zip(need_ocr_spans, ocr_res_list):
        ocr_text, ocr_score = ocr_res
        if ocr_score > OcrConfidence.min_confidence:
            span.content = ocr_text
            span.score = float(f"{ocr_score:.3f}")
            _clear_post_ocr_fallback(span)
        elif _restore_post_ocr_fallback(span):
            continue
        else:
            span.content = ""
            span.score = 0.0


def _line_content_parts(line: _AnalyzeLine) -> list[tuple[str, str]]:
    """提取一行内可输出的文本与行内公式，公式统一包装为反斜杠圆括号格式。"""
    parts: list[tuple[str, str]] = []
    for span in line.spans:
        if span.type == ContentType.TEXT:
            content = str(span.content or "").strip()
        elif span.type == ContentType.INLINE_EQUATION:
            latex = str(span.content or "").strip()
            content = f"\\({latex}\\)" if latex else ""
        else:
            continue
        if content:
            parts.append((span.type, content))
    return parts


def _lines_to_block_content(lines: list[_AnalyzeLine], block_type: str) -> str:
    """将真实行折叠为统一 block content，保留目录/代码换行并处理自然语言跨行连接。"""
    content_lines = [parts for line in lines if (parts := _line_content_parts(line))]
    if not content_lines:
        return ""

    rendered_lines = [" ".join(content for _, content in parts) for parts in content_lines]
    if block_type == BlockType.INDEX or block_type in CODE_CONTENT_BLOCK_TYPES:
        return "\n".join(rendered_lines).strip()

    text_for_language = "".join(
        content for parts in content_lines for span_type, content in parts if span_type == ContentType.TEXT
    )
    block_language = detect_lang(text_for_language)
    content_parts = [rendered_lines[0]]
    for line_idx in range(1, len(rendered_lines)):
        content_parts[-1], separator = resolve_text_line_boundary(
            content_parts[-1],
            block_language=block_language,
            next_content=rendered_lines[line_idx],
        )
        content_parts.extend([separator, rendered_lines[line_idx]])
    return "".join(content_parts).strip()


def _build_ocr_det_line_items(lines: list[_AnalyzeLine], page_size: tuple[float, float]) -> list[dict[str, Any]]:
    """将 Analyze 私有行转换为归一化行框。"""
    line_items = []
    for line in lines:
        normalized_bbox = _page_bbox_to_unit_bbox(line.bbox, page_size)
        if normalized_bbox is not None:
            line_items.append({"bbox": normalized_bbox})
    return line_items


def _apply_block_content_and_line_metadata(
    page_model_list: list[dict[str, Any]],
    block_lines: dict[int, list[_AnalyzeLine]],
    page_size: tuple[float, float],
) -> None:
    """将组行结果回填到 block，并为正文、视觉标题和视觉脚注保存行框。"""
    for block_item in page_model_list:
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        if block_type in LINE_METADATA_BLOCK_TYPES:
            block_item["lines"] = []
        else:
            block_item.pop("lines", None)

    for block_idx, lines in block_lines.items():
        block_item = page_model_list[block_idx]
        block_type = str(block_item.get("type") or block_item.get("label") or "")
        block_content = block_item.get("content")
        has_nonempty_content = bool(block_content.strip()) if isinstance(block_content, str) else bool(block_content)
        if not has_nonempty_content:
            block_item["content"] = _lines_to_block_content(lines, block_type)

        if block_type in LINE_METADATA_BLOCK_TYPES:
            block_item["lines"] = _build_ocr_det_line_items(lines, page_size)


def _fill_window_block_content_and_lines(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    ocr_res_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    ocr_det_type: set[str],
    local_model_context: HybridLocalModelContext,
) -> list[list[dict[str, Any]]]:
    """按页完成 span 回填与行级元数据构造，返回不含页面级 sidecar 的 model list。"""
    page_counts = {
        "images": len(images_list),
        "pdf_pages": len(pdf_pages),
        "model_list": len(model_list),
        "inline_formulas": len(inline_formula_list),
        "ocr_results": len(ocr_res_list),
    }
    if len(set(page_counts.values())) != 1:
        raise ValueError(f"Hybrid block content page count mismatch: {page_counts}")

    target_block_types = set(ocr_det_type) | TITLE_BLOCK_TYPES | {BlockType.TEXT}
    page_block_line_results: list[
        tuple[
            list[dict[str, Any]],
            dict[int, list[_AnalyzeLine]],
            tuple[float, float],
            list[PDFTextStyleLine],
        ]
    ] = []
    for image_dict, pdf_page, page_model_list, page_inline_formula_list, page_ocr_res_list in zip(
        images_list,
        pdf_pages,
        model_list,
        inline_formula_list,
        ocr_res_list,
    ):
        page_pil_image = image_dict["img_pil"]
        render_scale = float(image_dict["scale"])
        page_size = tuple(float(value) for value in pdf_page.size)
        page_spans = _build_page_text_formula_spans(
            page_inline_formula_list,
            page_ocr_res_list,
            page_size,
            render_scale,
        )
        style_lines: list[PDFTextStyleLine] = []
        if parse_mode == "txt":
            page_chars = None
            try:
                page_char_count = pdf_page.get_char_count()
            except Exception:
                page_char_count = None
            if (
                not isinstance(page_char_count, int)
                or isinstance(page_char_count, bool)
                or page_char_count <= MAX_NATIVE_TEXT_CHARS_PER_PAGE
            ):
                page_chars, _line_items, style_lines = build_pdf_native_visual_lines_and_styles(
                    pdf_page,
                )
            page_spans = _fill_native_pdf_text_spans(
                pdf_page,
                page_spans,
                page_pil_image,
                render_scale,
                page_size,
                page_chars=page_chars,
            )

        block_lines = _group_page_spans_by_block(
            page_model_list,
            page_spans,
            page_size,
            target_block_types,
        )
        page_block_line_results.append(
            (page_model_list, block_lines, page_size, style_lines)
        )

    if parse_mode == "txt":
        _apply_window_post_ocr(
            local_model_context,
            [
                block_lines
                for _, block_lines, _, _style_lines in page_block_line_results
            ],
        )

    for page_model_list, block_lines, page_size, style_lines in page_block_line_results:
        _apply_block_content_and_line_metadata(
            page_model_list,
            block_lines,
            page_size,
        )
        apply_pdf_strikethrough_styles(
            page_model_list,
            style_lines,
            page_size,
        )
    return model_list


def _page_bbox_to_unit_bbox(bbox: BBox, page_size: tuple[float, float]) -> list[float] | None:
    """将 PDF point bbox 转为页面级 0-1 坐标，并统一保留三位小数。"""
    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0 or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return None
    normalized_bbox = [
        round(max(0.0, min(1.0, x0 / page_width)), 3),
        round(max(0.0, min(1.0, y0 / page_height)), 3),
        round(max(0.0, min(1.0, x1 / page_width)), 3),
        round(max(0.0, min(1.0, y1 / page_height)), 3),
    ]
    if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
        return None
    return normalized_bbox
