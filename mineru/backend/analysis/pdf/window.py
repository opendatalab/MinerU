# Copyright (c) Opendatalab. All rights reserved.
"""PDF 处理窗口、资源生命周期和窗口内阶段编排。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger

from ....model.runtime.hybrid import HybridLocalModelContext
from ....model.flash.pdf.document import PDFDocument, PDFPage, PDFPageTextGeometry
from .images import load_images_from_pdf_bytes_range

from ..contracts import AnalyzeEffort
from .constants import (
    BATCH_RATIO,
    LAYOUT_BASE_BATCH_SIZE,
    MFR_BASE_BATCH_SIZE,
    NOT_EXTRACT_TYPES,
    PIPELINE_DET_TYPE,
)
from .geometry import _normalize_page_size
from .layout import (
    _build_vl_style_layout_blocks,
    _collect_table_items,
    _convert_vlm_results_to_model_list,
    _normalize_xhigh_vlm_blocks,
)
from .normalization import _apply_layout_title_split
from .formulas import (
    _apply_medium_display_formula_results,
    _apply_medium_formula_number_ocr,
    _build_formula_inputs,
    _split_formula_results,
    optimize_hybrid_formula_number_blocks,
)
from .ocr import (
    _apply_ocr_rec_results,
    _build_ocr_det_type_and_mfr_enable,
    _ocr_det,
    _apply_seal_ocr,
)
from .tables import (
    _apply_medium_table_recognition,
    _apply_native_txt_table_priority,
    _apply_table_orientations,
    _fill_flash_ocr_table_contents,
    _restore_native_high_table_blocks,
    _split_native_high_table_blocks,
)
from .visuals import (
    _attach_visual_block_images,
    _supplement_missing_image_block_containers,
)
from .text.content import (
    _fill_window_block_content_and_lines,
    _validate_text_formula_window_inputs,
)


def _configured_window_size(default: int = 64) -> int:
    """读取 PDF 分析窗口大小，非法配置回退到正整数默认值。"""
    import os

    raw_value = os.getenv("MINERU_PROCESSING_WINDOW_SIZE")
    if raw_value is None:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(f"Invalid MINERU_PROCESSING_WINDOW_SIZE value: {raw_value}, use default {default}")
        return default


@dataclass(frozen=True)
class _ProcessingWindow:
    """记录单个 Hybrid 处理窗口的页码范围。"""

    index: int
    total: int
    start: int
    end: int


def _build_processing_windows(page_count: int, configured_window_size: int) -> list[_ProcessingWindow]:
    """根据页数和配置窗口大小生成稳定的 Hybrid 分段处理计划。"""
    effective_window_size = min(page_count, configured_window_size) if page_count else 0
    if effective_window_size <= 0:
        return []

    total_windows = (page_count + effective_window_size - 1) // effective_window_size
    return [
        _ProcessingWindow(
            index=window_index,
            total=total_windows,
            start=window_start,
            end=min(page_count - 1, window_start + effective_window_size - 1),
        )
        for window_index, window_start in enumerate(range(0, page_count, effective_window_size))
    ]


def _get_window_pdf_pages(pdf_doc: PDFDocument, window: _ProcessingWindow) -> list[PDFPage]:
    """按窗口闭区间获取对应的 PDFPage 对象，供窗口内后续处理复用。"""
    return [pdf_doc[page_idx] for page_idx in range(window.start, window.end + 1)]


def _log_processing_window_plan(page_count: int, configured_window_size: int, total_windows: int) -> None:
    """输出 Hybrid 分段处理计划日志，避免同步和异步入口文案漂移。"""
    logger.info(
        f"Hybrid processing-window run. page_count={page_count}, "
        f"window_size={configured_window_size}, total_windows={total_windows}"
    )


def _log_processing_window(window: _ProcessingWindow, page_count: int, image_count: int) -> None:
    """输出单个 Hybrid 处理窗口的页码范围日志。"""
    logger.info(
        f"Hybrid processing window {window.index + 1}/{window.total}: "
        f"pages {window.start + 1}-{window.end + 1}/{page_count} "
        f"({image_count} pages)"
    )


def _close_images(images_list: list[dict[str, Any]]) -> None:
    """尽力关闭窗口内全部 PIL 图片，确保异常路径也能释放图像资源。"""
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _process_flash_ocr(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    local_model_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """使用本地 OCR 为 Flash layout block 填充正文和表格内容。"""
    _validate_text_formula_window_inputs(
        images_list,
        pdf_pages,
        model_list,
        images_layout_res,
    )

    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
    empty_formula_list: list[list[dict[str, Any]]] = [[] for _ in model_list]
    ocr_res_list = _ocr_det(
        local_model_context,
        np_images,
        model_list,
        empty_formula_list,
        True,
        PIPELINE_DET_TYPE,
    )
    _apply_ocr_rec_results(local_model_context, ocr_res_list)
    model_list = _fill_window_block_content_and_lines(
        images_list,
        pdf_pages,
        model_list,
        empty_formula_list,
        ocr_res_list,
        "ocr",
        PIPELINE_DET_TYPE,
        local_model_context,
    )
    _fill_flash_ocr_table_contents(
        images_list,
        model_list,
        local_model_context,
    )
    return model_list


def _process_text_and_formulas(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    effort: Literal["medium", "high", "xhigh"],
    local_model_context: HybridLocalModelContext,
    images_layout_res: list[list[dict[str, Any]]],
    page_text_geometries: list[PDFPageTextGeometry | None] | None = None,
) -> list[list[dict[str, Any]]]:
    """在当前窗口内完成 OCR、公式、原生文本及 block 行信息回填。"""

    _validate_text_formula_window_inputs(
        images_list,
        pdf_pages,
        model_list,
        images_layout_res,
    )
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]

    # 遍历model_list,对文本块截图交由OCR识别
    # 根据 parse_mode 和 effort 决定需要ocr的文本块的类型以及只开det还是det+rec
    ocr_det_type, mfr_enable = _build_ocr_det_type_and_mfr_enable(
        parse_mode=parse_mode,
        effort=effort,
    )

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    mfd_res = _build_formula_inputs(images_layout_res)
    images_formula_list = mfd_res
    interline_enable = effort == "medium"

    # medium 识别行内和行间公式；high/xhigh 的 txt 路径只识别行内公式。
    if mfr_enable and any(mfd_res):
        images_formula_list = local_model_context.mfr_model.batch_predict(
            mfd_res,
            np_images,
            batch_size=BATCH_RATIO * MFR_BASE_BATCH_SIZE,
            interline_enable=interline_enable,
        )

    inline_formula_list, display_formula_list = _split_formula_results(images_formula_list)
    if effort == "medium":
        # 表格解析必须早于正文行填充，确保表内图片和行内公式只由表格模型消费一次。
        _apply_medium_table_recognition(
            local_model_context,
            model_list,
            inline_formula_list,
            np_images,
        )
        # 将行间公式span回填入block
        _apply_medium_display_formula_results(
            model_list,
            display_formula_list,
            images_pil_list,
        )
        # 使用ocr识别行间公式标号
        _apply_medium_formula_number_ocr(
            local_model_context,
            model_list,
            np_images,
        )

    # 行间公式标号回填到block
    for page_model_list in model_list:
        page_model_list[:] = optimize_hybrid_formula_number_blocks(page_model_list)

    need_rec_img = parse_mode == "ocr" and effort == "medium"
    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = _ocr_det(
        local_model_context,
        np_images,
        model_list,
        mfd_res,
        need_rec_img,
        ocr_det_type,
    )

    # 如果有rec_img则做ocr_rec
    if need_rec_img:
        _apply_ocr_rec_results(local_model_context, ocr_res_list)

    return _fill_window_block_content_and_lines(
        images_list,
        pdf_pages,
        model_list,
        inline_formula_list,
        ocr_res_list,
        parse_mode,
        ocr_det_type,
        local_model_context,
        page_text_geometries,
    )


def process_pdf_windows(
    file_bytes: bytes,
    document: PDFDocument,
    *,
    effort: AnalyzeEffort,
    parse_mode: Literal["txt", "ocr"],
    image_analysis: bool,
    flash_txt_mode: bool,
    hybrid_model: HybridLocalModelContext | None,
    vlm_predictor: Any,
) -> list[list[dict[str, Any]]]:
    """按固定阶段处理全部 PDF 窗口并返回完整 model-list。"""
    page_count = document.page_count
    model_list: list[list[dict[str, Any]]] = []
    if flash_txt_mode:
        # Flash 先对整份 PDF 生成完整 model_list，不依赖页面渲染和处理窗口。
        from ....model.flash import PdfModel

        model_list = PdfModel().predict(document)

    configured_window_size = _configured_window_size(default=64)
    windows = _build_processing_windows(page_count, configured_window_size)
    _log_processing_window_plan(page_count, configured_window_size, len(windows))

    for window in windows:
        window_pages = _get_window_pdf_pages(document, window)
        images_list = load_images_from_pdf_bytes_range(
            pdf_bytes=file_bytes,
            start_page_id=window.start,
            end_page_id=window.end,
            image_type="pil_img",
        )
        try:
            if len(window_pages) != len(images_list):
                raise ValueError("Hybrid processing window PDF page count does not match image count")
            images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
            _log_processing_window(window, page_count, len(images_pil_list))
            page_text_geometries: list[PDFPageTextGeometry | None] | None = (
                [None] * len(window_pages)
                if not flash_txt_mode and parse_mode == "txt" and effort in {"medium", "high"}
                else None
            )

            if flash_txt_mode:
                # Flash 仅切割当前窗口的外层列表，用于页图释放前原地补充视觉块裁图。
                window_model_list = model_list[window.start : window.end + 1]
            else:
                local_model_context = hybrid_model
                if local_model_context is None:
                    raise ValueError("Hybrid local model context is required outside Flash TXT mode")
                np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]
                images_layout_res = local_model_context.layout_model.batch_predict(
                    images_pil_list, batch_size=min(8, BATCH_RATIO * LAYOUT_BASE_BATCH_SIZE)
                )

                # 使用小模型layout时对layout的表格做旋转检测
                if effort in ["flash", "medium", "high"]:
                    table_items = _collect_table_items(images_layout_res, np_images)
                    if table_items:
                        _apply_table_orientations(
                            table_items,
                            parse_mode,
                            window_pages,
                            images_list,
                            local_model_context,
                        )

                vl_style_layout_blocks = _build_vl_style_layout_blocks(images_layout_res, images_pil_list)

                if parse_mode == "txt" and effort in {"medium", "high"}:
                    native_table_summary = _apply_native_txt_table_priority(
                        vl_style_layout_blocks,
                        images_layout_res,
                        window_pages,
                        images_list,
                        effort=effort,
                        page_text_geometries=page_text_geometries,
                    )
                    if native_table_summary.total:
                        native_table_stats = {
                            "effort": effort,
                            "total": native_table_summary.total,
                            "accepted": native_table_summary.accepted,
                            "complex_fallbacks": native_table_summary.complex_fallbacks,
                            "rejected": native_table_summary.rejected,
                            "errors": native_table_summary.errors,
                            "removed_internal_text": native_table_summary.removed_internal_text,
                            "removed_formula_blocks": native_table_summary.removed_formula_blocks,
                            "removed_formula_layout_items": native_table_summary.removed_formula_layout_items,
                        }
                        logger.bind(native_table_priority=native_table_stats).info(
                            "Hybrid native table priority. "
                            f"effort={native_table_stats['effort']}, total={native_table_stats['total']}, "
                            f"accepted={native_table_stats['accepted']}, "
                            f"complex_fallbacks={native_table_stats['complex_fallbacks']}, "
                            f"rejected={native_table_stats['rejected']}, "
                            f"errors={native_table_stats['errors']}, "
                            f"removed_internal_text={native_table_stats['removed_internal_text']}, "
                            f"removed_formula_blocks={native_table_stats['removed_formula_blocks']}, "
                            f"removed_formula_layout_items={native_table_stats['removed_formula_layout_items']}"
                        )

                if parse_mode == "txt":
                    if effort == "medium":
                        window_model_list = vl_style_layout_blocks
                    elif effort == "high":
                        high_vlm_blocks, accepted_native_tables = _split_native_high_table_blocks(vl_style_layout_blocks)
                        high_vlm_results = vlm_predictor.batch_extract_with_layout(
                            images=images_pil_list,
                            blocks_list=high_vlm_blocks,
                            not_extract_list=NOT_EXTRACT_TYPES,
                            image_analysis=False,
                        )
                        window_model_list = _restore_native_high_table_blocks(
                            high_vlm_results,
                            accepted_native_tables,
                        )
                    elif effort == "xhigh":
                        window_model_list = vlm_predictor.batch_two_step_extract(
                            images=images_pil_list,
                            not_extract_list=NOT_EXTRACT_TYPES,
                            image_analysis=image_analysis,
                        )
                    else:
                        raise ValueError(f"Unsupported analyze effort: {effort}")
                elif parse_mode == "ocr":
                    if effort in ["flash", "medium"]:
                        window_model_list = vl_style_layout_blocks
                    elif effort == "high":
                        window_model_list = vlm_predictor.batch_extract_with_layout(
                            images=images_pil_list,
                            blocks_list=vl_style_layout_blocks,
                            image_analysis=False,
                        )
                    elif effort == "xhigh":
                        window_model_list = vlm_predictor.batch_two_step_extract(
                            images=images_pil_list,
                            image_analysis=image_analysis,
                        )
                    else:
                        raise ValueError(f"Unsupported analyze effort: {effort}")
                else:
                    raise ValueError(f"Unsupported parse mode: {parse_mode}")

                if effort in {"high", "xhigh"}:
                    window_model_list = _convert_vlm_results_to_model_list(window_model_list)
                if effort == "xhigh":
                    _normalize_xhigh_vlm_blocks(window_model_list)
                    _apply_layout_title_split(
                        window_model_list,
                        images_layout_res,
                        [_normalize_page_size(image) for image in images_pil_list],
                    )

                if effort == "flash":
                    window_model_list = _process_flash_ocr(
                        images_list,
                        window_pages,
                        window_model_list,
                        local_model_context,
                        images_layout_res,
                    )
                    # Flash OCR 在表内对象清理后复用统一公式编号合并，确保视觉裁图包含编号区域。
                    for page_model_list in window_model_list:
                        page_model_list[:] = optimize_hybrid_formula_number_blocks(page_model_list)
                else:
                    window_model_list = _process_text_and_formulas(
                        images_list,
                        window_pages,
                        window_model_list,
                        parse_mode,
                        effort,
                        local_model_context,
                        images_layout_res,
                        page_text_geometries,
                    )

                if effort in {"medium", "high"}:
                    _apply_seal_ocr(local_model_context, window_model_list, np_images)
                elif effort == "xhigh":
                    _supplement_missing_image_block_containers(
                        window_model_list,
                        vl_style_layout_blocks,
                    )

            _attach_visual_block_images(
                window_model_list,
                images_list,
                page_start_index=window.start,
            )
            if not flash_txt_mode:
                model_list.extend(window_model_list)
        finally:
            _close_images(images_list)

    return model_list
