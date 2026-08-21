# Copyright (c) Opendatalab. All rights reserved.
"""表格方向检测、Medium/Low 表格识别与文本投影。"""

from __future__ import annotations

import html
import math
from collections import Counter
from typing import Any, Literal

import cv2
import numpy as np
from loguru import logger

from mineru.backend.local_model_runtime import HybridLocalModelContext, run_ocr_inference
from mineru.model.model_types import AtomicModelName
from mineru.types import RAW_FORMULA_NUMBER, BBox, BlockType, ContentType
from mineru.utils.bbox_utils import (
    calculate_overlap_area_in_bbox1_area_ratio,
    normalize_to_int_bbox,
)
from mineru.utils.ocr_utils import mask_formula_regions_for_ocr_det
from mineru.utils.pdf_document import PDFPage, get_lines_from_chars
from mineru.utils.pdf_text_styles import PDFTextLinkLine, PDFTextStyleLine
from mineru.utils.spatial_text import project_ocr_table_text, project_pdf_table_text

from .constants import (
    BATCH_RATIO,
    OCR_DET_BASE_BATCH_SIZE,
    TABLE_TEXT_LINE_OVERLAP_THRESHOLD,
    TABLE_TEXT_ORIENTATION_ANGLES,
    TABLE_TEXT_ORIENTATION_MIN_DOMINANCE_RATIO,
    TABLE_TEXT_ORIENTATION_MIN_VALID_LINES,
    _LOW_TXT_VISUAL_RUN_ANGLES,
)
from .geometry import (
    _bbox_to_pixel_bbox,
    _encode_page_crop_as_jpeg_data_uri,
    _get_medium_table_virtual_image_bbox,
    _medium_bbox_to_quad,
    _normalize_medium_content,
    _normalize_page_size,
    _normalize_visual_block_angle,
    _rotate_medium_table_bbox,
    _rotate_visual_block_image_to_upright,
    _sidecar_bbox_to_page_bbox,
    _table_bbox_center,
)
from .text.models import _AnalyzeSpan
from .text.native import __replace_ligatures, __replace_unicode, _is_supported_rotation
from .text.styles import build_pdf_native_visual_lines_and_styles


def _apply_table_rotate_labels(
    table_items: list[dict[str, Any]],
    rotate_labels: list[str],
) -> None:
    """按分类输入顺序将表格旋转角写回原始 layout 检测项。"""
    if len(rotate_labels) != len(table_items):
        raise ValueError("Hybrid table orientation result count mismatch")
    for table_item, rotate_label in zip(table_items, rotate_labels):
        table_item["layout_item"]["angle"] = int(rotate_label or "0")


def _visual_line_items_to_spans(line_items: list[Any]) -> list[_AnalyzeSpan]:
    """把 Flash 视觉 run 转为 Hybrid 文本分配使用的私有 span。"""

    page_spans: list[_AnalyzeSpan] = []
    for line_item in line_items:
        content = __replace_ligatures(__replace_unicode(line_item.text)).strip()
        if not content:
            continue
        page_spans.append(
            _AnalyzeSpan(
                type=ContentType.TEXT,
                bbox=line_item.bbox,
                content=content,
                score=1.0,
            )
        )
    return page_spans


def _build_pdf_text_visual_run_data(
    pdf_page: PDFPage,
) -> tuple[list[_AnalyzeSpan], list[PDFTextStyleLine], list[PDFTextLinkLine]]:
    """一次构造 Low/TXT 视觉 span、文本样式和超链接证据。"""

    _page_chars, line_items, style_lines, link_lines = (
        build_pdf_native_visual_lines_and_styles(
            pdf_page,
            supported_angles=_LOW_TXT_VISUAL_RUN_ANGLES,
        )
    )
    return _visual_line_items_to_spans(line_items), style_lines, link_lines


def _build_pdf_text_visual_run_spans(pdf_page: PDFPage) -> list[_AnalyzeSpan]:
    """将 Low/TXT 原生粗行按 Flash 字符间隙拆成可独立分配的视觉 run。"""

    page_spans, _style_lines, _link_lines = _build_pdf_text_visual_run_data(
        pdf_page
    )
    return page_spans


def _detect_table_angle_from_pdf_lines(
    pdf_lines: list[dict[str, Any]],
    table_bbox: BBox,
) -> int | None:
    """统计表格框内标准方向文本行，满足强众数门槛时返回表格角度。"""
    angle_counts: Counter[int] = Counter()
    for pdf_line in pdf_lines:
        try:
            raw_bbox = pdf_line.get("bbox")
            line_bbox = getattr(raw_bbox, "bbox", raw_bbox)
            if line_bbox is None or len(line_bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in line_bbox]
            if x1 <= x0 or y1 <= y0:
                continue

            line_text = "".join(str(pdf_span.get("text", "") or "") for pdf_span in pdf_line.get("spans", [])).strip()
            if not line_text:
                continue

            rotation = float(pdf_line.get("rotation", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue

        if (
            calculate_overlap_area_in_bbox1_area_ratio(
                (x0, y0, x1, y1),
                table_bbox,
            )
            <= TABLE_TEXT_LINE_OVERLAP_THRESHOLD
        ):
            continue
        if not _is_supported_rotation(rotation):
            continue

        angle = int(round(math.degrees(rotation))) % 360
        if angle in TABLE_TEXT_ORIENTATION_ANGLES:
            angle_counts[angle] += 1

    valid_line_count = sum(angle_counts.values())
    if valid_line_count < TABLE_TEXT_ORIENTATION_MIN_VALID_LINES:
        return None

    max_count = max(angle_counts.values())
    dominant_angles = [angle for angle, count in angle_counts.items() if count == max_count]
    if len(dominant_angles) != 1:
        return None
    if max_count / valid_line_count < TABLE_TEXT_ORIENTATION_MIN_DOMINANCE_RATIO:
        return None
    return dominant_angles[0]


def _resolve_txt_table_orientations(
    table_items: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    images_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """优先用原生 PDF 文本行写回表格角度，并返回需要视觉兜底的表格。"""
    fallback_table_items: list[dict[str, Any]] = []
    page_lines_cache: dict[int, list[dict[str, Any]] | None] = {}

    for table_item in table_items:
        page_idx = table_item.get("page_idx")
        if not isinstance(page_idx, int) or page_idx < 0 or page_idx >= len(pdf_pages) or page_idx >= len(images_list):
            fallback_table_items.append(table_item)
            continue

        try:
            pdf_page = pdf_pages[page_idx]
            render_scale = float(images_list[page_idx].get("scale", 0) or 0)
            table_bbox = _sidecar_bbox_to_page_bbox(
                table_item["layout_item"].get("bbox"),
                tuple(float(value) for value in pdf_page.size),
                render_scale,
            )
        except Exception as exc:
            logger.warning(f"Hybrid txt table orientation falls back to visual model: page_idx={page_idx}, error={exc}")
            fallback_table_items.append(table_item)
            continue

        if table_bbox is None:
            fallback_table_items.append(table_item)
            continue

        if page_idx not in page_lines_cache:
            try:
                page_lines_cache[page_idx] = get_lines_from_chars(pdf_page.get_chars())
            except Exception as exc:
                logger.warning(f"Hybrid txt table orientation falls back to visual model: page_idx={page_idx}, error={exc}")
                page_lines_cache[page_idx] = None

        pdf_lines = page_lines_cache[page_idx]
        if pdf_lines is None:
            fallback_table_items.append(table_item)
            continue

        table_angle = _detect_table_angle_from_pdf_lines(pdf_lines, table_bbox)
        if table_angle is None:
            fallback_table_items.append(table_item)
            continue
        table_item["layout_item"]["angle"] = table_angle

    return fallback_table_items


def _apply_table_orientations(
    table_items: list[dict[str, Any]],
    parse_mode: Literal["txt", "ocr"],
    pdf_pages: list[PDFPage],
    images_list: list[dict[str, Any]],
    hybrid_model: HybridLocalModelContext,
) -> None:
    """按解析模式写回表格角度，文本证据不足时批量调用视觉方向模型。"""
    if parse_mode == "txt":
        fallback_table_items = _resolve_txt_table_orientations(
            table_items,
            pdf_pages,
            images_list,
        )
    elif parse_mode == "ocr":
        fallback_table_items = table_items
    else:
        raise ValueError(f"Unsupported parse mode: {parse_mode}")

    if not fallback_table_items:
        return

    rotate_labels = hybrid_model.table_orientation_cls_model.batch_predict(
        fallback_table_items,
        det_batch_size=BATCH_RATIO * OCR_DET_BASE_BATCH_SIZE,
        tqdm_enable=True,
    )
    _apply_table_rotate_labels(fallback_table_items, rotate_labels)


def _table_bbox_intersection(bbox1: BBox, bbox2: BBox) -> BBox | None:
    """计算两个 bbox 的有效交集，无重叠时返回 None。"""
    x0 = max(float(bbox1[0]), float(bbox2[0]))
    y0 = max(float(bbox1[1]), float(bbox2[1]))
    x1 = min(float(bbox1[2]), float(bbox2[2]))
    y1 = min(float(bbox1[3]), float(bbox2[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _select_table_owner(
    item_bbox: BBox,
    table_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """为 low/medium 对象匹配所属表格，多表命中时选择交叠面积最大的表格。"""
    center_x, center_y = _table_bbox_center(item_bbox)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for table_entry in table_entries:
        table_bbox = table_entry["table_bbox"]
        if not (
            float(table_bbox[0]) <= center_x <= float(table_bbox[2])
            and float(table_bbox[1]) <= center_y <= float(table_bbox[3])
        ):
            continue
        overlap_bbox = _table_bbox_intersection(item_bbox, table_bbox)
        if overlap_bbox is None:
            continue
        overlap_area = float(overlap_bbox[2] - overlap_bbox[0]) * float(overlap_bbox[3] - overlap_bbox[1])
        candidates.append((overlap_area, table_entry))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _collect_medium_table_tasks(
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> list[dict[str, Any]]:
    """从当前窗口 model_list 收集表格任务，并提前消费表内图片和行内公式。"""
    table_tasks: list[dict[str, Any]] = []
    for page_idx, (page_model_list, page_inline_formulas, np_image) in enumerate(
        zip(model_list, inline_formula_list, np_images)
    ):
        image_h, image_w = np_image.shape[:2]
        page_size = (image_w, image_h)
        table_entries: list[dict[str, Any]] = []
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            pixel_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            table_bbox = normalize_to_int_bbox(pixel_bbox, image_size=(image_h, image_w))
            if table_bbox is None:
                continue
            x0, y0, x1, y1 = table_bbox
            table_crop = np_image[y0:y1, x0:x1].copy()
            if table_crop.size == 0:
                continue
            table_entries.append(
                {
                    "table_block": block,
                    "table_bbox": table_bbox,
                    "table_crop": table_crop,
                    "angle": _normalize_visual_block_angle(block.get("angle", 0)),
                    "inline_objects": [],
                }
            )

        if not table_entries:
            continue

        consumed_image_ids: set[int] = set()
        consumed_formula_ids: set[int] = set()
        for block in page_model_list:
            if block.get("type") != BlockType.IMAGE:
                continue
            image_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if image_bbox is None:
                continue
            owner = _select_table_owner(image_bbox, table_entries)
            if owner is None:
                continue
            owner["inline_objects"].append(
                {
                    "kind": "image",
                    "page_bbox": image_bbox,
                    "score": float(block.get("score", 1.0) or 0.0),
                }
            )
            consumed_image_ids.add(id(block))

        for formula in page_inline_formulas:
            latex = _normalize_medium_content(formula.get("latex", ""))
            formula_bbox = _bbox_to_pixel_bbox(formula.get("bbox"), page_size)
            if formula_bbox is None:
                continue
            owner = _select_table_owner(formula_bbox, table_entries)
            if owner is None:
                continue
            owner["inline_objects"].append(
                {
                    "kind": "formula",
                    "page_bbox": formula_bbox,
                    "latex": latex,
                    "score": float(formula.get("score", 1.0) or 0.0),
                }
            )
            consumed_formula_ids.add(id(formula))

        # 匹配完成后立即删除原始对象；后续 OCR 或结构识别失败均不回滚。
        page_model_list[:] = [block for block in page_model_list if id(block) not in consumed_image_ids]
        page_inline_formulas[:] = [formula for formula in page_inline_formulas if id(formula) not in consumed_formula_ids]

        for table_entry in table_entries:
            table_bbox = table_entry["table_bbox"]
            table_crop = table_entry["table_crop"]
            angle = table_entry["angle"]
            crop_h, crop_w = table_crop.shape[:2]
            rotated_crop = _rotate_visual_block_image_to_upright(table_crop, angle)
            rotated_h, rotated_w = rotated_crop.shape[:2]
            prepared_objects: list[dict[str, Any]] = []
            for inline_object in table_entry["inline_objects"]:
                overlap_bbox = _table_bbox_intersection(inline_object["page_bbox"], table_bbox)
                if overlap_bbox is None:
                    continue
                relative_bbox = (
                    float(overlap_bbox[0]) - float(table_bbox[0]),
                    float(overlap_bbox[1]) - float(table_bbox[1]),
                    float(overlap_bbox[2]) - float(table_bbox[0]),
                    float(overlap_bbox[3]) - float(table_bbox[1]),
                )
                rotated_bbox = _rotate_medium_table_bbox(relative_bbox, crop_w, crop_h, angle)
                if inline_object["kind"] == "formula":
                    latex = inline_object["latex"]
                    content = f"<eq>{html.escape(latex)}</eq>" if latex else ""
                    token_bbox = rotated_bbox
                else:
                    image_src = _encode_page_crop_as_jpeg_data_uri(np_image, inline_object["page_bbox"], angle)
                    content = f'<img src="{image_src}"/>' if image_src else ""
                    token_bbox = _get_medium_table_virtual_image_bbox(
                        rotated_bbox,
                        (rotated_w, rotated_h),
                    )
                prepared_objects.append(
                    {
                        **inline_object,
                        "mask_bbox": rotated_bbox,
                        "token_bbox": token_bbox,
                        "content": content,
                    }
                )

            table_tasks.append(
                {
                    "page_idx": page_idx,
                    "table_block": table_entry["table_block"],
                    "table_bbox": table_bbox,
                    "angle": angle,
                    "table_img": rotated_crop,
                    "wired_table_img": rotated_crop,
                    "ocr_result": [],
                    "table_res": {},
                    "inline_objects": prepared_objects,
                }
            )
    return table_tasks


def _sort_medium_table_ocr_result(ocr_result: list[list[Any]]) -> None:
    """按 token 顶边和左边坐标排序，保证表格 OCR 输入顺序稳定。"""

    def sort_key(item: list[Any]) -> tuple[float, float]:
        """提取四点框的最小 y/x，兼容普通列表和 numpy 数组。"""
        box = np.asarray(item[0], dtype=np.float32).reshape(-1, 2)
        return float(np.min(box[:, 1])), float(np.min(box[:, 0]))

    ocr_result.sort(key=sort_key)


def _prepare_medium_table_ocr_results(
    local_model_context: HybridLocalModelContext,
    table_tasks: list[dict[str, Any]],
) -> None:
    """遮盖表内图片和公式后逐表执行 OCR，并合并内联对象 token。"""
    table_ocr_model = None
    try:
        table_ocr_model = local_model_context.get_ocr_model(
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
        )
    except Exception as exc:
        logger.warning(f"Hybrid medium table OCR model initialization failed: {exc}")

    for table_task in table_tasks:
        ocr_result: list[list[Any]] = []
        bgr_image = cv2.cvtColor(table_task["table_img"], cv2.COLOR_RGB2BGR)
        mask_boxes = [{"bbox": item["mask_bbox"]} for item in table_task["inline_objects"]]
        masked_image = mask_formula_regions_for_ocr_det(bgr_image, mask_boxes)

        if table_ocr_model is not None:
            try:
                page_ocr_results = run_ocr_inference(table_ocr_model.ocr, masked_image)
                raw_ocr_result = page_ocr_results[0] if page_ocr_results else None
                for raw_item in raw_ocr_result or []:
                    if not raw_item or len(raw_item) < 2:
                        continue
                    box, rec_result = raw_item[0], raw_item[1]
                    if not rec_result or len(rec_result) < 2:
                        continue
                    text, score = rec_result[0], rec_result[1]
                    normalized_text = _normalize_medium_content(text)
                    if not normalized_text:
                        continue
                    ocr_result.append(
                        [
                            np.asarray(box, dtype=np.float32),
                            html.escape(normalized_text),
                            float(score or 0.0),
                        ]
                    )
            except Exception as exc:
                logger.warning(f"Hybrid medium table OCR failed: page_idx={table_task['page_idx']}, error={exc}")

        for inline_object in table_task["inline_objects"]:
            if not inline_object["content"]:
                continue
            ocr_result.append(
                [
                    _medium_bbox_to_quad(inline_object["token_bbox"]),
                    inline_object["content"],
                    inline_object["score"],
                ]
            )
        _sort_medium_table_ocr_result(ocr_result)
        table_task["ocr_result"] = ocr_result


def _trim_medium_table_html(html_code: Any) -> str:
    """从模型输出中截取核心 table HTML；未带外围标签时保留原字符串。"""
    if not isinstance(html_code, str):
        return ""
    stripped_html = html_code.strip()
    lower_html = stripped_html.lower()
    start_index = lower_html.find("<table")
    end_index = lower_html.rfind("</table>")
    if start_index >= 0 and end_index >= start_index:
        return stripped_html[start_index : end_index + len("</table>")]
    return stripped_html


def _apply_medium_table_recognition(
    local_model_context: HybridLocalModelContext,
    model_list: list[list[dict[str, Any]]],
    inline_formula_list: list[list[dict[str, Any]]],
    np_images: list[np.ndarray],
) -> None:
    """基于 model_list 执行 medium 表格解析，采用分类 batch、无线 batch、有线单表调度。"""
    table_tasks = _collect_medium_table_tasks(model_list, inline_formula_list, np_images)
    if not table_tasks:
        return

    _prepare_medium_table_ocr_results(local_model_context, table_tasks)

    try:
        local_model_context.table_cls_model.batch_predict(table_tasks)
    except Exception as exc:
        # 分类失败不阻断无线模型，未获得分类结果的表格按 wireless-only 处理。
        logger.warning(f"Hybrid medium table classification failed: {exc}")

    try:
        local_model_context.wireless_table_model.batch_predict(table_tasks)
    except Exception as exc:
        logger.warning(f"Hybrid medium wireless table recognition failed: {exc}")

    for table_task in table_tasks:
        cls_label = table_task["table_res"].get("cls_label")
        try:
            cls_score = float(table_task["table_res"].get("cls_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            cls_score = 0.0
        use_wired_model = cls_label == AtomicModelName.WiredTable or (
            cls_label == AtomicModelName.WirelessTable and cls_score < 0.9
        )
        if use_wired_model:
            try:
                wireless_html = table_task["table_res"].get("html", "") or ""
                wired_html = local_model_context.wired_table_model.predict(
                    table_task["wired_table_img"],
                    table_task["ocr_result"],
                    wireless_html,
                )
                if wired_html is not None:
                    table_task["table_res"]["html"] = wired_html
            except Exception as exc:
                logger.warning(f"Hybrid medium wired table recognition failed: page_idx={table_task['page_idx']}, error={exc}")

        html_code = _trim_medium_table_html(table_task["table_res"].get("html", ""))
        if html_code:
            table_task["table_block"]["content"] = html_code


def _remove_low_table_inner_blocks(
    images_list: list[dict[str, Any]],
    model_list: list[list[dict[str, Any]]],
) -> None:
    """按表格中心点归属规则移除 low 表内图片、公式和公式编号块。"""
    inner_block_types = {
        BlockType.IMAGE,
        BlockType.EQUATION,
        RAW_FORMULA_NUMBER,
    }
    for image_dict, page_model_list in zip(images_list, model_list):
        page_size = _normalize_page_size(image_dict["img_pil"])
        table_entries = []
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            table_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if table_bbox is not None:
                table_entries.append({"table_bbox": table_bbox})

        if not table_entries:
            continue

        retained_blocks = []
        for block in page_model_list:
            if block.get("type") not in inner_block_types:
                retained_blocks.append(block)
                continue
            block_bbox = _bbox_to_pixel_bbox(block.get("bbox"), page_size)
            if block_bbox is None or _select_table_owner(block_bbox, table_entries) is None:
                retained_blocks.append(block)
        page_model_list[:] = retained_blocks


def _fill_low_table_contents(
    images_list: list[dict[str, Any]],
    pdf_pages: list[PDFPage],
    model_list: list[list[dict[str, Any]]],
    parse_mode: Literal["txt", "ocr"],
    local_model_context: HybridLocalModelContext,
) -> None:
    """为 low 表格回填空间投影纯文本，单表失败时保留空内容并继续。"""
    table_entries: list[dict[str, Any]] = []
    for page_idx, page_model_list in enumerate(model_list):
        table_idx = 0
        for block in page_model_list:
            if block.get("type") != BlockType.TABLE:
                continue
            block["content"] = ""
            table_entries.append(
                {
                    "page_idx": page_idx,
                    "table_idx": table_idx,
                    "block": block,
                }
            )
            table_idx += 1

    if not table_entries:
        return

    # 表格一旦认领内部对象便立即从 model_list 删除，后续文本回填失败也不恢复。
    _remove_low_table_inner_blocks(images_list, model_list)

    if parse_mode == "txt":
        page_chars_cache: dict[int, list[Any]] = {}
        for table_entry in table_entries:
            page_idx = table_entry["page_idx"]
            table_idx = table_entry["table_idx"]
            table_block = table_entry["block"]
            try:
                pdf_page = pdf_pages[page_idx]
                page_width, page_height = pdf_page.size
                table_bbox = _bbox_to_pixel_bbox(
                    table_block.get("bbox"),
                    (int(page_width), int(page_height)),
                )
                if table_bbox is None:
                    raise ValueError("invalid table bbox")
                if page_idx not in page_chars_cache:
                    page_chars_cache[page_idx] = pdf_page.get_chars()
                table_block["content"] = project_pdf_table_text(
                    page_chars_cache[page_idx],
                    table_bbox,
                    table_block.get("angle", 0),
                )
                if not table_block["content"]:
                    logger.warning(
                        "Hybrid low table text is empty: "
                        f"parse_mode={parse_mode}, page_idx={page_idx}, "
                        f"table_idx={table_idx}, bbox={table_block.get('bbox')}"
                    )
            except Exception as exc:
                logger.warning(
                    "Hybrid low table text failed: "
                    f"parse_mode={parse_mode}, page_idx={page_idx}, "
                    f"table_idx={table_idx}, bbox={table_block.get('bbox')}, error={exc}"
                )
        return

    if parse_mode != "ocr":
        raise ValueError(f"Unsupported parse mode: {parse_mode}")

    try:
        table_ocr_model = local_model_context.get_ocr_model(
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
        )
    except Exception as exc:
        for table_entry in table_entries:
            logger.warning(
                "Hybrid low table text failed: "
                f"parse_mode={parse_mode}, page_idx={table_entry['page_idx']}, "
                f"table_idx={table_entry['table_idx']}, "
                f"bbox={table_entry['block'].get('bbox')}, "
                f"error=OCR model initialization failed: {exc}"
            )
        return

    page_image_cache: dict[int, np.ndarray] = {}
    for table_entry in table_entries:
        page_idx = table_entry["page_idx"]
        table_idx = table_entry["table_idx"]
        table_block = table_entry["block"]
        try:
            if page_idx not in page_image_cache:
                page_image_cache[page_idx] = np.asarray(images_list[page_idx]["img_pil"]).copy()
            np_image = page_image_cache[page_idx]
            image_height, image_width = np_image.shape[:2]
            pixel_bbox = _bbox_to_pixel_bbox(
                table_block.get("bbox"),
                (image_width, image_height),
            )
            table_bbox = normalize_to_int_bbox(
                pixel_bbox,
                image_size=(image_height, image_width),
            )
            if table_bbox is None:
                raise ValueError("invalid table bbox")
            x0, y0, x1, y1 = table_bbox
            table_crop = np_image[y0:y1, x0:x1].copy()
            if table_crop.size == 0:
                raise ValueError("empty table crop")

            angle = _normalize_visual_block_angle(table_block.get("angle", 0))
            rotated_crop = _rotate_visual_block_image_to_upright(table_crop, angle)
            bgr_crop = cv2.cvtColor(rotated_crop, cv2.COLOR_RGB2BGR)
            page_ocr_results = run_ocr_inference(table_ocr_model.ocr, bgr_crop)
            raw_ocr_result = page_ocr_results[0] if page_ocr_results else None
            rotated_height, rotated_width = rotated_crop.shape[:2]
            table_block["content"] = project_ocr_table_text(
                raw_ocr_result,
                (rotated_width, rotated_height),
            )
            if not table_block["content"]:
                logger.warning(
                    "Hybrid low table text is empty: "
                    f"parse_mode={parse_mode}, page_idx={page_idx}, "
                    f"table_idx={table_idx}, bbox={table_block.get('bbox')}"
                )
        except Exception as exc:
            logger.warning(
                "Hybrid low table text failed: "
                f"parse_mode={parse_mode}, page_idx={page_idx}, "
                f"table_idx={table_idx}, bbox={table_block.get('bbox')}, error={exc}"
            )
