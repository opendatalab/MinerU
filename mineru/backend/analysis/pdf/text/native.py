# Copyright (c) Opendatalab. All rights reserved.
"""PDF 字符提取、Span 匹配和上下标内容重建。"""

from __future__ import annotations

import collections
import math
import re
import statistics
import unicodedata
from typing import Any, Callable, cast

import cv2
import numpy as np
from loguru import logger
from pdftext.schema import Char

from .....types import BBox, BlockType, ContentType
from .....model.flash.pdf.document import PDFPage, get_lines_from_chars
from .....model.flash.pdf.script_geometry import ScriptRole, classify_char_script_roles
from .....model.flash.pdf.text_styles import PDF_NATIVE_SCRIPT_MARKUP_KEY
from .....utils.geometry import calculate_overlap_area_in_bbox1_area_ratio
from .....utils.image import calculate_contrast
from ..images import get_crop_img
from .models import _AnalyzeSpan

MAX_NATIVE_TEXT_CHARS_PER_PAGE = 65535
PRIVATE_USE_AREA_START = 0xE000
PRIVATE_USE_AREA_END = 0xF8FF
PRIVATE_USE_TEXT_COUNT_THRESHOLD = 2
PRIVATE_USE_TEXT_RATIO_THRESHOLD = 0.05
PRIVATE_USE_TEXT_RUN_THRESHOLD = 2
POST_OCR_FALLBACK_CONTENT_KEY = "_post_ocr_fallback_content"
POST_OCR_FALLBACK_SCORE_KEY = "_post_ocr_fallback_score"
POST_OCR_REASON_KEY = "_post_ocr_reason"
POST_OCR_REASON_PRIVATE_USE_TEXT = "private_use_text"
SPACING_DIACRITIC_MIN_OVERLAP_RATIO = 0.5
_SPACING_DIACRITIC_TO_COMBINING = {
    "`": "\u0300",
    "^": "\u0302",
    "¨": "\u0308",
    "¯": "\u0304",
    "´": "\u0301",
    "¸": "\u0327",
    "ˆ": "\u0302",
    "ˇ": "\u030c",
    "˘": "\u0306",
    "˙": "\u0307",
    "˚": "\u030a",
    "˛": "\u0328",
    "˜": "\u0303",
    "˝": "\u030b",
}


def __replace_ligatures(text: str) -> str:
    """将 PDF 字符流中的常见拉丁连字替换为普通字符序列。"""
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}
    return re.sub("|".join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)


def __replace_unicode(text: str) -> str:
    """清理 PDF 字符流中的换行和特殊控制字符。"""
    ligatures = {
        "\r\n": "",
        "\u0002": "-",
    }
    return re.sub("|".join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)


"""pdf_text dict方案 char级别"""


def txt_spans_extract(
    pdf_page: PDFPage,
    spans: list[_AnalyzeSpan],
    pil_img: Any,
    scale: float,
    all_bboxes: list[tuple[Any, ...]],
    all_discarded_blocks: list[tuple[Any, ...]],
    *,
    page_chars: list[Char] | dict[str, list[Char]] | None = None,
    tight_bboxes: dict[int, BBox] | None = None,
    origins: dict[int, tuple[float, float]] | None = None,
    detect_scripts: bool = True,
) -> list[_AnalyzeSpan]:
    """从 PDF 原生字符中提取文本 Span，并允许复用调用方已读取的页面字符。"""
    page_char_count = None
    try:
        page_char_count = pdf_page.get_char_count()
    except Exception as exc:
        logger.debug(f"Failed to get page char count before txt extraction: {exc}")

    if page_char_count is not None and page_char_count > MAX_NATIVE_TEXT_CHARS_PER_PAGE:
        logger.info(f"Fallback to post-OCR in txt_spans_extract due to high char count: count_chars={page_char_count}")
        need_ocr_spans = [span for span in spans if span.type == ContentType.TEXT]
        return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)

    if page_chars is None:
        page_text_geometry = pdf_page.get_chars_with_geometry()
        page_chars = page_text_geometry.chars
        tight_bboxes = page_text_geometry.tight_bboxes
        origins = page_text_geometry.origins
    page_all_chars = _get_chars_for_span_fill(page_chars)

    # 计算所有span的高度的中位数
    span_height_list = []
    for span in spans:
        if span.type in [ContentType.TEXT]:
            span_height = span.bbox[3] - span.bbox[1]
            span.metadata["height"] = span_height
            span.metadata["width"] = span.bbox[2] - span.bbox[0]
            span_height_list.append(span_height)
    if len(span_height_list) == 0:
        return spans
    else:
        median_span_height = statistics.median(span_height_list)

    useful_spans = []
    unuseful_spans = []
    # 纵向span的两个特征：1. 高度超过多个line 2. 高宽比超过某个值
    vertical_spans = []
    for span in spans:
        if span.type in [ContentType.TEXT]:
            for block in all_bboxes + all_discarded_blocks:
                if block[7] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.EQUATION]:
                    continue
                if calculate_overlap_area_in_bbox1_area_ratio(span.bbox, block[0:4]) > 0.5:
                    if (
                        span.metadata["height"] > median_span_height * 2.3
                        and span.metadata["height"] > span.metadata["width"] * 2.3
                    ):
                        vertical_spans.append(span)
                    elif block in all_bboxes:
                        useful_spans.append(span)
                    else:
                        unuseful_spans.append(span)
                    break

    """垂直的span框直接用line进行填充"""
    if len(vertical_spans) > 0:
        pdf_lines = [line for line in get_lines_from_chars(page_chars) if _is_supported_rotation(line["rotation"])]
        for pdf_line in pdf_lines:
            for span in vertical_spans:
                if calculate_overlap_area_in_bbox1_area_ratio(pdf_line["bbox"].bbox, span.bbox) > 0.5:
                    for pdf_span in pdf_line["spans"]:
                        span.content += pdf_span["text"]
                    break

        for span in vertical_spans:
            if len(span.content) == 0:
                spans.remove(span)

    """水平的span框先用char填充，再用ocr填充空的span框"""
    new_spans = []

    for span in useful_spans + unuseful_spans:
        if span.type in [ContentType.TEXT]:
            span.metadata["chars"] = []
            new_spans.append(span)

    need_ocr_spans = fill_char_in_spans(
        new_spans,
        page_all_chars,
        median_span_height,
        tight_bboxes=tight_bboxes,
        origins=origins,
        detect_scripts=detect_scripts,
    )

    return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)


def _is_supported_rotation(rotation: float) -> bool:
    """判断 pdftext 旋转角是否属于当前可回填的四个标准方向。"""
    rotation_degrees = math.degrees(rotation)
    return any(abs(rotation_degrees - angle) < 0.1 for angle in [0, 90, 180, 270])


def _rotation_distance_degrees(first: float, second: float) -> float:
    """返回两个弧度方向之间不超过 180 度的最短夹角。"""
    first_degrees = math.degrees(first) % 360
    second_degrees = math.degrees(second) % 360
    return abs((first_degrees - second_degrees + 180) % 360 - 180)


def _get_char_fill_key(char: Char) -> tuple[str, Any]:
    """生成字符回填判定 key，优先使用 pdftext 提供的页内 char_idx。"""
    char_idx = char.get("char_idx")
    if char_idx is not None:
        return ("char_idx", char_idx)
    return ("object_id", id(char))


def _iter_line_chars(line: dict[str, Any]) -> list[Char]:
    """按 pdftext line/span 结构展开字符，兼容缺少 chars 字段的异常 span。"""
    return [char for span in line.get("spans", []) for char in span.get("chars", [])]


def _is_visible_standard_rotation_char(char: Char) -> bool:
    """判断字符是否是可见的标准方向正文字符，避免换行控制符误放行水印。"""
    text = str(char.get("char", ""))
    if not text or text.isspace() or text in {"\r", "\n"}:
        return False

    bbox = char.get("bbox")
    if bbox is None:
        return False

    x0, y0, x1, y1 = [float(v) for v in bbox]
    return x1 > x0 and y1 > y0 and _is_supported_rotation(float(char.get("rotation", 0)))


def _get_chars_for_span_fill(page_chars: list[Char] | dict[str, list[Char]]) -> list[Char]:
    """选择允许参与 span 回填的字符，保留正文内仿斜体并过滤整行斜向水印。"""
    if isinstance(page_chars, dict):
        all_chars = page_chars["chars"]
    else:
        all_chars = page_chars

    fill_char_keys = {_get_char_fill_key(char) for char in all_chars if _is_supported_rotation(float(char.get("rotation", 0)))}

    rotated_chars = [char for char in all_chars if not _is_supported_rotation(float(char.get("rotation", 0)))]
    if not rotated_chars:
        return [char for char in all_chars if _get_char_fill_key(char) in fill_char_keys]

    for line in get_lines_from_chars(all_chars):
        line_rotation = float(line.get("rotation", 0))
        if not _is_supported_rotation(line_rotation):
            continue

        line_chars = _iter_line_chars(line)
        if not any(_is_visible_standard_rotation_char(char) for char in line_chars):
            continue

        # 标准方向正文行内的局部旋转字符通常是仿斜体强调，需要允许回填。
        for char in line_chars:
            char_rotation = float(char.get("rotation", 0))
            if (
                not _is_supported_rotation(char_rotation)
                and _rotation_distance_degrees(char_rotation, line_rotation) <= SPAN_FILL_LOCAL_ROTATION_MAX_DEGREES
            ):
                fill_char_keys.add(_get_char_fill_key(char))

    return [char for char in all_chars if _get_char_fill_key(char) in fill_char_keys]


def _prepare_post_ocr_spans(
    need_ocr_spans: list[_AnalyzeSpan],
    spans: list[_AnalyzeSpan],
    pil_img: Any,
    scale: float,
) -> list[_AnalyzeSpan]:
    """为缺少原生文本的 Span 准备 OCR 裁图，并过滤低对比度无效区域。"""
    if len(need_ocr_spans) == 0:
        return spans

    for span in need_ocr_spans:
        # 对span的bbox截图再ocr
        span_pil_img = get_crop_img(span.bbox, pil_img, scale)
        span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
        # 计算span的对比度，低于0.17的span不进行ocr，等于0.17的临界框保留给后置OCR。
        if calculate_contrast(span_img, img_mode="bgr") < 0.17:
            if _restore_post_ocr_fallback(span):
                continue
            if span in spans:
                spans.remove(span)
            continue

        span.content = ""
        span.score = 1.0
        span.image = span_img

    return spans


class SpanBlockMatcher:
    """按 block 顺序消费 span，并用 y 方向索引减少无效重叠计算。"""

    def __init__(self, spans: list[_AnalyzeSpan]) -> None:
        """复制待匹配 Span，并建立按纵向位置查询的网格索引。"""
        self.spans = list(spans)
        self.used_span_indices = set()
        self.grid_size = self._get_grid_size(self.spans)
        self.grid = self._build_grid(self.spans)

    @staticmethod
    def _get_grid_size(spans: list[_AnalyzeSpan]) -> float:
        """根据 span 高度估算索引网格大小，避免过细或过粗。"""
        heights = [span.bbox[3] - span.bbox[1] for span in spans if span.bbox and span.bbox[3] > span.bbox[1]]
        if not heights:
            return 1
        return max(1, statistics.median(heights))

    def _build_grid(self, spans: list[_AnalyzeSpan]) -> dict[int, list[int]]:
        """将 span 按 y 方向网格登记，后续按 block bbox 快速取候选。"""
        grid = collections.defaultdict(list)
        for index, span in enumerate(spans):
            bbox = span.bbox
            if not bbox:
                continue
            start_cell, end_cell = self._cell_range(bbox)
            for cell_idx in range(start_cell, end_cell + 1):
                grid[cell_idx].append(index)
        return grid

    def _cell_range(self, bbox: BBox) -> tuple[int, int]:
        """计算 bbox 覆盖的 y 方向网格范围。"""
        return (int(bbox[1] / self.grid_size), int(bbox[3] / self.grid_size))

    def _candidate_indices_for_block(self, block_bbox: BBox) -> list[int]:
        """取出与 block 纵向范围可能相交的 span 原始索引。"""
        start_cell, end_cell = self._cell_range(block_bbox)
        candidate_indices = set()
        for cell_idx in range(start_cell, end_cell + 1):
            candidate_indices.update(self.grid.get(cell_idx, []))
        return sorted(candidate_indices)

    def collect_for_block(
        self,
        block_bbox: BBox,
        overlap_ratio_getter: Callable | None = None,
        threshold: float = 0.5,
    ) -> list[_AnalyzeSpan]:
        """返回当前 block 命中的 span，并标记为已消费以保持旧归属语义。"""
        if overlap_ratio_getter is None:
            overlap_ratio_getter = self._default_overlap_ratio

        block_spans = []
        for span_idx in self._candidate_indices_for_block(block_bbox):
            if span_idx in self.used_span_indices:
                continue
            span = self.spans[span_idx]
            if overlap_ratio_getter(span, block_bbox) > threshold:
                block_spans.append(span)
                self.used_span_indices.add(span_idx)
        return block_spans

    @staticmethod
    def _default_overlap_ratio(span: _AnalyzeSpan, block_bbox: BBox) -> float:
        """默认沿用旧逻辑：计算 span 面积中落入 block 的比例。"""
        return calculate_overlap_area_in_bbox1_area_ratio(span.bbox, block_bbox)


def _coerce_finite_bbox(value: Any) -> BBox | None:
    """把可迭代四元组收敛为合法有限 bbox。"""
    try:
        raw_bbox = getattr(value, "bbox", value)
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return cast(BBox, bbox)


def _coerce_finite_whitespace_bbox(value: Any) -> BBox | None:
    """解析空白字符的 loose bbox，并允许 PDF 中常见的零面积 advance point。"""
    try:
        raw_bbox = getattr(value, "bbox", value)
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox) or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return None
    return cast(BBox, bbox)


def _char_geometry_key(char: Char) -> int | None:
    """返回可用于 side-map 查询的合法 PDFium char_idx。"""
    char_idx = char.get("char_idx")
    if isinstance(char_idx, bool) or not isinstance(char_idx, int):
        return None
    return char_idx


def _span_match_rank(char_bbox: BBox, span_bbox: BBox, span_index: int) -> tuple[float, float, int]:
    """按归一化纵向中心距离、横向中心距离和原顺序稳定排序。"""
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_x = (span_bbox[0] + span_bbox[2]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_width = max(span_bbox[2] - span_bbox[0], 1e-6)
    span_height = max(span_bbox[3] - span_bbox[1], 1e-6)
    return (
        abs(char_center_y - span_center_y) / span_height,
        abs(char_center_x - span_center_x) / span_width,
        span_index,
    )


def _match_char_bbox_to_span(
    char_bbox: BBox,
    char_text: str,
    span_bboxes: list[BBox],
    grid: dict[int, list[int]],
    grid_size: float,
) -> int | None:
    """在给定 bbox 对应的全部候选 Span 中返回几何距离最优者。"""
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    candidate_span_indices = grid.get(int(char_center_y / grid_size), [])
    matches = []
    for span_index in candidate_span_indices:
        span_bbox = span_bboxes[span_index]
        if (
            char_text not in LINE_STOP_FLAG
            and char_text not in LINE_START_FLAG
            and not span_bbox[0] < char_center_x < span_bbox[2]
        ):
            continue
        if calculate_char_in_span(char_bbox, span_bbox, char_text):
            matches.append(span_index)
    if not matches:
        return None
    return min(
        matches,
        key=lambda span_index: _span_match_rank(char_bbox, span_bboxes[span_index], span_index),
    )


def _match_whitespace_bbox_to_first_span(
    char_bbox: BBox,
    char_text: str,
    span_bboxes: list[BBox],
    grid: dict[int, list[int]],
    grid_size: float,
) -> int | None:
    """按旧 loose 顺序归属无可见字形的空白，避免重叠 Span 改变词间距。"""
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    for span_index in grid.get(int(char_center_y / grid_size), []):
        span_bbox = span_bboxes[span_index]
        if (
            char_text not in LINE_STOP_FLAG
            and char_text not in LINE_START_FLAG
            and not span_bbox[0] < char_center_x < span_bbox[2]
        ):
            continue
        if calculate_char_in_span(char_bbox, span_bbox, char_text):
            return span_index
    return None


def fill_char_in_spans(
    spans: list[_AnalyzeSpan],
    all_chars: list[Char],
    median_span_height: float,
    *,
    tight_bboxes: dict[int, BBox] | None = None,
    origins: dict[int, tuple[float, float]] | None = None,
    detect_scripts: bool = True,
) -> list[_AnalyzeSpan]:
    """以 tight-first、loose-fallback 将字符分配到 Span，并返回待 OCR Span。"""
    spans = sorted(spans, key=lambda x: x.bbox[1])
    tight_bboxes = tight_bboxes or {}
    origins = origins or {}

    grid_size = max(1, median_span_height)
    grid = collections.defaultdict(list)
    span_bboxes = []
    for span_index, span in enumerate(spans):
        span_bbox = span.bbox
        span_bboxes.append(span_bbox)
        start_cell = int(span_bbox[1] / grid_size)
        end_cell = int(span_bbox[3] / grid_size)
        for cell_idx in range(start_cell, end_cell + 1):
            grid[cell_idx].append(span_index)

    assigned_span_indices: list[int | None] = [None] * len(all_chars)
    for char_position, char in enumerate(all_chars):
        char_idx = _char_geometry_key(char)
        tight_bbox = _coerce_finite_bbox(tight_bboxes.get(char_idx)) if char_idx is not None else None
        loose_bbox = _coerce_finite_bbox(char.get("bbox"))
        char_text = str(char.get("char", ""))
        if char_text.isspace():
            continue
        for candidate_bbox in (tight_bbox, loose_bbox):
            if candidate_bbox is None:
                continue
            assigned_span_indices[char_position] = _match_char_bbox_to_span(
                candidate_bbox,
                char_text,
                span_bboxes,
                grid,
                grid_size,
            )
            if assigned_span_indices[char_position] is not None:
                break

    previous_visible_owners: list[int | None] = [None] * len(all_chars)
    previous_owner = None
    for char_position, char in enumerate(all_chars):
        char_text = str(char.get("char", ""))
        if char_text in CONTROL_LINE_BREAK_CHARS:
            previous_owner = None
            continue
        previous_visible_owners[char_position] = previous_owner
        if not char_text.isspace() and assigned_span_indices[char_position] is not None:
            previous_owner = assigned_span_indices[char_position]

    next_visible_owners: list[int | None] = [None] * len(all_chars)
    next_owner = None
    for char_position in range(len(all_chars) - 1, -1, -1):
        char_text = str(all_chars[char_position].get("char", ""))
        if char_text in CONTROL_LINE_BREAK_CHARS:
            next_owner = None
            continue
        next_visible_owners[char_position] = next_owner
        if not char_text.isspace() and assigned_span_indices[char_position] is not None:
            next_owner = assigned_span_indices[char_position]

    for char_position, char in enumerate(all_chars):
        char_text = str(char.get("char", ""))
        if not char_text.isspace():
            continue
        loose_bbox = _coerce_finite_whitespace_bbox(char.get("bbox"))
        if loose_bbox is None:
            continue
        previous_owner = previous_visible_owners[char_position]
        next_owner = next_visible_owners[char_position]
        same_neighbor_owner = previous_owner is not None and previous_owner == next_owner
        neighbor_owner = previous_owner if same_neighbor_owner else None
        if neighbor_owner is None:
            neighbor_owner = previous_owner if next_owner is None else next_owner if previous_owner is None else None
        if (
            char_text not in CONTROL_LINE_BREAK_CHARS
            and neighbor_owner is not None
            and (same_neighbor_owner or calculate_char_in_span(loose_bbox, span_bboxes[neighbor_owner], char_text))
        ):
            assigned_span_indices[char_position] = neighbor_owner
        else:
            assigned_span_indices[char_position] = _match_whitespace_bbox_to_first_span(
                loose_bbox,
                char_text,
                span_bboxes,
                grid,
                grid_size,
            )

    for char, span_index in zip(all_chars, assigned_span_indices):
        if span_index is not None:
            spans[span_index].metadata["chars"].append(char)

    need_ocr_spans = []
    for span in spans:
        private_use_signal = _get_private_use_text_signal(span.metadata["chars"])
        should_post_ocr_private_use = _should_fallback_to_post_ocr_for_private_use_text(private_use_signal)
        chars_to_content(
            span,
            tight_bboxes=tight_bboxes,
            origins=origins,
            detect_scripts=detect_scripts,
        )
        # 有的span中虽然没有字但有一两个空的占位符，用宽高和content长度过滤
        if should_post_ocr_private_use and span.content:
            span.metadata[POST_OCR_FALLBACK_CONTENT_KEY] = span.content
            span.metadata[POST_OCR_FALLBACK_SCORE_KEY] = span.score
            span.metadata[POST_OCR_REASON_KEY] = POST_OCR_REASON_PRIVATE_USE_TEXT
            need_ocr_spans.append(span)
        elif len(span.content) * span.metadata["height"] < span.metadata["width"] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            need_ocr_spans.append(span)
        del span.metadata["height"], span.metadata["width"]
    return need_ocr_spans


LINE_STOP_FLAG = (
    ".",
    "!",
    "?",
    "。",
    "！",
    "？",
    ")",
    "）",
    '"',
    "”",
    ":",
    "：",
    ";",
    "；",
    "]",
    "】",
    "}",
    "}",
    ">",
    "》",
    "、",
    ",",
    "，",
    "-",
    "—",
    "–",
)
LINE_START_FLAG = (
    "(",
    "（",
    '"',
    "“",
    "【",
    "{",
    "《",
    "<",
    "「",
    "『",
    "【",
    "[",
)

Span_Height_Ratio = 0.33  # 字符的中轴和span的中轴高度差不能超过1/3span高度
SPAN_FILL_LOCAL_ROTATION_MAX_DEGREES = 30.0
CONTROL_LINE_BREAK_CHARS = {"\r", "\n"}
_ScriptRole = ScriptRole


def _is_private_use_char(char: str) -> bool:
    """判断单个字符是否落在 Unicode 私用区，用于识别字体映射异常。"""
    return len(char) == 1 and PRIVATE_USE_AREA_START <= ord(char) <= PRIVATE_USE_AREA_END


def _get_private_use_text_signal(chars: list[Char]) -> dict[str, float | int]:
    """统计 span 字符中的私用区信号，供局部后置 OCR 决策使用。"""
    pua_count = 0
    text_char_count = 0
    current_pua_run = 0
    max_pua_run = 0

    for char in chars:
        for text_char in char.get("char", ""):
            if text_char.isspace():
                current_pua_run = 0
                continue

            text_char_count += 1
            if _is_private_use_char(text_char):
                pua_count += 1
                current_pua_run += 1
                max_pua_run = max(max_pua_run, current_pua_run)
            else:
                current_pua_run = 0

    pua_ratio = 0.0
    if text_char_count > 0:
        pua_ratio = pua_count / text_char_count

    return {
        "pua_count": pua_count,
        "text_char_count": text_char_count,
        "pua_ratio": pua_ratio,
        "max_pua_run": max_pua_run,
    }


def _should_fallback_to_post_ocr_for_private_use_text(signal: dict[str, float | int]) -> bool:
    """连续或高占比 PUA 才转后置 OCR，降低孤立私用符号误召回。"""
    pua_count = signal["pua_count"]
    if pua_count < PRIVATE_USE_TEXT_COUNT_THRESHOLD:
        return False

    return signal["max_pua_run"] >= PRIVATE_USE_TEXT_RUN_THRESHOLD or signal["pua_ratio"] >= PRIVATE_USE_TEXT_RATIO_THRESHOLD


def _clear_post_ocr_fallback(span: _AnalyzeSpan) -> None:
    """清理后置 OCR 内部兜底字段，避免进入最终 middle-json 输出。"""
    span.metadata.pop(POST_OCR_FALLBACK_CONTENT_KEY, None)
    span.metadata.pop(POST_OCR_FALLBACK_SCORE_KEY, None)
    span.metadata.pop(POST_OCR_REASON_KEY, None)


def _restore_post_ocr_fallback(span: _AnalyzeSpan) -> bool:
    """在后置 OCR 无法使用时恢复原始文本兜底，返回是否已恢复。"""
    if POST_OCR_FALLBACK_CONTENT_KEY not in span.metadata:
        _clear_post_ocr_fallback(span)
        return False

    span.content = span.metadata[POST_OCR_FALLBACK_CONTENT_KEY]
    if POST_OCR_FALLBACK_SCORE_KEY in span.metadata:
        span.score = span.metadata[POST_OCR_FALLBACK_SCORE_KEY]
    _clear_post_ocr_fallback(span)
    return True


def calculate_char_in_span(
    char_bbox: BBox,
    span_bbox: BBox,
    char: str,
    span_height_ratio: float = Span_Height_Ratio,
) -> bool:
    """根据字符中心、Span 中轴和边界标点规则判断字符是否属于 Span。"""
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        and abs(char_center_y - span_center_y)
        < span_height * span_height_ratio  # 字符的中轴和span的中轴高度差不能超过Span_Height_Ratio
    ):
        return True
    # 如果char是LINE_STOP_FLAG，就不用中心点判定，换一种方案（左边界在span区域内，高度判定和之前逻辑一致）
    # 主要是给结尾符号一个进入span的机会，这个char还应该离span右边界较近
    if char in LINE_STOP_FLAG:
        if (
            (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2]
            and char_center_x > span_bbox[0]
            and span_bbox[1] < char_center_y < span_bbox[3]
            and abs(char_center_y - span_center_y) < span_height * span_height_ratio
        ):
            return True
    elif char in LINE_START_FLAG:
        if (
            span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height)
            and char_center_x < span_bbox[2]
            and span_bbox[1] < char_center_y < span_bbox[3]
            and abs(char_center_y - span_center_y) < span_height * span_height_ratio
        ):
            return True
    return False


def _get_char_bbox_metrics(char: Char) -> dict[str, float]:
    """提取字符 bbox 的宽高和中心点，统一兼容 list 与 pdftext Bbox 对象。"""
    bbox = char["bbox"]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return {
        "width": x1 - x0,
        "height": y1 - y0,
        "center_y": (y0 + y1) / 2,
    }


def _get_char_bbox_metrics_list(chars: list[Char]) -> list[dict[str, float]]:
    """预计算 span 内全部字符的 bbox 指标，避免上下标判断重复解析 bbox。"""
    return [_get_char_bbox_metrics(char) for char in chars]


def _classify_char_script_roles(
    chars: list[Char],
    *,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    protected_body_indices: set[int] | None = None,
) -> list[_ScriptRole]:
    """调用共享几何分类器识别上下标，并保留旧私有入口。"""
    return classify_char_script_roles(
        chars,
        tight_bboxes=tight_bboxes,
        origins=origins,
        protected_body_indices=protected_body_indices,
    )


def _append_script_wrapped_text(parts: list[str], role: str | None, text: str) -> None:
    """把连续同类上下标文本包裹成 HTML 标签，正文保持原样。"""
    if not text:
        return
    if role == "sup":
        parts.append(f"<sup>{text}</sup>")
    elif role == "sub":
        parts.append(f"<sub>{text}</sub>")
    else:
        parts.append(text)


def _wrap_script_runs(role_text_parts: list[tuple[str, str]]) -> str:
    """合并连续正文、上标、下标 run，避免每个字符单独生成标签。"""
    wrapped_parts: list[str] = []
    current_role = None
    current_text_parts: list[str] = []

    for role, text in role_text_parts:
        if role != current_role:
            _append_script_wrapped_text(wrapped_parts, current_role, "".join(current_text_parts))
            current_role = role
            current_text_parts = [text]
        else:
            current_text_parts.append(text)

    _append_script_wrapped_text(wrapped_parts, current_role, "".join(current_text_parts))
    return "".join(wrapped_parts)


def _axis_overlap_ratio(first_bbox: Any, second_bbox: Any, start_index: int, end_index: int) -> float:
    """计算两个字符框在指定坐标轴上相对较短边的重叠比例。"""
    try:
        first_start = float(first_bbox[start_index])
        first_end = float(first_bbox[end_index])
        second_start = float(second_bbox[start_index])
        second_end = float(second_bbox[end_index])
    except (IndexError, TypeError, ValueError):
        return 0.0
    denominator = min(first_end - first_start, second_end - second_start)
    if denominator <= 0:
        return 0.0
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    return overlap / denominator


def _compose_overlapping_spacing_diacritic(base: Char, modifier: Char) -> Char | None:
    """把与字母 bbox 重叠的 spacing diacritic 规范化为单一 NFC 字符。"""
    base_text = str(base.get("char", ""))
    modifier_text = str(modifier.get("char", ""))
    combining = _SPACING_DIACRITIC_TO_COMBINING.get(modifier_text)
    if (
        len(base_text) != 1
        or combining is None
        or not unicodedata.category(base_text).startswith("L")
        or _axis_overlap_ratio(base.get("bbox"), modifier.get("bbox"), 0, 2) < SPACING_DIACRITIC_MIN_OVERLAP_RATIO
        or _axis_overlap_ratio(base.get("bbox"), modifier.get("bbox"), 1, 3) < SPACING_DIACRITIC_MIN_OVERLAP_RATIO
    ):
        return None
    composed = unicodedata.normalize("NFC", f"{base_text}{combining}")
    if len(composed) != 1 or composed == base_text:
        return None
    merged = dict(base)
    merged["char"] = composed
    base_index = base.get("char_idx")
    modifier_index = modifier.get("char_idx")
    if isinstance(base_index, int) and isinstance(modifier_index, int):
        merged["char_idx"] = min(base_index, modifier_index)
    return cast(Char, merged)


def _merge_overlapping_spacing_diacritics_with_protection(
    chars: list[Char],
) -> tuple[list[Char], set[int]]:
    """合并 spacing diacritic，并返回必须保持正文角色的合成字符位置。"""
    merged_chars: list[Char] = []
    protected_body_indices: set[int] = set()
    cursor = 0
    while cursor < len(chars):
        current = chars[cursor]
        following = chars[cursor + 1] if cursor + 1 < len(chars) else None
        merged: Char | None = None
        if following is not None and str(current.get("char", "")) in _SPACING_DIACRITIC_TO_COMBINING:
            merged = _compose_overlapping_spacing_diacritic(following, current)
        elif following is not None and str(following.get("char", "")) in _SPACING_DIACRITIC_TO_COMBINING:
            merged = _compose_overlapping_spacing_diacritic(current, following)
        if merged is not None:
            merged_chars.append(merged)
            protected_body_indices.add(len(merged_chars) - 1)
            cursor += 2
            continue
        merged_chars.append(current)
        cursor += 1
    return merged_chars, protected_body_indices


def _merge_overlapping_spacing_diacritics(chars: list[Char]) -> list[Char]:
    """兼容调用方：只返回合并后的 spacing diacritic 字符流。"""
    merged_chars, _protected_body_indices = _merge_overlapping_spacing_diacritics_with_protection(chars)
    return merged_chars


def chars_to_content(
    span: _AnalyzeSpan,
    *,
    tight_bboxes: dict[int, BBox] | None = None,
    origins: dict[int, tuple[float, float]] | None = None,
    detect_scripts: bool = True,
) -> None:
    """将 Span 内字符重建为文本，并合并连续的上标、下标片段。"""
    span.metadata.pop(PDF_NATIVE_SCRIPT_MARKUP_KEY, None)
    # 检查span中的char是否为空
    if len(span.metadata["chars"]) != 0:
        chars = span.metadata["chars"]
        # 大多数情况下 char 已按 PDF 原始顺序进入，只有乱序时才排序。
        if any(chars[idx]["char_idx"] > chars[idx + 1]["char_idx"] for idx in range(len(chars) - 1)):
            chars = sorted(chars, key=lambda x: x["char_idx"])
        chars, protected_body_indices = _merge_overlapping_spacing_diacritics_with_protection(chars)

        char_metrics = _get_char_bbox_metrics_list(chars)
        # Calculate the width of each character
        char_widths = [metrics["width"] for metrics in char_metrics]
        # Calculate the median width
        median_width = statistics.median(char_widths)
        script_roles: list[_ScriptRole] = ["body"] * len(chars)
        if detect_scripts:
            script_roles = _classify_char_script_roles(
                chars,
                tight_bboxes=tight_bboxes or {},
                origins=origins or {},
                protected_body_indices=protected_body_indices,
            )

        role_text_parts = []
        for idx, char1 in enumerate(chars):
            if char1["char"] in CONTROL_LINE_BREAK_CHARS:
                continue
            char2 = chars[idx + 1] if idx + 1 < len(chars) else None
            role1 = script_roles[idx]
            role2 = script_roles[idx + 1] if char2 else None

            # 如果下一个char的x0和上一个char的x1距离超过0.25个字符宽度，则需要在中间插入一个空格
            role_text_parts.append((role1, char1["char"]))
            if (
                char2
                and char2["bbox"][0] - char1["bbox"][2] > median_width * 0.25
                and char1["char"] != " "
                and char2["char"] != " "
            ):
                space_role = role1 if role1 == role2 else "body"
                role_text_parts.append((space_role, " "))

        content = _wrap_script_runs(role_text_parts)
        content = __replace_unicode(content)
        content = __replace_ligatures(content)
        span.content = content.strip()
        if any(role in {"sup", "sub"} and text for role, text in role_text_parts):
            span.metadata[PDF_NATIVE_SCRIPT_MARKUP_KEY] = True

    del span.metadata["chars"]
