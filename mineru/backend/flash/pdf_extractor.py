"""Flash PDF 原生文本与表格提取器。

Flash 仅处理 PDF 原生文本；需要 OCR 时统一委托 Hybrid low。
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any, Literal, Sequence

from loguru import logger
from pdftext.schema import Char

from mineru.backend.hybrid.table_text import project_pdf_table_text
from mineru.backend.utils.char_utils import is_hyphen_at_line_end, resolve_text_line_boundary
from mineru.backend.utils.xycut_pp_sorter import sort_entries
from mineru.cli_old.common import read_fn
from mineru.types import BBox
from mineru.utils.language import detect_lang
from mineru.utils.pdf_document import PDFDocument, get_lines_from_chars


_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.?|表格?)[\s:.–—-]*(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)\b(?P<suffix>.*)$",
    re.IGNORECASE,
)
_TABLE_NOTE_RE = re.compile(
    r"^(?:notes?|sources?)\b|^(?:注释?|说明)\s*[:：]?|^for\s+[*†‡]"
    r"|^(?:\d+|[*†‡])\s+\S"
    r"|^(?:[*†‡]|[a-z]|p|t|ns|na)\s+(?:indicates?|denotes?|rainfall\b|total\b|low\b|for\b)",
    re.IGNORECASE,
)
_TABLE_SPLIT_NUMBER_RE = re.compile(
    r"^(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)[.:：]?$",
    re.IGNORECASE,
)
_FORMULA_NUMBER_SUFFIX_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<marker>[(（﹙][^()（）﹙﹚\r\n]+[)）﹚])\s*$"
)
_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MIN_RASTER_IMAGE_PAGE_AREA_RATIO = 0.005
_MIN_FORM_IMAGE_PAGE_AREA_RATIO = 0.01
_MAX_FORM_IMAGE_PAGE_AREA_RATIO = 0.8
_IMAGE_CONTAINER_OVERLAP_THRESHOLD = 0.5
_TEXT_SEMANTIC_TYPES = {
    "doc_title",
    "paragraph_title",
    "header",
    "footer",
    "page_number",
}
_OUTPUT_BLOCK_TYPES = {"text", "table", "image", "equation"} | _TEXT_SEMANTIC_TYPES
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[\-\u2013\u2014\u00b7\u2022]*\s*(?:\u7b2c\s*)?"
    r"(?P<value>\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+)"
    r"(?:\s*(?:/|of|\u5171)\s*(?:\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+))?"
    r"\s*(?:\u9875)?\s*[\-\u2013\u2014\u00b7\u2022]*\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _LineItem:
    """保存单个可视文本行及其原始几何信息。"""

    text: str
    bbox: BBox
    angle: int
    source_index: int
    chars: list[Char] = field(default_factory=list)
    visual_row_id: int | None = None
    run_index: int = 0
    effective_height: float = 0.0
    font_signature: tuple[str, int] | None = None
    font_coverage: float = 0.0
    dominant_font_weight: float | None = None
    median_glyph_width: float | None = None
    split_from_row: bool = False
    semantic_type: str | None = None


@dataclass(slots=True)
class _Fragment:
    """保存表格规则使用的单元文本片段。"""

    text: str
    bbox: BBox
    local_bbox: BBox
    line_index: int
    visual_row_id: int | None = None


@dataclass(slots=True)
class _VisualRow:
    """保存同一局部水平带内的表格片段。"""

    fragments: list[_Fragment]
    center_y: float
    bbox: BBox
    visual_row_id: int | None = None


@dataclass(slots=True)
class _AxisLine:
    """保存 PDF 路径中的横竖线。"""

    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(slots=True)
class _LocalAxisLine:
    """保存转入当前文本方向后的横竖线。"""

    bbox: BBox
    original_bbox: BBox
    orientation: Literal["horizontal", "vertical"]
    width: float


@dataclass(slots=True)
class _TableCandidate:
    """保存已通过三横线与文本分布校验的表格候选。"""

    bbox: BBox
    local_bbox: BBox
    angle: int
    score: float
    core_bbox: BBox | None = None
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _GraphicCandidate:
    """保存由紧凑绘图线组件形成的图形文本容器候选。"""

    core_bbox: BBox
    lane_index: int
    line_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class _TextLane:
    """保存同一文本方向下的局部栏带与已归属文本行。"""

    left: float
    right: float
    lines: list[tuple[_LineItem, BBox]] = field(default_factory=list)
    is_span: bool = False


@dataclass(slots=True)
class _FormulaAnchor:
    """保存公式右缘锚点及其是否落在正文密集区下方。"""

    line: _LineItem
    bbox: BBox
    detached_below_body: bool = False


@dataclass(slots=True)
class _PageSource:
    """保存单页原生文本分析所需的文本、字符、绘图线和视觉容器。"""

    page_size: tuple[float, float]
    lines: list[_LineItem]
    chars: list[Char]
    drawing_lines: list[_AxisLine]
    image_bboxes: list[BBox] = field(default_factory=list)
    form_bboxes: list[BBox] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedPage:
    """保存容器认领完成后、等待跨页与文本类型判定的轻量页面。"""

    page_size: tuple[float, float]
    remaining_lines: list[_LineItem]
    table_bboxes: list[BBox]
    drawing_lines: list[_AxisLine]
    fixed_blocks: list[dict[str, Any]]


@dataclass(slots=True)
class _MarginalCandidate:
    """保存页眉页脚带中的单行候选及其正向归一化几何。"""

    page_index: int
    line: _LineItem
    local_bbox: BBox
    local_page_size: tuple[float, float]
    region: Literal["header", "footer", "side"]


@dataclass(slots=True)
class _LaneBodyProfile:
    """保存标题判定使用的栏带正文排版基线，不包含文本内容特征。"""

    body_height: float
    body_font: tuple[str, int] | None
    body_weight: float | None
    regular_gap: float
    style_support: dict[tuple[str, int], float]


def extract_pages_text(filepath: str, start_page: int = 0, end_page: int | None = None) -> list[str]:
    """Extract plain text from each PDF page, preserving empty pages."""

    pages: list[str] = []
    with PDFDocument(filepath) as pdf_doc:
        page_count = pdf_doc.page_count
        end = page_count if end_page is None else min(end_page, page_count)

        # 保留旧 Flash parser 依赖的逐页纯文本接口。
        for page_idx in range(start_page, end):
            pages.append(pdf_doc.get_page_text(page_idx))
    return pages


def _analyze_native_document(pdf_doc: PDFDocument) -> list[list[dict[str, Any]]]:
    """逐页读取数字 PDF，并在轻量页面上完成跨页文本类型判定。"""

    prepared_pages: list[_PreparedPage] = []
    for page_idx in range(pdf_doc.page_count):
        page_size = pdf_doc.page_size(page_idx)
        chars = pdf_doc.get_page_chars(page_idx)
        lines = _build_native_line_items(
            get_lines_from_chars(chars),
            page_size,
            page_rotation=pdf_doc.page_rotation(page_idx),
        )
        drawing_lines = _get_pdf_drawing_lines(pdf_doc, page_idx)
        source = _PageSource(
            page_size=page_size,
            lines=lines,
            chars=chars,
            drawing_lines=drawing_lines,
            image_bboxes=pdf_doc.get_page_image_bboxes(page_idx),
            form_bboxes=pdf_doc.get_page_form_bboxes(page_idx),
        )
        prepared_pages.append(_prepare_page_source(source))

    _classify_repeated_page_marginals(prepared_pages)
    return [
        _finalize_prepared_page(prepared, page_index)
        for page_index, prepared in enumerate(prepared_pages)
    ]


def _build_native_line_items(
    pdf_lines: Sequence[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_rotation: int = 0,
) -> list[_LineItem]:
    """将 pdftext 粗行按字符间隙精修成 Flash 视觉 run。"""

    items: list[_LineItem] = []
    for visual_row_id, pdf_line in enumerate(pdf_lines):
        bbox = _clip_bbox(_coerce_bbox(pdf_line.get("bbox")), page_size)
        if bbox is None:
            continue
        spans = pdf_line.get("spans") or []
        chars = [char for span in spans for char in (span.get("chars") or []) if isinstance(char, dict)]
        coarse_item = _LineItem(
            text="".join(str(span.get("text") or "") for span in spans),
            bbox=bbox,
            angle=(_normalize_pdftext_angle(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360,
            source_index=-1,
            chars=chars,
            visual_row_id=visual_row_id,
        )
        items.extend(_split_native_visual_runs(coarse_item, page_size))

    merged_items = _merge_native_inline_scripts(items, page_size)
    for source_index, item in enumerate(merged_items):
        # source_index 必须在页内唯一，表格投影和失败回滚都依赖该精确成员标识。
        item.source_index = source_index
    return merged_items


def _split_native_visual_runs(
    line: _LineItem,
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """保留字符源顺序与空白信息，并将一个 pdftext 粗行拆成远距视觉 run。"""

    tokens: list[tuple[Char, str, BBox | None, BBox | None]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if raw_char in {"\r", "\n"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle) if bbox is not None else None
        tokens.append((char, raw_char, bbox, local_bbox))

    visible_indices = [
        index
        for index, (_char, raw_char, _bbox, local_bbox) in enumerate(tokens)
        if raw_char.isprintable() and not raw_char.isspace() and local_bbox is not None
    ]
    if not visible_indices:
        text = _normalize_native_run_text(line.text)
        if not text:
            return []
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        line.text = text
        line.effective_height = max(0.1, local_bbox[3] - local_bbox[1])
        return [line]

    glyph_widths = [
        max(0.1, tokens[index][3][2] - tokens[index][3][0])  # type: ignore[index]
        for index in visible_indices
    ]
    median_glyph_width = statistics.median(glyph_widths)
    local_page_width = page_size[1] if line.angle in {90, 270} else page_size[0]
    hard_gap_threshold = max(15.0, 3.0 * median_glyph_width, 0.02 * local_page_width)
    adjacent_gaps: list[float] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        if gap < hard_gap_threshold:
            adjacent_gaps.append(gap)
    # 少于三个相邻样本无法可靠代表“常规”字距；零间隙也是真实的紧排字距，
    # 必须纳入统计，避免唯一的 15pt cell gap 反过来抬高软拆阈值。
    median_regular_gap = statistics.median(adjacent_gaps) if len(adjacent_gaps) >= 3 else 0.0

    split_indices: list[int] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        has_source_whitespace = any(tokens[index][1].isspace() for index in range(previous + 1, current))
        soft_gap_threshold = max(
            8.0,
            2.2 * median_glyph_width,
            3.0 * median_regular_gap,
        )
        if gap >= hard_gap_threshold or (has_source_whitespace and gap >= soft_gap_threshold):
            split_indices.append(current)

    ranges: list[tuple[int, int]] = []
    start = 0
    for split_index in split_indices:
        ranges.append((start, split_index))
        start = split_index
    ranges.append((start, len(tokens)))

    output: list[_LineItem] = []
    for run_index, (start, end) in enumerate(ranges):
        run_tokens = tokens[start:end]
        run_text = _normalize_native_run_text("".join(token[1] for token in run_tokens))
        run_bboxes = [
            token[2]
            for token in run_tokens
            if token[2] is not None and token[1].isprintable() and not token[1].isspace()
        ]
        if not run_text or not run_bboxes:
            continue
        run_chars = [token[0] for token in run_tokens]
        run_item = _LineItem(
            text=run_text,
            bbox=_bbox_union_many(run_bboxes),
            angle=line.angle,
            source_index=-1,
            chars=run_chars,
            visual_row_id=line.visual_row_id,
            run_index=run_index,
            split_from_row=len(ranges) > 1,
        )
        _fill_native_typography(run_item, page_size)
        output.append(run_item)
    return output


def _horizontal_bbox_gap(first_bbox: BBox, second_bbox: BBox) -> float:
    """返回两个局部 bbox 在 x 轴上的无方向净空，重叠时为零。"""

    return max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)


def _normalize_native_run_text(text: str) -> str:
    """清理原生 run 文本，并把字母后的 PDF 软断词标记转换成 ASCII hyphen。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=\s*$)", "-", normalized)
    normalized = _sanitize_pdf_control_text(normalized, preserve_newlines=False)
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    return normalized.strip()


def _sanitize_pdf_control_text(text: str, *, preserve_newlines: bool) -> str:
    """删除 PDF 字体编码残留控制字符，并按调用场景决定是否保留物理换行。"""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=[\t ]*(?:\n|$))", "-", normalized)
    normalized = normalized.replace("\t", " ")
    if not preserve_newlines:
        normalized = normalized.replace("\n", "")
    return _PDF_CONTROL_CHAR_RE.sub("", normalized)


def _fill_native_typography(line: _LineItem, page_size: tuple[float, float]) -> None:
    """使用非空字符 bbox 高度和 dominant font 填充原生行排版特征。"""

    heights: list[float] = []
    glyph_widths: list[float] = []
    font_counts: dict[tuple[str, int], int] = {}
    font_weights: dict[tuple[str, int], list[float]] = {}
    valid_font_chars = 0
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if not raw_char.isprintable() or raw_char.isspace():
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        heights.append(max(0.1, local_bbox[3] - local_bbox[1]))
        glyph_widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
        font = char.get("font") or {}
        font_name = str(font.get("name") or "")
        if not font_name:
            continue
        try:
            font_flags = int(font.get("flags") or 0)
        except (TypeError, ValueError):
            font_flags = 0
        signature = (font_name, font_flags)
        font_counts[signature] = font_counts.get(signature, 0) + 1
        try:
            font_weight = float(font.get("weight"))
        except (TypeError, ValueError):
            font_weight = math.nan
        if math.isfinite(font_weight):
            font_weights.setdefault(signature, []).append(font_weight)
        valid_font_chars += 1

    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    line.effective_height = statistics.median(heights) if heights else max(0.1, local_bbox[3] - local_bbox[1])
    line.median_glyph_width = statistics.median(glyph_widths) if glyph_widths else None
    if font_counts and valid_font_chars:
        line.font_signature, dominant_count = max(font_counts.items(), key=lambda item: item[1])
        line.font_coverage = dominant_count / valid_font_chars
        dominant_weights = font_weights.get(line.font_signature, [])
        line.dominant_font_weight = statistics.median(dominant_weights) if dominant_weights else None
    else:
        line.font_signature = None
        line.font_coverage = 0.0
        line.dominant_font_weight = None


def _merge_native_inline_scripts(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """以 mutual-nearest 规则把跨粗行的小字号前后置标记合入主体视觉行。"""

    candidates: list[tuple[float, int, int, Literal["prefix", "suffix"]]] = []
    for small_index, small in enumerate(lines):
        compact_text = "".join(char for char in small.text if not char.isspace())
        if not compact_text:
            continue
        small_local_bbox = _rotate_bbox_to_upright(small.bbox, page_size, small.angle)
        for base_index, base in enumerate(lines):
            if small_index == base_index or small.angle != base.angle or small.visual_row_id == base.visual_row_id:
                continue
            if small.effective_height <= 0 or base.effective_height <= 0:
                continue
            height_ratio = small.effective_height / base.effective_height
            if not 0.35 <= height_ratio <= 0.8:
                continue
            if len(compact_text) > 8 and small_local_bbox[2] - small_local_bbox[0] > 3.0 * base.effective_height:
                continue
            base_local_bbox = _rotate_bbox_to_upright(base.bbox, page_size, base.angle)
            vertical_overlap = max(
                0.0,
                min(small_local_bbox[3], base_local_bbox[3]) - max(small_local_bbox[1], base_local_bbox[1]),
            )
            small_height = max(0.1, small_local_bbox[3] - small_local_bbox[1])
            overlap_ratio = vertical_overlap / small_height
            if overlap_ratio < 0.5:
                continue
            center_offset = abs(_bbox_center_y(small_local_bbox) - _bbox_center_y(base_local_bbox))
            if center_offset < max(0.5, 0.12 * base.effective_height):
                # 同基线居中的小字号文本更可能是表格相邻 cell，而不是上下标。
                continue
            gap_limit = max(1.5, 0.35 * base.effective_height)
            edge_options: list[tuple[float, Literal["prefix", "suffix"], float]] = []
            for position, gap in (
                ("prefix", base_local_bbox[0] - small_local_bbox[2]),
                ("suffix", small_local_bbox[0] - base_local_bbox[2]),
            ):
                if -0.35 * base.effective_height <= gap <= gap_limit:
                    edge_options.append((abs(gap), position, gap))
            if not edge_options:
                continue
            _edge_distance, position, gap = min(edge_options, key=lambda item: item[0])

            outside_offset = max(
                base_local_bbox[1] - small_local_bbox[1],
                small_local_bbox[3] - base_local_bbox[3],
            )
            tightly_attached = abs(gap) <= max(1.0, 0.1 * base.effective_height)
            if outside_offset < max(0.5, 0.08 * base.effective_height) and not tightly_attached:
                # 紧贴边缘的小字号上下标可能完全落入高字形 bbox；其余内嵌小字仍按普通 cell 排除。
                continue
            metric = abs(gap) + (1.0 - overlap_ratio) * base.effective_height
            candidates.append((metric, small_index, base_index, position))

    best_base_for_small: dict[int, tuple[float, int, Literal["prefix", "suffix"]]] = {}
    best_small_for_base: dict[tuple[int, str], tuple[float, int]] = {}
    for metric, small_index, base_index, position in candidates:
        if small_index not in best_base_for_small or metric < best_base_for_small[small_index][0]:
            best_base_for_small[small_index] = (metric, base_index, position)
        base_key = (base_index, position)
        if base_key not in best_small_for_base or metric < best_small_for_base[base_key][0]:
            best_small_for_base[base_key] = (metric, small_index)

    matches: dict[int, dict[str, int]] = {}
    for small_index, (_metric, base_index, position) in best_base_for_small.items():
        if best_small_for_base.get((base_index, position), (math.inf, -1))[1] == small_index:
            matches.setdefault(base_index, {})[position] = small_index

    consumed_small_indices = {small_index for positions in matches.values() for small_index in positions.values()}
    merged_base_indices: set[int] = set()

    def merge_children(base_index: int, visiting: set[int]) -> None:
        """先合并更小的依赖标记，再把当前完整节点递归合入更大的主体行。"""

        if base_index in merged_base_indices or base_index in visiting:
            return
        visiting.add(base_index)
        positions = matches.get(base_index, {})
        for child_index in positions.values():
            merge_children(child_index, visiting)
        base = lines[base_index]
        merged_bbox = base.bbox
        merged_chars = list(base.chars)
        if "prefix" in positions:
            prefix_index = positions["prefix"]
            prefix = lines[prefix_index]
            base.text = f"{prefix.text.strip()} {base.text.lstrip()}"
            merged_bbox = _bbox_union(merged_bbox, prefix.bbox)
            merged_chars = [*prefix.chars, *merged_chars]
            base.split_from_row = base.split_from_row or prefix.split_from_row
        if "suffix" in positions:
            suffix_index = positions["suffix"]
            suffix = lines[suffix_index]
            base.text = f"{base.text.rstrip()}{suffix.text.strip()}"
            merged_bbox = _bbox_union(merged_bbox, suffix.bbox)
            merged_chars.extend(suffix.chars)
            base.split_from_row = base.split_from_row or suffix.split_from_row
        base.bbox = merged_bbox
        base.chars = merged_chars
        _fill_native_typography(base, page_size)
        visiting.remove(base_index)
        merged_base_indices.add(base_index)

    # 从最终不会被消费的根主体开始，确保 small -> medium -> large 链不会丢失最小节点。
    root_base_indices = [base_index for base_index in matches if base_index not in consumed_small_indices]
    for base_index in root_base_indices:
        merge_children(base_index, set())
    for base_index in matches:
        merge_children(base_index, set())

    output = [line for index, line in enumerate(lines) if index not in consumed_small_indices]
    output.sort(key=lambda item: (item.visual_row_id if item.visual_row_id is not None else math.inf, item.run_index))
    return output


def _normalize_pdftext_angle(value: Any) -> int:
    """将 pdftext 弧度方向就近归一到四个标准角度。"""

    try:
        angle_radians = float(value or 0.0)
    except (TypeError, ValueError):
        return 0
    angle_degrees = math.degrees(angle_radians)
    normalized = int(round(angle_degrees / 90.0) * 90) % 360
    return normalized if normalized in {0, 90, 180, 270} else 0


def _get_pdf_drawing_lines(pdf_doc: PDFDocument, page_idx: int) -> list[_AxisLine]:
    """读取 PDFDocument 的公共绘图线结果，并隔离具体 PDFium 类型。"""

    output: list[_AxisLine] = []
    for drawing_line in pdf_doc.get_page_drawing_lines(page_idx):
        bbox = _coerce_bbox(drawing_line.bbox)
        if bbox is None:
            continue
        output.append(
            _AxisLine(
                bbox=bbox,
                width=max(0.0, float(drawing_line.width)),
                orientation=drawing_line.orientation,
            )
        )
    return output


def _prepare_page_source(source: _PageSource) -> _PreparedPage:
    """先完成表格、Form、图形和点阵图认领，留下可跨页比较的轻量文本行。"""

    form_bboxes = _select_form_image_bboxes(source)
    candidates = [
        candidate
        for candidate in _detect_table_candidates(source)
        if not any(
            _form_supersedes_nested_bbox(form_bbox, candidate.bbox)
            for form_bbox in form_bboxes
        )
    ]
    table_blocks, claimed_line_indices = _materialize_table_blocks(
        source,
        candidates,
    )
    table_bboxes = [block["bbox"] for block in table_blocks]
    active_form_bboxes = [
        form_bbox
        for form_bbox in form_bboxes
        if not any(
            _bbox_overlap_in_smaller(form_bbox, table_bbox)
            >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for table_bbox in table_bboxes
        )
    ]
    form_image_blocks, claimed_form_line_indices = _build_form_image_blocks(
        source,
        active_form_bboxes,
        claimed_line_indices,
    )
    graphic_blocks, claimed_graphic_line_indices = _build_graphic_like_blocks(
        source,
        table_bboxes + active_form_bboxes,
        claimed_line_indices | claimed_form_line_indices,
    )
    raster_image_blocks, claimed_raster_line_indices = _build_raster_image_blocks(
        source,
        table_blocks + form_image_blocks + graphic_blocks,
        claimed_line_indices | claimed_form_line_indices | claimed_graphic_line_indices,
    )
    claimed_line_indices = (
        claimed_line_indices
        | claimed_form_line_indices
        | claimed_graphic_line_indices
        | claimed_raster_line_indices
    )
    remaining_lines = _merge_same_baseline_text_lines(
        [line for line in source.lines if line.source_index not in claimed_line_indices],
        source.page_size,
        table_bboxes,
    )
    _compact_prepared_lines(remaining_lines, source.page_size)
    return _PreparedPage(
        page_size=source.page_size,
        remaining_lines=remaining_lines,
        table_bboxes=table_bboxes,
        drawing_lines=source.drawing_lines,
        fixed_blocks=table_blocks + form_image_blocks + graphic_blocks + raster_image_blocks,
    )


def _compact_prepared_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """缓存后续仍需的字符尺度并释放字符字典，限制跨页阶段内存占用。"""

    for line in lines:
        if line.median_glyph_width is None:
            line.median_glyph_width = _median_native_glyph_width(line, page_size)
        line.chars.clear()


def _finalize_prepared_page(
    prepared: _PreparedPage,
    page_index: int,
) -> list[dict[str, Any]]:
    """按边缘类型、公式、标题、正文的优先级完成单页文本并排序。"""

    marginal_lines = [line for line in prepared.remaining_lines if line.semantic_type in {"header", "footer", "page_number"}]
    formula_input = [line for line in prepared.remaining_lines if line.semantic_type is None]
    formula_blocks, remaining_lines = _build_formula_like_blocks(
        formula_input,
        prepared.table_bboxes,
        prepared.page_size,
    )
    remaining_lines = _restore_dense_split_visual_rows(
        remaining_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    _classify_page_titles(
        remaining_lines,
        prepared.page_size,
        page_index=page_index,
        container_bboxes=[block["bbox"] for block in prepared.fixed_blocks],
    )
    remaining_lines = _merge_title_resolved_visual_rows(
        remaining_lines,
        prepared.page_size,
    )
    remaining_lines = _merge_post_semantic_text_runs(
        remaining_lines,
        prepared.page_size,
        prepared.table_bboxes,
    )
    text_blocks = _build_text_blocks(
        marginal_lines + remaining_lines,
        prepared.table_bboxes,
        prepared.page_size,
        prepared.drawing_lines,
    )
    absolute_blocks = prepared.fixed_blocks + formula_blocks + text_blocks
    sorted_blocks = _sort_blocks_with_visual_row_groups(absolute_blocks, prepared.page_size)
    return [
        normalized
        for block in sorted_blocks
        if (normalized := _normalize_output_block(block, prepared.page_size)) is not None
    ]


def _analyze_page_source(source: _PageSource) -> list[dict[str, Any]]:
    """兼容单页测试入口；单页不凭边缘位置猜测页眉、页脚或页码。"""

    if not source.lines and not source.image_bboxes:
        return []
    return _finalize_prepared_page(_prepare_page_source(source), page_index=0)


def _classify_repeated_page_marginals(pages: list[_PreparedPage]) -> None:
    """仅用相邻或同奇偶页的重复证据标注页码、页眉和页脚。"""

    if len(pages) < 2:
        return
    candidates = [
        candidate
        for page_index, page in enumerate(pages)
        for line in page.remaining_lines
        if (candidate := _build_marginal_candidate(page_index, line, page.page_size)) is not None
    ]

    for left_index, left in enumerate(candidates):
        left_value = _parse_page_number_value(left.line.text)
        if left_value is None:
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            right_value = _parse_page_number_value(right.line.text)
            if (
                page_delta > 0
                and right_value is not None
                and right_value - left_value == page_delta
                and _page_number_candidates_match(left, right)
            ):
                left.line.semantic_type = "page_number"
                right.line.semantic_type = "page_number"

    for left_index, left in enumerate(candidates):
        if left.line.semantic_type == "page_number":
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and left.region != "side"
                and right.region != "side"
                and right.line.semantic_type != "page_number"
                and _marginal_geometry_matches(left, right)
                and _marginal_text_matches(left.line.text, right.line.text)
            ):
                left.line.semantic_type = left.region
                right.line.semantic_type = right.region


def _build_marginal_candidate(
    page_index: int,
    line: _LineItem,
    page_size: tuple[float, float],
) -> _MarginalCandidate | None:
    """把页面上下百分之十五内的常规小行转换成跨页比较候选。"""

    if line.semantic_type is not None:
        return None
    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    local_page_size = (page_size[1], page_size[0]) if line.angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return None
    normalized_center_y = _bbox_center_y(local_bbox) / local_page_height
    normalized_center_x = _bbox_center_x(local_bbox) / local_page_width
    if normalized_center_y <= 0.15:
        region: Literal["header", "footer", "side"] = "header"
    elif normalized_center_y >= 0.85:
        region = "footer"
    elif (
        normalized_center_y <= 0.18
        or normalized_center_y >= 0.82
        or (
            (normalized_center_x <= 0.15 or normalized_center_x >= 0.85)
            and (normalized_center_y <= 0.3 or normalized_center_y >= 0.7)
        )
    ):
        # 仅页码递增逻辑会消费 side；稳定文本不会被侧栏位置猜成页眉页脚。
        region = "side"
    else:
        return None
    if _line_effective_height(line, local_bbox) > 0.06 * local_page_height:
        return None
    return _MarginalCandidate(
        page_index=page_index,
        line=line,
        local_bbox=local_bbox,
        local_page_size=local_page_size,
        region=region,
    )


def _page_number_candidates_match(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """校验连续页码的同边缘几何，横竖版切换时允许边缘位置随版面改变。"""

    if _marginal_geometry_matches(first, second):
        return True
    first_landscape = first.local_page_size[0] > first.local_page_size[1]
    second_landscape = second.local_page_size[0] > second.local_page_size[1]
    if first_landscape == second_landscape or first.line.angle != second.line.angle:
        return False
    first_height = _line_effective_height(first.line, first.local_bbox) / first.local_page_size[1]
    second_height = _line_effective_height(second.line, second.local_bbox) / second.local_page_size[1]
    return min(first_height, second_height) > 0 and max(first_height, second_height) / min(
        first_height,
        second_height,
    ) <= 1.5


def _marginal_geometry_matches(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """比较边缘候选的方向、纵向带、字号以及同侧或镜像横向位置。"""

    if first.region != second.region or first.line.angle != second.line.angle:
        return False
    first_width, first_height = first.local_page_size
    second_width, second_height = second.local_page_size
    first_y = _bbox_center_y(first.local_bbox) / first_height
    second_y = _bbox_center_y(second.local_bbox) / second_height
    if abs(first_y - second_y) > 0.025:
        return False
    first_line_height = _line_effective_height(first.line, first.local_bbox) / first_height
    second_line_height = _line_effective_height(second.line, second.local_bbox) / second_height
    if min(first_line_height, second_line_height) <= 0 or max(first_line_height, second_line_height) / min(
        first_line_height,
        second_line_height,
    ) > 1.35:
        return False
    if (
        first.line.font_signature is not None
        and second.line.font_signature is not None
        and first.line.font_coverage >= 0.75
        and second.line.font_coverage >= 0.75
        and first.line.font_signature != second.line.font_signature
    ):
        return False

    first_normalized_bbox = (
        first.local_bbox[0] / first_width,
        first.local_bbox[1] / first_height,
        first.local_bbox[2] / first_width,
        first.local_bbox[3] / first_height,
    )
    second_normalized_bbox = (
        second.local_bbox[0] / second_width,
        second.local_bbox[1] / second_height,
        second.local_bbox[2] / second_width,
        second.local_bbox[3] / second_height,
    )
    same_side = (
        _bbox_axis_overlap_ratio(first_normalized_bbox, second_normalized_bbox, axis="x") >= 0.4
        or abs(_bbox_center_x(first_normalized_bbox) - _bbox_center_x(second_normalized_bbox)) <= 0.08
    )
    mirrored = abs(
        _bbox_center_x(first_normalized_bbox) + _bbox_center_x(second_normalized_bbox) - 1.0
    ) <= 0.12
    return same_side or mirrored


def _parse_page_number_value(text: str) -> int | None:
    """解析整行阿拉伯、罗马或中文页码；混有稳定正文的行不作为纯页码。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = _PAGE_NUMBER_RE.fullmatch(normalized)
    if match is None:
        return None
    value = match.group("value")
    if value.isdecimal():
        return int(value)
    if re.fullmatch(r"[ivxlcdm]+", value, re.IGNORECASE):
        return _roman_number_to_int(value)
    return _chinese_page_number_to_int(value)


def _roman_number_to_int(value: str) -> int | None:
    """把页码中的规范罗马数字转换成整数，非法组合返回空。"""

    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    normalized = value.upper()
    total = 0
    previous = 0
    for char in reversed(normalized):
        current = roman_values.get(char)
        if current is None:
            return None
        total += -current if current < previous else current
        previous = max(previous, current)
    if total <= 0 or total > 4999:
        return None
    return total


def _chinese_page_number_to_int(value: str) -> int | None:
    """把常见百位以内中文页码转换成整数，供跨页递增校验使用。"""

    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if all(char in digits for char in value):
        try:
            return int("".join(str(digits[char]) for char in value))
        except ValueError:
            return None
    total = 0
    current_digit = 0
    for char in value:
        if char in digits:
            current_digit = digits[char]
        elif char == "十":
            total += (current_digit or 1) * 10
            current_digit = 0
        elif char == "百":
            total += (current_digit or 1) * 100
            current_digit = 0
        else:
            return None
    return total + current_digit if total + current_digit > 0 else None


def _marginal_text_matches(first_text: str, second_text: str) -> bool:
    """在屏蔽变化数字后比较边缘稳定文本，短文本只接受完全一致。"""

    first = _normalize_marginal_text(first_text)
    second = _normalize_marginal_text(second_text)
    if not first or not second:
        return False
    if first == second:
        return True
    if min(len(first), len(second)) < 8:
        return False
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio() >= 0.92


def _normalize_marginal_text(text: str) -> str:
    """统一边缘重复文本的宽窄字符、大小写、空白和可变数字。"""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", "", normalized).strip()


def _classify_page_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
) -> None:
    """只用页面几何与字体排版标注首页文档标题和各页段落标题。"""

    for angle in sorted({line.angle for line in lines if line.semantic_type is None}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(
            _line_effective_height(line, bbox) for line, bbox in line_geometry
        )
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))

        document_title_bottom: float | None = None
        if page_index == 0:
            document_title_bottom = _classify_document_title(
                lanes,
                local_page_height,
            )
        preserve_front_matter_boundaries = _document_title_uses_page_fallback(
            lanes,
        )

        local_container_bboxes = [
            _rotate_bbox_to_upright(bbox, page_size, angle)
            for bbox in container_bboxes
        ]
        for lane in lanes:
            profile = _infer_lane_body_profile(lane)
            _classify_paragraph_titles_in_lane(
                lane,
                profile,
                local_page_height,
                local_container_bboxes,
                document_title_bottom=document_title_bottom,
                preserve_front_matter_boundaries=preserve_front_matter_boundaries,
            )


def _infer_lane_body_profile(lane: _TextLane) -> _LaneBodyProfile:
    """从栏带的长行主体估计正文行高、主字体、字重、常规行距和样式占比。"""

    available = [item for item in lane.lines if item[0].semantic_type is None]
    if not available:
        return _LaneBodyProfile(1.0, None, None, 0.35, {})
    lane_width = max(0.1, lane.right - lane.left)
    long_lines = [item for item in available if item[1][2] - item[1][0] >= 0.45 * lane_width]
    body_rows = long_lines or available
    body_line_ids = {id(line) for line, _bbox in body_rows}
    body_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in body_rows)

    font_support: dict[tuple[str, int], float] = {}
    style_support: dict[tuple[str, int], float] = {}
    total_style_width = 0.0
    for line, bbox in available:
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        line_width = max(0.1, bbox[2] - bbox[0])
        style_support[line.font_signature] = style_support.get(line.font_signature, 0.0) + line_width
        total_style_width += line_width
        if id(line) in body_line_ids and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35:
            font_support[line.font_signature] = font_support.get(line.font_signature, 0.0) + line_width
    body_font = max(font_support, key=font_support.get) if font_support else None
    if total_style_width > 0:
        style_support = {signature: width / total_style_width for signature, width in style_support.items()}

    body_weights = [
        line.dominant_font_weight
        for line, bbox in body_rows
        if line.dominant_font_weight is not None
        and (body_font is None or line.font_signature == body_font)
        and 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.35
    ]
    regular_gap, _gap_mad = _estimate_lane_gap(lane)
    return _LaneBodyProfile(
        body_height=max(0.1, body_height),
        body_font=body_font,
        body_weight=statistics.median(body_weights) if body_weights else None,
        regular_gap=regular_gap,
        style_support=style_support,
    )


def _classify_document_title(
    lanes: list[_TextLane],
    local_page_height: float,
) -> float | None:
    """从首页上部选取显著大字号锚点，并用同版式邻行扩展多行文档标题。"""

    candidates: list[tuple[float, _TextLane, int, tuple[_LineItem, BBox]]] = []
    lane_profiles = [(lane, _infer_lane_body_profile(lane)) for lane in lanes]
    column_body_heights = [profile.body_height for lane, profile in lane_profiles if not lane.is_span]
    document_body_height = (
        statistics.median(column_body_heights)
        if column_body_heights
        else min((profile.body_height for _lane, profile in lane_profiles), default=1.0)
    )
    for lane, profile in lane_profiles:
        lane_width = max(0.1, lane.right - lane.left)
        available = [item for item in lane.lines if item[0].semantic_type is None]
        for row_index, item in enumerate(available[:5]):
            line, bbox = item
            reference_height = min(profile.body_height, 1.25 * document_body_height)
            height_ratio = _line_effective_height(line, bbox) / max(0.1, reference_height)
            document_height_ratio = _line_effective_height(line, bbox) / max(
                0.1,
                document_body_height,
            )
            width_ratio = (bbox[2] - bbox[0]) / lane_width
            centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.15 * lane_width
            spans_columns_fallback = (
                lane.is_span
                and centered
                and width_ratio >= 0.65
                and document_height_ratio >= 1.3
            )
            if (
                _bbox_center_y(bbox) > 0.45 * local_page_height
                or (height_ratio < 1.4 and not spans_columns_fallback)
                or width_ratio < 0.2
                or (not centered and height_ratio < 1.7)
            ):
                continue
            top_preference = max(0.0, 0.45 - _bbox_center_y(bbox) / local_page_height)
            score = height_ratio + (0.75 if centered else 0.0) + top_preference - 0.05 * row_index
            candidates.append((score, lane, row_index, item))
    if not candidates:
        return None

    _score, lane, _row_index, anchor = max(candidates, key=lambda item: item[0])
    anchor_line, anchor_bbox = anchor
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    anchor_line.semantic_type = "doc_title"
    selected = [anchor]
    ordered = lane.lines
    anchor_index = ordered.index(anchor)
    for direction in (-1, 1):
        index = anchor_index + direction
        previous_bbox = anchor_bbox
        while 0 <= index < len(ordered):
            candidate_line, candidate_bbox = ordered[index]
            if candidate_line.semantic_type is not None:
                break
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.8 <= candidate_height / anchor_height <= 1.25:
                break
            if not _title_fonts_compatible(anchor_line, candidate_line):
                break
            vertical_gap = max(candidate_bbox[1] - previous_bbox[3], previous_bbox[1] - candidate_bbox[3], 0.0)
            lane_width = max(0.1, lane.right - lane.left)
            centered = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(anchor_bbox)) <= 0.18 * lane_width
            aligned = abs(candidate_bbox[0] - anchor_bbox[0]) <= 0.75 * anchor_height
            if vertical_gap > 1.1 * anchor_height or not (centered or aligned):
                break
            candidate_line.semantic_type = "doc_title"
            selected.append((candidate_line, candidate_bbox))
            previous_bbox = candidate_bbox
            index += direction
    return max(bbox[3] for _line, bbox in selected)


def _document_title_uses_page_fallback(lanes: list[_TextLane]) -> bool:
    """判断首页标题是否依赖跨栏 1.30 倍全文正文行高兜底。"""

    title_heights = [
        _line_effective_height(line, bbox)
        for lane in lanes
        for line, bbox in lane.lines
        if line.semantic_type == "doc_title"
    ]
    column_body_heights = [
        _infer_lane_body_profile(lane).body_height
        for lane in lanes
        if not lane.is_span
    ]
    if not title_heights or not column_body_heights:
        return False
    document_body_height = statistics.median(column_body_heights)
    return any(
        1.3 <= title_height / max(0.1, document_body_height) < 1.4
        for title_height in title_heights
    )


def _classify_paragraph_titles_in_lane(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    preserve_front_matter_boundaries: bool,
) -> None:
    """以字号、样式、留白、对齐、栏宽和容器邻接判定段落标题。"""

    lane_width = max(0.1, lane.right - lane.left)
    rows = lane.lines
    selected_indices: set[int] = set()
    front_matter_boundary = _infer_front_matter_boundary(
        lane,
        profile,
        local_page_height,
        document_title_bottom=document_title_bottom,
    )
    for index, (line, bbox) in enumerate(rows):
        if line.semantic_type is not None:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        inside_front_matter = (
            front_matter_boundary is not None
            and _bbox_center_y(bbox) <= front_matter_boundary
        )
        if inside_front_matter and not preserve_front_matter_boundaries:
            continue
        if _line_near_visual_container(bbox, container_bboxes, profile.body_height):
            continue

        line_height = _line_effective_height(line, bbox)
        height_ratio = line_height / profile.body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.12 * lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        style_differs = (
            profile.body_font is not None
            and line.font_signature is not None
            and line.font_coverage >= 0.75
            and line.font_signature != profile.body_font
        )
        style_support = profile.style_support.get(line.font_signature, 1.0) if line.font_signature is not None else 1.0
        weight_emphasized = (
            profile.body_weight is not None
            and line.dominant_font_weight is not None
            and line.dominant_font_weight >= max(profile.body_weight + 100.0, 1.15 * profile.body_weight)
        )
        gap_above = _normalized_title_gap(rows, index, direction=-1, body_height=profile.body_height)
        gap_below = _normalized_title_gap(rows, index, direction=1, body_height=profile.body_height)
        has_spacing_signal = gap_above >= 0.35

        if _visual_row_has_body_style_sibling(rows, index):
            continue
        if _is_near_full_mixed_inline_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        if (
            index > 0
            and rows[index - 1][0].semantic_type == "paragraph_title"
            and width_ratio >= 0.85
            and gap_above < 1.0
            and not _title_fonts_compatible(rows[index - 1][0], line)
        ):
            continue
        if height_ratio < 0.9 and not inside_front_matter and not _has_following_body_row(
            rows,
            index,
            lane_width,
            profile,
        ):
            continue
        if width_ratio >= 0.9 and height_ratio < 1.15 and not (
            style_differs and gap_above >= 0.65
        ):
            continue
        score = 0.0
        if style_differs:
            score += 2.0
            if style_support <= 0.2:
                score += 0.75
        if weight_emphasized:
            score += 1.0
        if height_ratio >= 1.18:
            score += 2.0
        elif height_ratio <= 0.9 and centered:
            score += 0.75
        if width_ratio <= 0.78:
            score += 0.75
        if centered and width_ratio <= 0.85:
            score += 1.25
        if gap_above >= 0.65:
            score += 1.25
        elif gap_above >= 0.35:
            score += 0.5
        if gap_below >= 0.35:
            score += 0.75
        if left_aligned:
            score += 0.25

        strong_layout_signal = (
            height_ratio >= 1.18
            or style_differs
            or weight_emphasized
            or (centered and width_ratio <= 0.7 and gap_above >= 0.35 and gap_below >= 0.2)
        )
        if score >= 4.0 and has_spacing_signal and strong_layout_signal:
            if _is_full_width_inline_heading(
                rows,
                index,
                lane_width,
                profile,
            ):
                continue
            line.semantic_type = "paragraph_title"
            selected_indices.add(index)

    _unify_visual_row_title_types(
        lane,
        selected_indices,
    )
    _expand_paragraph_title_neighbors(
        lane,
        selected_indices,
        profile,
    )
    if front_matter_boundary is not None and preserve_front_matter_boundaries:
        _protect_front_matter_title_types(
            lane,
            profile,
            front_matter_boundary,
        )


def _protect_front_matter_title_types(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    front_matter_boundary: float,
) -> None:
    """把作者区误命中的标题降为正文，并仅保留原本需要跨行聚合的内部标记。"""

    front_title_indices = [
        index
        for index, (line, bbox) in enumerate(lane.lines)
        if line.semantic_type == "paragraph_title"
        and _bbox_center_y(bbox) <= front_matter_boundary
    ]
    connected_indices: set[int] = set()
    for position, first_index in enumerate(front_title_indices):
        for second_index in front_title_indices[position + 1 :]:
            if _should_connect_semantic_rows(
                lane.lines[first_index],
                lane.lines[second_index],
                lane,
                profile.regular_gap,
                [],
                [],
            ):
                connected_indices.update((first_index, second_index))
    for index in front_title_indices:
        lane.lines[index][0].semantic_type = (
            "text" if index in connected_indices else None
        )


def _visual_row_has_body_style_sibling(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
) -> bool:
    """检查同一完整视觉行是否同时包含标题样式与正文样式 run。"""

    line, _bbox = rows[index]
    if line.visual_row_id is None:
        return False
    siblings = [
        sibling
        for sibling, _sibling_bbox in rows
        if sibling is not line and sibling.visual_row_id == line.visual_row_id
    ]
    return any(not _title_fonts_compatible(line, sibling) for sibling in siblings)


def _is_near_full_mixed_inline_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别近满栏混合字体行内强调，避免把粗体条目头单独标成标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        line.font_coverage >= 0.95
        or (bbox[2] - bbox[0]) < 0.85 * lane_width
        or not 0.85 <= line_height / profile.body_height <= 1.15
        or index + 1 >= len(rows)
    ):
        return False
    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    gap_below = _effective_text_row_gap(rows[index], rows[index + 1])
    return (
        next_line.semantic_type is None
        and 0.8 <= next_height / profile.body_height <= 1.2
        and gap_below <= profile.regular_gap + 0.25 * profile.body_height
    )


def _is_full_width_inline_heading(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """识别后接常规正文续行的满栏正常字号行内标题。"""

    line, bbox = rows[index]
    line_height = _line_effective_height(line, bbox)
    if (
        not 0.85 <= line_height / profile.body_height <= 1.15
        or (bbox[2] - bbox[0]) < 0.9 * lane_width
        or index + 1 >= len(rows)
    ):
        return False

    next_line, next_bbox = rows[index + 1]
    next_height = _line_effective_height(next_line, next_bbox)
    next_uses_body_font = (
        profile.body_font is None
        or next_line.font_signature is None
        or next_line.font_coverage < 0.75
        or next_line.font_signature == profile.body_font
    )
    return (
        next_line.semantic_type is None
        and next_uses_body_font
        and 0.75 <= next_height / profile.body_height <= 1.15
        and (next_bbox[2] - next_bbox[0]) >= 0.75 * lane_width
        and _effective_text_row_gap(rows[index], rows[index + 1])
        <= profile.regular_gap + 0.25 * profile.body_height
    )


def _has_following_body_row(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    lane_width: float,
    profile: _LaneBodyProfile,
) -> bool:
    """检查小字号候选下方是否仍有正常字号的正文续行。"""

    for line, bbox in rows[index + 1 :]:
        height_ratio = _line_effective_height(line, bbox) / profile.body_height
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        if 0.9 <= height_ratio <= 1.2 and width_ratio >= 0.45:
            return True
    return False


def _unify_visual_row_title_types(
    lane: _TextLane,
    selected_indices: set[int],
) -> None:
    """按完整 visual_row_id 统一标题类型：同样式整行晋升，混合样式整行降级。"""

    row_indices: dict[int, list[int]] = {}
    for index, (line, _bbox) in enumerate(lane.lines):
        if line.visual_row_id is not None:
            row_indices.setdefault(line.visual_row_id, []).append(index)
    for indices in row_indices.values():
        title_indices = [
            index
            for index in indices
            if lane.lines[index][0].semantic_type == "paragraph_title"
        ]
        if not title_indices:
            continue
        anchor_line, anchor_bbox = lane.lines[title_indices[0]]
        anchor_height = _line_effective_height(anchor_line, anchor_bbox)
        compatible = all(
            0.75
            <= _line_effective_height(lane.lines[index][0], lane.lines[index][1])
            / anchor_height
            <= 1.25
            and _title_fonts_compatible(anchor_line, lane.lines[index][0])
            for index in indices
        )
        if compatible:
            for index in indices:
                lane.lines[index][0].semantic_type = "paragraph_title"
                selected_indices.add(index)
            continue
        for index in title_indices:
            lane.lines[index][0].semantic_type = None
            selected_indices.discard(index)


def _infer_front_matter_boundary(
    lane: _TextLane,
    profile: _LaneBodyProfile,
    local_page_height: float,
    *,
    document_title_bottom: float | None,
) -> float | None:
    """用短行后衔接满栏正文的几何转折点界定首页作者等前置信息。"""

    if document_title_bottom is None:
        return None
    default_boundary = max(
        document_title_bottom + 2.5 * profile.body_height,
        0.39 * local_page_height,
    )
    lane_width = max(0.1, lane.right - lane.left)
    for index, (line, bbox) in enumerate(lane.lines[:-1]):
        if bbox[1] <= document_title_bottom:
            continue
        next_line, next_bbox = lane.lines[index + 1]
        width_ratio = (bbox[2] - bbox[0]) / lane_width
        next_width_ratio = (next_bbox[2] - next_bbox[0]) / lane_width
        left_aligned = abs(bbox[0] - lane.left) <= 0.65 * profile.body_height
        centered = abs(
            _bbox_center_x(bbox) - (lane.left + lane.right) / 2.0
        ) <= 0.12 * lane_width
        next_height_ratio = _line_effective_height(next_line, next_bbox) / profile.body_height
        gap_below = max(0.0, _effective_text_row_gap((line, bbox), (next_line, next_bbox)))
        if (
            width_ratio <= 0.35
            and next_width_ratio >= 0.75
            and (left_aligned or centered)
            and 0.75 <= next_height_ratio <= 1.35
            and gap_below <= 1.5 * profile.body_height
        ):
            return min(default_boundary, bbox[1] - 0.1)
    return default_boundary


def _normalized_title_gap(
    rows: list[tuple[_LineItem, BBox]],
    index: int,
    *,
    direction: Literal[-1, 1],
    body_height: float,
) -> float:
    """返回标题候选与相邻物理行之间按正文行高归一化的有效净空。"""

    neighbor_index = index + direction
    if not 0 <= neighbor_index < len(rows):
        return 0.5
    if direction < 0:
        gap = _effective_text_row_gap(rows[neighbor_index], rows[index])
    else:
        gap = _effective_text_row_gap(rows[index], rows[neighbor_index])
    return max(0.0, gap) / max(0.1, body_height)


def _line_near_visual_container(
    line_bbox: BBox,
    container_bboxes: list[BBox],
    body_height: float,
) -> bool:
    """检查短行是否紧邻图、表等容器，以纯几何方式抑制 caption 误判。"""

    for container_bbox in container_bboxes:
        if _bbox_axis_overlap_ratio(line_bbox, container_bbox, axis="x") < 0.35:
            continue
        vertical_gap = max(line_bbox[1] - container_bbox[3], container_bbox[1] - line_bbox[3], 0.0)
        if vertical_gap <= 1.5 * body_height:
            return True
    return False


def _title_fonts_compatible(first: _LineItem, second: _LineItem) -> bool:
    """检查标题字体和字重是否兼容；低字体覆盖率时仍保留可靠字重证据。"""

    font_conflicts = (
        first.font_signature is not None
        and second.font_signature is not None
        and first.font_coverage >= 0.75
        and second.font_coverage >= 0.75
        and first.font_signature != second.font_signature
    )
    weight_conflicts = (
        first.dominant_font_weight is not None
        and second.dominant_font_weight is not None
        and abs(first.dominant_font_weight - second.dominant_font_weight) >= 100.0
        and max(first.dominant_font_weight, second.dominant_font_weight)
        >= 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )
    return not (font_conflicts or weight_conflicts)


def _expand_paragraph_title_neighbors(
    lane: _TextLane,
    selected_indices: set[int],
    profile: _LaneBodyProfile,
) -> None:
    """用相邻行的字体、尺寸、对齐和紧凑净空补齐折行段落标题。"""

    if not selected_indices:
        return
    rows = lane.lines
    lane_width = max(0.1, lane.right - lane.left)
    pending = list(selected_indices)
    while pending:
        selected_index = pending.pop()
        selected_line, selected_bbox = rows[selected_index]
        selected_height = _line_effective_height(selected_line, selected_bbox)
        for candidate_index in (selected_index - 1, selected_index + 1):
            if not 0 <= candidate_index < len(rows) or candidate_index in selected_indices:
                continue
            candidate_line, candidate_bbox = rows[candidate_index]
            if candidate_line.semantic_type is not None:
                continue
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if not 0.75 <= candidate_height / selected_height <= 1.25:
                continue
            if candidate_bbox[2] - candidate_bbox[0] > 0.8 * lane_width:
                continue
            if not _title_fonts_compatible(selected_line, candidate_line):
                continue
            vertical_gap = max(
                candidate_bbox[1] - selected_bbox[3],
                selected_bbox[1] - candidate_bbox[3],
                0.0,
            )
            centers_align = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(selected_bbox)) <= 0.15 * lane_width
            left_edges_align = abs(candidate_bbox[0] - selected_bbox[0]) <= 0.75 * profile.body_height
            wrapped_indent = abs(candidate_bbox[0] - selected_bbox[0]) <= 2.0 * profile.body_height
            if vertical_gap > 0.65 * profile.body_height or not (
                centers_align or left_edges_align or wrapped_indent
            ):
                continue
            candidate_line.semantic_type = "paragraph_title"
            selected_indices.add(candidate_index)
            pending.append(candidate_index)


def _detect_table_candidates(source: _PageSource) -> list[_TableCandidate]:
    """按文本方向融合三条长横线与规则文本分布，生成表格候选。"""

    candidates: list[_TableCandidate] = []
    angles = sorted({line.angle for line in source.lines})
    for angle in angles:
        angle_lines = [line for line in source.lines if line.angle == angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(angle_lines, source.page_size)
        if not fragments:
            continue
        median_height = _median_fragment_height(fragments)
        rows = _cluster_fragment_rows(fragments, median_height)
        local_axis_lines = _transform_axis_lines(
            source.drawing_lines,
            source.page_size,
            angle,
        )
        candidates.extend(
            _build_rule_table_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
            )
        )
    return _merge_table_candidates(candidates)


def _build_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_Fragment]:
    """将精修后的原生 run 转换成表格单元候选。"""

    fragments: list[_Fragment] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        fragments.append(
            _Fragment(
                text=line.text,
                bbox=line.bbox,
                local_bbox=local_bbox,
                line_index=line.source_index,
                # 复用原生粗行身份，避免同一字符行内不同 cell
                # 因轻微基线差异被误拆成多行。
                visual_row_id=line.visual_row_id,
            )
        )
    return fragments


def _cluster_fragment_rows(
    fragments: list[_Fragment],
    median_height: float,
) -> list[_VisualRow]:
    """优先复用原生视觉行身份，其余片段按中心线容差聚成表格行。"""

    tolerance = max(2.0, median_height * 0.5)
    native_groups: dict[int, list[_Fragment]] = {}
    geometric_fragments: list[_Fragment] = []
    for fragment in fragments:
        if fragment.visual_row_id is None:
            geometric_fragments.append(fragment)
        else:
            native_groups.setdefault(fragment.visual_row_id, []).append(fragment)

    # 先锁定同一原生粗行拆出的 run，再允许不同粗行按基线几何合并；
    # 旋转表格常把同一数据行的各 cell 分成多个 pdftext 粗行，不能只依赖 row id。
    seed_groups = [*native_groups.values(), *[[fragment] for fragment in geometric_fragments]]
    seed_groups.sort(
        key=lambda group: (
            statistics.fmean(_bbox_center_y(item.local_bbox) for item in group),
            min(item.local_bbox[0] for item in group),
        )
    )
    grouped: list[list[_Fragment]] = []
    for seed_group in seed_groups:
        center_y = statistics.fmean(_bbox_center_y(item.local_bbox) for item in seed_group)
        target_group: list[_Fragment] | None = None
        for group in grouped:
            group_center = statistics.fmean(_bbox_center_y(item.local_bbox) for item in group)
            if abs(center_y - group_center) <= tolerance:
                target_group = group
                break
        if target_group is None:
            grouped.append(list(seed_group))
        else:
            target_group.extend(seed_group)

    rows: list[_VisualRow] = []
    for group in grouped:
        group.sort(key=lambda item: item.local_bbox[0])
        bbox = _bbox_union_many([item.local_bbox for item in group])
        visual_row_ids = {item.visual_row_id for item in group if item.visual_row_id is not None}
        rows.append(
            _VisualRow(
                fragments=group,
                center_y=sum(_bbox_center_y(item.local_bbox) for item in group) / len(group),
                bbox=bbox,
                visual_row_id=next(iter(visual_row_ids)) if len(visual_row_ids) == 1 else None,
            )
        )
    rows.sort(key=lambda row: row.center_y)
    return rows


def _build_rule_table_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
) -> list[_TableCandidate]:
    """用三条长横线确定区域，再以连续多列文本分布确认表格。"""

    candidates: list[_TableCandidate] = []
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
        core_rows: list[_VisualRow] = []
        for row in rows:
            clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=0.0)
            if clipped_row is not None and rule_bbox[1] <= clipped_row.center_y <= rule_bbox[3]:
                core_rows.append(clipped_row)

        dense_rows = _longest_dense_multi_cell_rows(core_rows, median_height)
        if len(dense_rows) < 3:
            continue
        stable_columns, column_coverage = _count_stable_columns(dense_rows, median_height)
        if stable_columns < 2 or column_coverage < 0.5:
            continue

        caption_line = _find_table_caption(lines, rule_bbox, page_size, angle, median_height)
        candidate = _expand_rule_table_candidate(
            rule_group,
            core_rows,
            rows,
            page_size,
            angle,
            median_height,
            caption_line,
        )
        candidate.score = float(len(rule_group) + len(dense_rows) + stable_columns)
        candidates.append(candidate)
    return candidates


def _group_long_horizontal_rules(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """按近似左右端点和纵向距离聚合长横线，并去除同位置重复路径。"""

    minimum_length = max(40.0, 10.0 * median_height)
    horizontal_lines = [
        line for line in axis_lines if line.orientation == "horizontal" and line.bbox[2] - line.bbox[0] >= minimum_length
    ]
    endpoint_tolerance = max(4.0, 2.0 * median_height)
    span_groups: list[list[_LocalAxisLine]] = []
    for line in sorted(horizontal_lines, key=lambda item: (item.bbox[0], item.bbox[2], item.bbox[1])):
        target = next(
            (
                group
                for group in span_groups
                if abs(line.bbox[0] - group[0].bbox[0]) <= endpoint_tolerance
                and abs(line.bbox[2] - group[0].bbox[2]) <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            span_groups.append([line])
        else:
            target.append(line)

    output: list[list[_LocalAxisLine]] = []
    maximum_vertical_gap = 24.0 * median_height
    for span_group in span_groups:
        vertical_groups: list[list[_LocalAxisLine]] = []
        for line in sorted(span_group, key=lambda item: _bbox_center_y(item.bbox)):
            if (
                not vertical_groups
                or _bbox_center_y(line.bbox) - _bbox_center_y(vertical_groups[-1][-1].bbox) > maximum_vertical_gap
            ):
                vertical_groups.append([line])
            else:
                vertical_groups[-1].append(line)

        for vertical_group in vertical_groups:
            unique_lines: list[_LocalAxisLine] = []
            for line in vertical_group:
                if any(abs(_bbox_center_y(line.bbox) - _bbox_center_y(item.bbox)) <= 1.0 for item in unique_lines):
                    continue
                unique_lines.append(line)
            if len(unique_lines) >= 3:
                output.append(unique_lines)
    return output


def _clip_visual_row_to_corridor(
    row: _VisualRow,
    corridor_bbox: BBox,
    *,
    margin: float,
) -> _VisualRow | None:
    """仅保留横向走廊内的片段，避免同基线的另一栏文本污染表格区域。"""

    fragments = [
        fragment
        for fragment in row.fragments
        if corridor_bbox[0] - margin <= _bbox_center_x(fragment.local_bbox) <= corridor_bbox[2] + margin
    ]
    if not fragments:
        return None
    fragments.sort(key=lambda fragment: fragment.local_bbox[0])
    return _VisualRow(
        fragments=fragments,
        center_y=sum(_bbox_center_y(fragment.local_bbox) for fragment in fragments) / len(fragments),
        bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
        visual_row_id=row.visual_row_id,
    )


def _longest_dense_multi_cell_rows(
    rows: list[_VisualRow],
    median_height: float,
) -> list[_VisualRow]:
    """返回行距不超过四倍行高的最长连续多单元格文本段。"""

    segments: list[list[_VisualRow]] = []
    for row in (item for item in rows if len(item.fragments) >= 2):
        if not segments or row.center_y - segments[-1][-1].center_y > 4.0 * median_height:
            segments.append([row])
        else:
            segments[-1].append(row)
    return max(segments, key=len, default=[])


def _expand_rule_table_candidate(
    rule_group: list[_LocalAxisLine],
    core_rows: list[_VisualRow],
    all_rows: list[_VisualRow],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    caption_line: _LineItem | None,
) -> _TableCandidate:
    """合并横线核心、上方标题和下方脚注，构造统一投影候选。"""

    rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
    core_line_indices = {fragment.line_index for row in core_rows for fragment in row.fragments}
    caption_rows = _collect_caption_rows(all_rows, caption_line, rule_bbox, median_height)
    footnote_rows = _collect_footnote_rows(
        all_rows,
        rule_bbox,
        median_height,
        core_line_indices,
    )
    included_rows = [*caption_rows, *core_rows, *footnote_rows]
    core_local_bbox = _bbox_union(rule_bbox, _bbox_union_many([row.bbox for row in core_rows]))
    local_bbox = _bbox_union(core_local_bbox, _bbox_union_many([row.bbox for row in included_rows]))
    return _TableCandidate(
        bbox=_rotate_bbox_from_upright(local_bbox, page_size, angle),
        local_bbox=local_bbox,
        angle=angle,
        score=0.0,
        core_bbox=_rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        line_indices={fragment.line_index for row in included_rows for fragment in row.fragments},
    )


def _collect_caption_rows(
    rows: list[_VisualRow],
    caption_line: _LineItem | None,
    rule_bbox: BBox,
    median_height: float,
) -> list[_VisualRow]:
    """收集显式标题所在行及其到表格上边界之间的连续换行。"""

    if caption_line is None:
        return []
    caption_row_index = next(
        (
            index
            for index, row in enumerate(rows)
            if any(fragment.line_index == caption_line.source_index for fragment in row.fragments)
        ),
        None,
    )
    if caption_row_index is None:
        return []

    output: list[_VisualRow] = []
    previous_bbox: BBox | None = None
    margin = 2.0 * median_height
    for index, row in enumerate(rows[caption_row_index:], start=caption_row_index):
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=margin)
        if clipped_row is None:
            continue
        if index > caption_row_index and clipped_row.center_y >= rule_bbox[1]:
            break
        if previous_bbox is not None and max(0.0, clipped_row.bbox[1] - previous_bbox[3]) > 2.0 * median_height:
            break
        output.append(clipped_row)
        previous_bbox = clipped_row.bbox

    if not output or rule_bbox[1] - output[-1].bbox[3] > 2.0 * median_height:
        return []
    return output


def _collect_footnote_rows(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
    core_line_indices: set[int],
) -> list[_VisualRow]:
    """从表格下边界开始吸收带通用标记的脚注及其连续换行。"""

    output: list[_VisualRow] = []
    bottom = rule_bbox[3]
    note_chain_started = False
    margin = 2.0 * median_height
    selected_line_indices = set(core_line_indices)
    for row in rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=margin)
        if clipped_row is None or clipped_row.bbox[3] <= bottom:
            continue
        line_indices = {fragment.line_index for fragment in clipped_row.fragments}
        if line_indices.issubset(selected_line_indices):
            bottom = max(bottom, clipped_row.bbox[3])
            continue
        if max(0.0, clipped_row.bbox[1] - bottom) > 1.5 * median_height:
            break
        if not note_chain_started and not _is_table_note_text(_visual_row_text(clipped_row)):
            break
        output.append(clipped_row)
        selected_line_indices.update(line_indices)
        bottom = max(bottom, clipped_row.bbox[3])
        note_chain_started = True
    return output


def _visual_row_text(row: _VisualRow) -> str:
    """按局部 x 顺序拼接视觉行文本，供拆分脚注标记判断。"""

    return " ".join(fragment.text.strip() for fragment in row.fragments if fragment.text.strip())


def _is_table_note_text(text: str) -> bool:
    """判断表后首行是否具有明确的注释、来源或脚注标记。"""

    return bool(_TABLE_NOTE_RE.match(str(text or "").strip()))


def _find_table_caption(
    lines: list[_LineItem],
    core_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> _LineItem | None:
    """在核心表格上方最多十二倍行高内查找显式 Table/表标题。"""

    candidates: list[tuple[float, _LineItem]] = []
    for line in lines:
        text = line.text.strip()
        caption_match = _TABLE_CAPTION_RE.match(text)
        is_split_label = text.lower().rstrip(".") in {"table", "tab", "表", "表格"}
        if caption_match is None and not is_split_label:
            continue
        if caption_match is not None:
            suffix = caption_match.group("suffix").strip(" .:–—-")
            # 小写连续句通常是“Table 5 also ...”这类正文，不应作为标题。
            if suffix and suffix[0].islower():
                continue
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        if _bbox_axis_overlap_ratio(local_bbox, core_bbox, axis="x") < 0.05:
            continue
        if is_split_label:
            has_number_peer = bool(
                _find_caption_number_peers(
                    line,
                    lines,
                    page_size,
                    angle,
                    median_height,
                )
            )
            if not has_number_peer:
                continue
        gap = core_bbox[1] - local_bbox[3]
        if -median_height <= gap <= 12.0 * median_height:
            candidates.append((abs(gap), line))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _find_caption_number_peers(
    caption_line: _LineItem,
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
) -> list[_LineItem]:
    """查找与拆分 Table/表 标签同一视觉行的编号文本。"""

    caption_local_bbox = _rotate_bbox_to_upright(caption_line.bbox, page_size, angle)
    peers: list[_LineItem] = []
    for peer in lines:
        if peer.source_index == caption_line.source_index:
            continue
        if not _TABLE_SPLIT_NUMBER_RE.match(peer.text.strip()):
            continue
        peer_local_bbox = _rotate_bbox_to_upright(peer.bbox, page_size, angle)
        gap = peer_local_bbox[0] - caption_local_bbox[2]
        if _bbox_axis_overlap_ratio(caption_local_bbox, peer_local_bbox, axis="y") >= 0.5 and 0.0 <= gap <= 4.0 * median_height:
            peers.append(peer)
    return sorted(
        peers,
        key=lambda peer: _rotate_bbox_to_upright(peer.bbox, page_size, angle)[0],
    )


def _count_stable_columns(
    rows: list[_VisualRow],
    median_height: float,
) -> tuple[int, float]:
    """分别聚类片段左边界、中心和右边界，返回最稳定的列分布。"""

    tolerance = max(3.0, median_height * 0.75)
    best_result = (0, 0.0)
    # 三种对齐方式分别聚类，避免把同一片段的不同锚点混算为多列。
    for alignment in ("left", "center", "right"):
        clusters: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            for fragment in row.fragments:
                left, _top, right, _bottom = fragment.local_bbox
                if alignment == "left":
                    anchor = left
                elif alignment == "center":
                    anchor = (left + right) / 2
                else:
                    anchor = right
                cluster = next(
                    (item for item in clusters if abs(anchor - float(item["mean"])) <= tolerance),
                    None,
                )
                if cluster is None:
                    clusters.append({"mean": anchor, "values": [anchor], "rows": {row_index}})
                else:
                    cluster["values"].append(anchor)
                    cluster["rows"].add(row_index)
                    cluster["mean"] = sum(cluster["values"]) / len(cluster["values"])
        stable_coverages = [
            len(cluster["rows"]) / len(rows)
            for cluster in clusters
            if len(cluster["rows"]) / len(rows) >= 0.5
        ]
        result = (
            len(stable_coverages),
            min(stable_coverages) if stable_coverages else 0.0,
        )
        # 仅在结果严格更优时更新，平局时保留既有的左对齐优先级。
        if result > best_result:
            best_result = result
    return best_result


def _transform_axis_lines(
    lines: list[_AxisLine],
    page_size: tuple[float, float],
    angle: int,
) -> list[_LocalAxisLine]:
    """将原页面横竖线转入当前文本方向的局部坐标。"""

    output: list[_LocalAxisLine] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, angle)
        orientation: Literal["horizontal", "vertical"] = (
            "horizontal" if local_bbox[2] - local_bbox[0] >= local_bbox[3] - local_bbox[1] else "vertical"
        )
        output.append(
            _LocalAxisLine(
                bbox=local_bbox,
                original_bbox=line.bbox,
                orientation=orientation,
                width=line.width,
            )
        )
    return output


def _merge_table_candidates(candidates: list[_TableCandidate]) -> list[_TableCandidate]:
    """合并同方向且明显重叠的横线候选，避免同一表格重复输出。"""

    merged: list[_TableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        target = next(
            (
                item
                for item in merged
                if item.angle == candidate.angle and _bbox_overlap_in_smaller(candidate.bbox, item.bbox) >= 0.2
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        target.bbox = _bbox_union(target.bbox, candidate.bbox)
        target.local_bbox = _bbox_union(target.local_bbox, candidate.local_bbox)
        if target.core_bbox is None:
            target.core_bbox = candidate.core_bbox
        elif candidate.core_bbox is not None:
            target.core_bbox = _bbox_union(target.core_bbox, candidate.core_bbox)
        target.line_indices.update(candidate.line_indices)
        target.score = max(target.score, candidate.score)
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def _materialize_table_blocks(
    source: _PageSource,
    candidates: list[_TableCandidate],
) -> tuple[list[dict[str, Any]], set[int]]:
    """为候选生成空间投影 content，仅认领投影成功的文本行。"""

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_bbox_overlap_in_smaller(candidate.bbox, block["bbox"]) >= 0.5 for block in blocks):
            continue
        output_angle = candidate.angle
        projection_line_indices = _candidate_projection_line_indices(source, candidate)
        try:
            candidate_chars = [
                char for line in source.lines if line.source_index in projection_line_indices for char in line.chars
            ]
            content = project_pdf_table_text(
                candidate_chars,
                candidate.bbox,
                angle=candidate.angle,
            )
        except Exception as exc:
            # 单个表格的字符投影异常只撤销该候选，不能中止整页提取。
            logger.warning(f"Flash table projection failed and rolled back: bbox={candidate.bbox}, error={exc}")
            continue
        if not content or not content.strip():
            continue
        blocks.append(
            {
                "type": "table",
                "bbox": candidate.bbox,
                "angle": output_angle,
                "content": content,
            }
        )
        # 只认领候选明确接纳的视觉行，避免远标题与表格之间的正文被矩形 bbox 连带删除。
        claimed.update(projection_line_indices)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _bbox_overlap_in_first(first: BBox, second: BBox) -> float:
    """计算交集面积占第一个 bbox 面积的比例。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = _bbox_area(first)
    return width * height / first_area if first_area > 0 else 0.0


def _form_supersedes_nested_bbox(form_bbox: BBox, nested_bbox: BBox) -> bool:
    """判断 Form 是否应整体吞并其内部面积明显更小的候选容器。"""

    form_area = _bbox_area(form_bbox)
    nested_area = _bbox_area(nested_bbox)
    return (
        form_area > 0
        and nested_area < 0.5 * form_area
        and _bbox_overlap_in_first(nested_bbox, form_bbox) >= 0.9
    )


def _select_form_image_bboxes(source: _PageSource) -> list[BBox]:
    """按页面占比、行高和内部视觉证据筛选矢量 Form 图片候选。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0 or not source.form_bboxes:
        return []
    effective_heights = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.angle == 0
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    output: list[BBox] = []
    for raw_bbox in source.form_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = _bbox_area(bbox) / page_area
        if not (
            _MIN_FORM_IMAGE_PAGE_AREA_RATIO <= area_ratio <= _MAX_FORM_IMAGE_PAGE_AREA_RATIO
            and width >= 4.0 * median_height
            and height >= 4.0 * median_height
        ):
            continue

        member_rows = {
            line.visual_row_id if line.visual_row_id is not None else line.source_index
            for line in source.lines
            if _bbox_overlap_in_first(line.bbox, bbox) >= 0.9
        }
        internal_drawing_count = sum(
            _bbox_overlap_in_first(drawing_line.bbox, bbox) >= 0.9
            for drawing_line in source.drawing_lines
        )
        if len(member_rows) < 2 and internal_drawing_count < 4:
            continue
        output.append(bbox)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _build_form_image_blocks(
    source: _PageSource,
    form_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """把 Form 及其完整内含文本输出为 image，并保持 source_index 唯一认领。"""

    if not form_bboxes:
        return [], set()
    members_by_candidate: list[list[_LineItem]] = [[] for _ in form_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        matching_indices = [
            candidate_index
            for candidate_index, bbox in enumerate(form_bboxes)
            if _bbox_overlap_in_first(line.bbox, bbox) >= 0.9
        ]
        if not matching_indices:
            continue
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(form_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks = [
        {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        for bbox, members in zip(form_bboxes, members_by_candidate, strict=True)
    ]
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _build_graphic_like_blocks(
    source: _PageSource,
    table_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """在表格认领后把紧凑绘图组件及其短标签聚成内部图形文本块。"""

    lines = [line for line in source.lines if line.source_index not in claimed_line_indices]
    if len(lines) < 2 or len(source.drawing_lines) < 4:
        return [], set()

    effective_heights = [
        max(
            0.1,
            line.effective_height
            or min(
                max(0.1, line.bbox[2] - line.bbox[0]),
                max(0.1, line.bbox[3] - line.bbox[1]),
            ),
        )
        for line in lines
    ]
    median_height = statistics.median(effective_heights)
    lanes = _infer_graphic_text_lanes(lines, source.page_size, median_height)
    candidates = _detect_graphic_candidates(
        source.drawing_lines,
        source.page_size,
        median_height,
        lanes,
        table_bboxes,
    )
    if not candidates:
        return [], set()

    row_groups: dict[tuple[int, int, int, int], list[_LineItem]] = {}
    for line in lines:
        lane_index = _graphic_lane_index(line.bbox, lanes)
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault(
            (line.angle, row_kind, row_identity, lane_index),
            [],
        ).append(line)

    for row_lines in row_groups.values():
        row_lane_index = _graphic_lane_index(row_lines[0].bbox, lanes)
        matches: list[tuple[int, float, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate.lane_index != row_lane_index:
                continue
            member_flags = [
                _is_graphic_label_member(line, candidate.core_bbox, median_height)
                for line in row_lines
            ]
            # 同一 pdftext 视觉行必须整体归属或整体保留，避免只吞掉 caption 的短碎片。
            if not all(member_flags):
                continue
            inside_count = sum(
                _point_in_bbox(
                    (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                    candidate.core_bbox,
                )
                for line in row_lines
            )
            mean_distance = statistics.fmean(
                _bbox_distance(line.bbox, candidate.core_bbox) for line in row_lines
            )
            matches.append((-inside_count, mean_distance, candidate_index))
        if not matches:
            continue
        candidate_index = min(matches)[2]
        candidates[candidate_index].line_indices.update(line.source_index for line in row_lines)

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    lines_by_index = {line.source_index: line for line in lines}
    for candidate in candidates:
        members = [
            lines_by_index[source_index]
            for source_index in sorted(candidate.line_indices)
            if source_index in lines_by_index
        ]
        if len(members) < 2:
            continue
        block = _graphic_members_to_block(candidate, members, source.page_size)
        if block is None:
            continue
        blocks.append(block)
        claimed.update(line.source_index for line in members)

    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _infer_graphic_text_lanes(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    median_height: float,
) -> list[_TextLane]:
    """用横排正文推断页内栏带，供不同角度的图形标签共享栏归属。"""

    line_geometry = [(line, line.bbox) for line in lines if line.angle == 0]
    if not line_geometry:
        return [_TextLane(left=0.0, right=page_size[0])]
    angle_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    angle_median_height = statistics.median(angle_heights) if angle_heights else median_height
    lanes = [
        lane
        for lane in _infer_text_lanes(
            line_geometry,
            page_size[0],
            angle_median_height,
        )
        if not lane.is_span
    ]
    return lanes or [_TextLane(left=0.0, right=page_size[0])]


def _graphic_lane_index(bbox: BBox, lanes: list[_TextLane]) -> int:
    """按中心点、水平覆盖和距离为 bbox 选择唯一栏带。"""

    center_x = _bbox_center_x(bbox)
    best_index = 0
    best_score = (-1, -1.0, -math.inf)
    for lane_index, lane in enumerate(lanes):
        inside = int(lane.left <= center_x <= lane.right)
        overlap = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left))
        if inside:
            distance = 0.0
        else:
            distance = min(abs(center_x - lane.left), abs(center_x - lane.right))
        score = (inside, overlap, -distance)
        if score > best_score:
            best_score = score
            best_index = lane_index
    return best_index


def _detect_graphic_candidates(
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    median_height: float,
    lanes: list[_TextLane],
    table_bboxes: list[BBox],
) -> list[_GraphicCandidate]:
    """从非表格绘图线连通分量中筛选尺寸受限的图形容器。"""

    tolerance = max(2.0, 0.75 * median_height)
    candidates: list[_GraphicCandidate] = []
    for component in _connected_drawing_line_components(drawing_lines, tolerance):
        horizontal_count = sum(line.orientation == "horizontal" for line in component)
        vertical_count = len(component) - horizontal_count
        core_bbox = _bbox_union_many([line.bbox for line in component])
        width = core_bbox[2] - core_bbox[0]
        height = core_bbox[3] - core_bbox[1]
        if (
            len(component) < 4
            or horizontal_count < 2
            or vertical_count < 2
            or width < 4.0 * median_height
            or height < 3.0 * median_height
            or width > 0.5 * page_size[0]
            or height > 0.5 * page_size[1]
        ):
            continue
        if any(_bbox_overlap_in_smaller(core_bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes):
            continue
        candidates.append(
            _GraphicCandidate(
                core_bbox=core_bbox,
                lane_index=_graphic_lane_index(core_bbox, lanes),
            )
        )
    return candidates


def _connected_drawing_line_components(
    drawing_lines: list[_AxisLine],
    tolerance: float,
) -> list[list[_AxisLine]]:
    """按 bbox 间距连接相邻绘图线，并返回互不重叠的连通分量。"""

    parents = list(range(len(drawing_lines)))

    def find(index: int) -> int:
        """查找绘图线连通分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个距离满足条件的绘图线分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(drawing_lines):
        for second_index in range(first_index + 1, len(drawing_lines)):
            if _bbox_distance(first.bbox, drawing_lines[second_index].bbox) <= tolerance:
                union(first_index, second_index)

    components: dict[int, list[_AxisLine]] = {}
    for line_index, line in enumerate(drawing_lines):
        components.setdefault(find(line_index), []).append(line)
    return list(components.values())


def _is_graphic_label_member(
    line: _LineItem,
    core_bbox: BBox,
    median_height: float,
) -> bool:
    """判断短文本是否位于图形核心内部或对应轴向的邻近标签区。"""

    center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
    if _point_in_bbox(center, core_bbox):
        return True

    line_height = max(
        0.1,
        line.effective_height
        or min(
            max(0.1, line.bbox[2] - line.bbox[0]),
            max(0.1, line.bbox[3] - line.bbox[1]),
        ),
    )
    if line.angle in {90, 270}:
        primary_length = line.bbox[3] - line.bbox[1]
        core_primary_length = core_bbox[3] - core_bbox[1]
    else:
        primary_length = line.bbox[2] - line.bbox[0]
        core_primary_length = core_bbox[2] - core_bbox[0]
    if primary_length > min(5.0 * line_height, 0.5 * core_primary_length):
        return False

    horizontal_gap = max(core_bbox[0] - line.bbox[2], line.bbox[0] - core_bbox[2], 0.0)
    vertical_gap = max(core_bbox[1] - line.bbox[3], line.bbox[1] - core_bbox[3], 0.0)
    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="x") >= 0.15:
        return vertical_gap <= 2.5 * median_height
    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="y") >= 0.15:
        horizontal_limit = max(2.5 * median_height, 0.2 * (core_bbox[2] - core_bbox[0]))
        return horizontal_gap <= horizontal_limit
    return False


def _image_members_to_content(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> str:
    """按视觉行和页内位置生成图片内部文本，保留不同视觉行之间的换行。"""

    row_groups: dict[tuple[int, int, int], list[_LineItem]] = {}
    for line in members:
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault((line.angle, row_kind, row_identity), []).append(line)

    rows: list[tuple[BBox, str]] = []
    for row_lines in row_groups.values():
        row_bbox = _bbox_union_many([line.bbox for line in row_lines])
        angle = row_lines[0].angle
        local_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in row_lines
        ]
        content = _join_formula_visual_row(local_geometry, page_size)
        if content:
            rows.append((row_bbox, content))
    rows.sort(key=lambda item: (item[0][1], item[0][0]))
    return _sanitize_pdf_control_text(
        "\n".join(row_content for _row_bbox, row_content in rows),
        preserve_newlines=True,
    ).strip()


def _graphic_members_to_block(
    candidate: _GraphicCandidate,
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """生成含内部文本的矢量图 image block，并合并绘图核心与标签 bbox。"""

    content = _image_members_to_content(members, page_size)
    if not content:
        return None
    return {
        "type": "image",
        "bbox": _bbox_union(candidate.core_bbox, _bbox_union_many([line.bbox for line in members])),
        "angle": 0,
        "content": content,
    }


def _bbox_area(bbox: BBox) -> float:
    """返回合法 bbox 的非负面积。"""

    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _build_raster_image_blocks(
    source: _PageSource,
    container_blocks: list[dict[str, Any]],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """过滤点阵图、避让高优先级容器，并唯一认领图片内部的原生文本。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0:
        return [], set()

    container_bboxes = [
        bbox
        for block in container_blocks
        if (bbox := _coerce_bbox(block.get("bbox"))) is not None
    ]
    candidate_bboxes: list[BBox] = []
    for raw_bbox in source.image_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None or _bbox_area(bbox) / page_area < _MIN_RASTER_IMAGE_PAGE_AREA_RATIO:
            continue
        if any(
            _bbox_overlap_in_smaller(bbox, container_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        candidate_bboxes.append(bbox)
    if not candidate_bboxes:
        return [], set()

    members_by_candidate: list[list[_LineItem]] = [[] for _ in candidate_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        matching_indices = [
            candidate_index
            for candidate_index, bbox in enumerate(candidate_bboxes)
            if _point_in_bbox(center, bbox)
        ]
        if not matching_indices:
            continue
        # 重叠点阵图共享内部文本时归属最小容器，避免 content 重复。
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(candidate_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks = [
        {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        for bbox, members in zip(candidate_bboxes, members_by_candidate, strict=True)
    ]
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _candidate_projection_line_indices(
    source: _PageSource,
    candidate: _TableCandidate,
) -> set[int]:
    """合并核心成员、同基线续段及非零角度表格的表头文本。"""

    line_indices = set(candidate.line_indices)
    if candidate.core_bbox is not None:
        for line in source.lines:
            if _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.core_bbox,
            ):
                line_indices.add(line.source_index)
    if candidate.angle == 0:
        _expand_candidate_same_baseline_members(source, candidate, line_indices)
        return line_indices
    if candidate.core_bbox is None:
        return line_indices

    candidate_local_bbox = _rotate_bbox_to_upright(
        candidate.bbox,
        source.page_size,
        candidate.angle,
    )
    core_local_bbox = _rotate_bbox_to_upright(
        candidate.core_bbox,
        source.page_size,
        candidate.angle,
    )
    for line in source.lines:
        if line.angle != candidate.angle:
            continue
        page_center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        if not _point_in_bbox(page_center, candidate.bbox):
            continue
        local_bbox = _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        local_center_y = _bbox_center_y(local_bbox)
        if not candidate_local_bbox[1] <= local_center_y <= candidate_local_bbox[3]:
            continue
        if _bbox_axis_overlap_ratio(local_bbox, core_local_bbox, axis="x") < 0.05:
            continue
        line_indices.add(line.source_index)
    return line_indices


def _expand_candidate_same_baseline_members(
    source: _PageSource,
    candidate: _TableCandidate,
    line_indices: set[int],
) -> None:
    """迭代吸收完整候选框内与已认领成员同基线相邻的 angle=0 续段。"""

    local_bboxes = {
        line.source_index: _rotate_bbox_to_upright(
            line.bbox,
            source.page_size,
            candidate.angle,
        )
        for line in source.lines
        if line.angle == candidate.angle
    }
    changed = True
    while changed:
        changed = False
        selected_lines = [
            line
            for line in source.lines
            if line.angle == candidate.angle and line.source_index in line_indices
        ]
        for line in source.lines:
            if line.angle != candidate.angle or line.source_index in line_indices:
                continue
            if not _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                candidate.bbox,
            ):
                continue
            line_bbox = local_bboxes[line.source_index]
            line_height = _line_effective_height(line, line_bbox)
            for selected in selected_lines:
                if (
                    line.font_signature is not None
                    and selected.font_signature is not None
                    and line.font_signature != selected.font_signature
                ):
                    continue
                selected_bbox = local_bboxes[selected.source_index]
                if not _same_baseline_geometry(
                    line_bbox,
                    line_height,
                    selected_bbox,
                    _line_effective_height(selected, selected_bbox),
                ):
                    continue
                line_indices.add(line.source_index)
                changed = True
                break


def _merge_same_baseline_text_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在表格认领后合并同基线、同字体且水平邻近的正文 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找同行合并并查集的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        """合并两个满足同行条件的文本 run。"""

        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left_line in enumerate(lines):
        for right_index in range(left_index + 1, len(lines)):
            right_line = lines[right_index]
            if _can_merge_same_baseline_pair(
                left_line,
                local_bboxes[left_index],
                right_line,
                local_bboxes[right_index],
                table_bboxes,
            ):
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)

    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(key=lambda index: (local_bboxes[index][0], local_bboxes[index][1], lines[index].source_index))
        output.append(_merge_same_baseline_group(indices, lines, local_bboxes, page_size))
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _merge_post_semantic_text_runs(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在容器、公式和标题结束后合并紧贴同基线的普通混合字体 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [
        _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        for line in lines
    ]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找后处理同行合并分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个后处理同行分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first_line in enumerate(lines):
        if first_line.semantic_type is not None:
            continue
        first_bbox = local_bboxes[first_index]
        first_height = _line_effective_height(first_line, first_bbox)
        for second_index in range(first_index + 1, len(lines)):
            second_line = lines[second_index]
            if second_line.semantic_type is not None or first_line.angle != second_line.angle:
                continue
            if _connection_crosses_table(
                first_line.bbox,
                second_line.bbox,
                table_bboxes,
            ):
                continue
            second_bbox = local_bboxes[second_index]
            second_height = _line_effective_height(second_line, second_bbox)
            if _post_semantic_same_baseline_geometry(
                first_bbox,
                first_height,
                second_bbox,
                second_height,
            ):
                union(first_index, second_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(
            key=lambda index: (
                local_bboxes[index][0],
                local_bboxes[index][1],
                lines[index].source_index,
            )
        )
        output.append(
            _merge_same_baseline_group(
                indices,
                lines,
                local_bboxes,
                page_size,
            )
        )
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _post_semantic_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """放宽上下标字号差异，仅合并水平紧贴且垂直充分交叠的普通 run。"""

    pair_height = max(first_height, second_height)
    if min(first_height, second_height) <= 0 or pair_height / min(
        first_height,
        second_height,
    ) > 1.6:
        return False
    y_overlap = max(
        0.0,
        min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
    )
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    horizontal_gap = right_bbox[0] - left_bbox[2]
    return -0.2 * pair_height <= horizontal_gap <= max(2.0, 0.35 * pair_height)


def _can_merge_same_baseline_pair(
    first: _LineItem,
    first_bbox: BBox,
    second: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """判断两个剩余文本 run 是否属于同一条物理基线。"""

    if first.angle != second.angle:
        return False
    if first.semantic_type != second.semantic_type:
        return False
    if first.visual_row_id == second.visual_row_id and (first.split_from_row or second.split_from_row):
        return False
    if _connection_crosses_table(first.bbox, second.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first, first_bbox)
    second_height = _line_effective_height(second, second_bbox)
    has_compatible_dominant_font = not (
        first.font_signature is None
        or second.font_signature is None
        or first.font_coverage < 0.75
        or second.font_coverage < 0.75
        or first.font_signature != second.font_signature
    )
    if has_compatible_dominant_font and _same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    ):
        return True
    return _touching_same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    )


def _touching_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """为低字体覆盖率 run 提供严格的紧贴同基线几何兜底。"""

    pair_height = max(first_height, second_height)
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    smaller_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / smaller_bbox_height < 0.7:
        return False
    if abs(_bbox_center_y(first_bbox) - _bbox_center_y(second_bbox)) > 0.5 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    return -0.15 * pair_height <= signed_gap <= 0.75


def _same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
    *,
    maximum_gap: float | None = None,
) -> bool:
    """仅依据行高、垂直交叠和水平净空判断两个局部 bbox 是否同基线相邻。"""

    pair_height = max(first_height, second_height)
    if min(first_height, second_height) <= 0 or pair_height / min(first_height, second_height) > 1.35:
        return False
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7 and abs(first_bbox[3] - second_bbox[3]) > 0.25 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    gap_limit = max(3.0, 0.75 * pair_height) if maximum_gap is None else maximum_gap
    return -0.25 * pair_height <= signed_gap <= gap_limit


def _merge_same_baseline_group(
    indices: list[int],
    lines: list[_LineItem],
    local_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序合并一个同基线分量，并保留全部字符与几何信息。"""

    members = [lines[index] for index in indices]
    content_parts = [members[0].text.strip()]
    for previous_index, current_index in zip(indices, indices[1:]):
        previous_bbox = local_bboxes[previous_index]
        current_bbox = local_bboxes[current_index]
        signed_gap = current_bbox[0] - previous_bbox[2]
        glyph_width = statistics.median(
            [
                width
                for member in (lines[previous_index], lines[current_index])
                if (width := _median_native_glyph_width(member, page_size)) is not None
            ]
            or [1.0]
        )
        separator = "" if signed_gap <= max(0.5, 0.25 * glyph_width) else " "
        content_parts.extend([separator, lines[current_index].text.strip()])

    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in members]),
        angle=members[0].angle,
        source_index=min(member.source_index for member in members),
        chars=[char for member in members for char in member.chars],
        visual_row_id=min(
            (member.visual_row_id for member in members if member.visual_row_id is not None),
            default=None,
        ),
        run_index=min(member.run_index for member in members),
        effective_height=statistics.median(member.effective_height for member in members),
        font_signature=members[0].font_signature,
        font_coverage=min(member.font_coverage for member in members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight
            for member in members
            if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width
            for member in members
            if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in members)
        else None,
        split_from_row=any(member.split_from_row for member in members),
        semantic_type=members[0].semantic_type,
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _median_native_glyph_width(line: _LineItem, page_size: tuple[float, float]) -> float | None:
    """返回单个原生 run 的可见字符中位宽度，缺少字符时返回空。"""

    if line.median_glyph_width is not None:
        return line.median_glyph_width
    widths: list[float] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if not raw_char.isprintable() or raw_char.isspace() or bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
    line.median_glyph_width = statistics.median(widths) if widths else None
    return line.median_glyph_width


def _build_formula_like_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[_LineItem]]:
    """仅依据栏带、右侧短锚点和空间连通关系聚合公式状区域。"""

    blocks: list[dict[str, Any]] = []
    claimed_source_indices: set[int] = set()
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if len(angle_geometry) < 2:
            continue
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in angle_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(angle_geometry, local_page_width, median_height)
        for lane in lanes:
            if lane.is_span or len(lane.lines) < 2:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            dominant_body_font = _infer_formula_body_font(lane, median_height)
            anchors = _find_formula_spatial_anchors(
                lane,
                median_height,
                dominant_body_font,
            )
            if not anchors:
                continue
            anchor_centers = [_bbox_center_y(anchor.bbox) for anchor in anchors]
            lane_top = min(bbox[1] for _line, bbox in lane.lines)
            lane_bottom = max(bbox[3] for _line, bbox in lane.lines)
            for anchor_index, anchor in enumerate(anchors):
                anchor_line = anchor.line
                if anchor_line.source_index in claimed_source_indices:
                    continue
                band_top = lane_top
                band_bottom = lane_bottom
                if anchor_index > 0:
                    band_top = max(
                        band_top,
                        (anchor_centers[anchor_index - 1] + anchor_centers[anchor_index]) / 2.0,
                    )
                if anchor_index + 1 < len(anchors):
                    band_bottom = min(
                        band_bottom,
                        (anchor_centers[anchor_index] + anchor_centers[anchor_index + 1]) / 2.0,
                    )
                members = _grow_formula_spatial_component(
                    lane,
                    anchor,
                    band_top,
                    band_bottom,
                    claimed_source_indices,
                    table_bboxes,
                    dominant_body_font,
                    median_height,
                )
                if len(members) < 2:
                    continue
                if len(members) == 2 and _bbox_axis_overlap_ratio(
                    members[0][1],
                    members[1][1],
                    axis="y",
                ) < 0.2:
                    continue
                block = _formula_members_to_block(
                    members,
                    page_size,
                    angle,
                    anchor_source_index=anchor_line.source_index,
                )
                if block is None:
                    continue
                blocks.append(block)
                claimed_source_indices.update(line.source_index for line, _bbox in members)

    remaining_lines = [line for line in lines if line.source_index not in claimed_source_indices]
    return blocks, remaining_lines


def _find_formula_spatial_anchors(
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None = None,
) -> list[_FormulaAnchor]:
    """查找栏带右缘短块或带编号后缀的非正文字体公式锚点。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_interval = _formula_lane_body_interval(lane, median_height)
    if body_interval is None:
        return []
    body_top, body_bottom = body_interval
    anchors: list[_FormulaAnchor] = []
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        line_width = bbox[2] - bbox[0]
        is_short_right_anchor = line_width <= max(4.0 * line_height, 0.12 * lane_width)
        has_formula_number_suffix = _split_trailing_formula_number(line.text) is not None
        is_wide_numbered_anchor = (
            has_formula_number_suffix
            and line_width <= 0.75 * lane_width
            and dominant_body_font is not None
            and (
                line.font_signature != dominant_body_font
                or line.font_coverage < 0.75
            )
        )
        if not is_short_right_anchor and not is_wide_numbered_anchor:
            continue
        same_row_fragments = [
            other_line
            for other_line, _other_bbox in lane.lines
            if line.visual_row_id is not None
            and other_line.visual_row_id == line.visual_row_id
            and (line.split_from_row or other_line.split_from_row)
        ]
        if len(same_row_fragments) >= 3 and not any(
            other_line.font_coverage < 0.75
            or (
                dominant_body_font is not None
                and other_line.font_signature != dominant_body_font
            )
            for other_line in same_row_fragments
            if other_line.source_index != line.source_index
        ):
            # 一条粗行被多个大空格拆成密集词组时更像普通排版行，不能把末词当作公式编号锚点。
            continue
        if abs(lane.right - bbox[2]) > max(3.0, 0.02 * lane_width):
            continue
        center_y = _bbox_center_y(bbox)
        detached_below_body = body_bottom < center_y <= body_bottom + 6.0 * median_height
        if not body_top <= center_y <= body_bottom and not detached_below_body:
            continue
        has_left_peer = any(
            other_line.source_index != line.source_index
            and other_bbox[2] - other_bbox[0] <= 0.75 * lane_width
            and _bbox_center_x(other_bbox) < bbox[0]
            and (
                _formula_detached_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
                if detached_below_body
                else _formula_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
            )
            for other_line, other_bbox in lane.lines
        )
        if has_left_peer:
            anchors.append(
                _FormulaAnchor(
                    line=line,
                    bbox=bbox,
                    detached_below_body=detached_below_body,
                )
            )
    return _deduplicate_formula_anchors(anchors, median_height)


def _infer_formula_body_font(
    lane: _TextLane,
    median_height: float,
) -> tuple[str, int] | None:
    """从栏内常规宽正文行推断 dominant font，供公式扩张排除正文前缀。"""

    lane_width = max(0.1, lane.right - lane.left)
    font_counts: dict[tuple[str, int], int] = {}
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        if bbox[2] - bbox[0] < 0.35 * lane_width:
            continue
        if not 0.8 * median_height <= line_height <= 1.25 * median_height:
            continue
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        font_counts[line.font_signature] = font_counts.get(line.font_signature, 0) + 1
    if not font_counts:
        return None
    return max(font_counts.items(), key=lambda item: (item[1], item[0]))[0]


def _formula_lane_body_interval(
    lane: _TextLane,
    median_height: float,
) -> tuple[float, float] | None:
    """用连续出现的常规宽行确定栏带正文纵向范围，排除孤立页眉。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_lines = sorted(
        (
            item
            for item in lane.lines
            if item[1][2] - item[1][0]
            >= max(4.0 * _line_effective_height(*item), 0.35 * lane_width)
        ),
        key=lambda item: (item[1][1], item[1][0]),
    )
    if len(body_lines) < 3:
        return None
    dense_lines: list[tuple[_LineItem, BBox]] = []
    for index, item in enumerate(body_lines):
        has_close_previous = index > 0 and item[1][1] - body_lines[index - 1][1][3] <= 1.5 * median_height
        has_close_next = (
            index + 1 < len(body_lines)
            and body_lines[index + 1][1][1] - item[1][3] <= 1.5 * median_height
        )
        if has_close_previous or has_close_next:
            dense_lines.append(item)
    if len(dense_lines) < 3:
        return None
    return (
        min(bbox[1] for _line, bbox in dense_lines),
        max(bbox[3] for _line, bbox in dense_lines),
    )


def _deduplicate_formula_anchors(
    anchors: list[_FormulaAnchor],
    median_height: float,
) -> list[_FormulaAnchor]:
    """同一高度出现多个右缘短块时只保留最靠右的空间锚点。"""

    if not anchors:
        return []
    output: list[_FormulaAnchor] = []
    tolerance = max(1.5, 0.35 * median_height)
    for anchor in sorted(anchors, key=lambda item: (_bbox_center_y(item.bbox), -item.bbox[2])):
        if output and abs(_bbox_center_y(anchor.bbox) - _bbox_center_y(output[-1].bbox)) <= tolerance:
            if (anchor.bbox[2], -anchor.bbox[0]) > (output[-1].bbox[2], -output[-1].bbox[0]):
                output[-1] = anchor
            continue
        output.append(anchor)
    return output


def _grow_formula_spatial_component(
    lane: _TextLane,
    anchor: _FormulaAnchor,
    band_top: float,
    band_bottom: float,
    claimed_source_indices: set[int],
    table_bboxes: list[BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """从右缘锚点的左侧首批成员出发，按二维邻接扩展公式分量。"""

    anchor_line, anchor_bbox = anchor.line, anchor.bbox
    anchor_geometry = (anchor_line, anchor_bbox)
    lane_width = max(0.1, lane.right - lane.left)
    candidates = [
        item
        for item in lane.lines
        if item[0].source_index not in claimed_source_indices
        and item[1][2] - item[1][0] <= 0.8 * lane_width
        and band_top <= _bbox_center_y(item[1]) <= band_bottom
        and not _is_formula_body_barrier(
            item,
            lane,
            dominant_body_font,
            median_height,
        )
        and not _is_formula_body_prefix(
            item,
            lane,
            anchor_geometry,
            dominant_body_font,
            median_height,
        )
    ]
    seeds = [
        item
        for item in candidates
        if item[0].source_index != anchor_line.source_index
        and _bbox_center_x(item[1]) < anchor_bbox[0]
        and (
            _formula_detached_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
            if anchor.detached_below_body
            else _formula_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
        )
        and not _connection_crosses_table(anchor_line.bbox, item[0].bbox, table_bboxes)
    ]
    if not seeds:
        return []

    members = [anchor_geometry, *seeds]
    member_sources = {line.source_index for line, _bbox in members}
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            candidate_line, candidate_bbox = candidate
            if candidate_line.source_index in member_sources:
                continue
            if any(
                _formula_lines_are_connected(
                    member_line,
                    member_bbox,
                    candidate_line,
                    candidate_bbox,
                    table_bboxes,
                )
                for member_line, member_bbox in members
            ):
                members.append(candidate)
                member_sources.add(candidate_line.source_index)
                changed = True
    return members


def _is_formula_body_barrier(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """识别近满栏、高置信正文行，阻止公式连通分量跨越正文边界。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    lane_width = max(0.1, lane.right - lane.left)
    line_height = _line_effective_height(line, bbox)
    return (
        line.font_signature == dominant_body_font
        and bbox[2] - bbox[0] >= 0.85 * lane_width
        and 0.8 * median_height <= line_height <= 1.25 * median_height
    )


def _is_formula_body_prefix(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    anchor: tuple[_LineItem, BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """识别锚点上方左对齐的常规正文行，防止公式空间扩张越界认领。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    anchor_line, anchor_bbox = anchor
    line_height = _line_effective_height(line, bbox)
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    if _bbox_center_y(bbox) > _bbox_center_y(anchor_bbox) - 0.2 * max(line_height, anchor_height):
        return False
    if abs(bbox[0] - lane.left) > max(3.0, 0.75 * median_height):
        return False
    return (
        line.font_signature == dominant_body_font
        and line.font_coverage >= 0.75
        and 0.8 * median_height <= line_height <= 1.25 * median_height
    )


def _formula_detached_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """放宽正文密集区下方锚点的同高匹配，以接纳多行分段公式底部。"""

    has_vertical_overlap = min(anchor_bbox[3], candidate_bbox[3]) > max(anchor_bbox[1], candidate_bbox[1])
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return has_vertical_overlap or center_difference <= max(anchor_height, candidate_height)


def _formula_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """判断左侧短行是否与右缘锚点处在同一公式高度带。"""

    overlap_ratio = _bbox_axis_overlap_ratio(anchor_bbox, candidate_bbox, axis="y")
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return overlap_ratio >= 0.3 or center_difference <= 0.6 * max(anchor_height, candidate_height)


def _formula_lines_are_connected(
    first_line: _LineItem,
    first_bbox: BBox,
    second_line: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """按垂直接近和水平覆盖判断两个公式成员是否空间连通。"""

    if first_line.angle != second_line.angle:
        return False
    if _connection_crosses_table(first_line.bbox, second_line.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first_line, first_bbox)
    second_height = _line_effective_height(second_line, second_bbox)
    pair_height = max(first_height, second_height)
    vertical_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y")
    vertical_gap = max(first_bbox[1] - second_bbox[3], second_bbox[1] - first_bbox[3], 0.0)
    if vertical_overlap < 0.2 and vertical_gap > 0.6 * pair_height:
        return False
    horizontal_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x")
    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    return horizontal_overlap > 0.0 or horizontal_gap <= 1.5 * pair_height


def _is_detached_formula_sidecar(
    anchor: tuple[_LineItem, BBox],
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> bool:
    """仅依据 bbox 判断右侧锚点是否为与公式主体分离的窄幅 sidecar。"""

    anchor_line, anchor_bbox = anchor
    body_bboxes = [
        bbox
        for line, bbox in members
        if line.source_index != anchor_line.source_index
    ]
    if not body_bboxes:
        return False

    body_bbox = _bbox_union_many(body_bboxes)
    component_bbox = _bbox_union(body_bbox, anchor_bbox)
    effective_height = max(0.1, median_height)
    anchor_width = max(0.0, anchor_bbox[2] - anchor_bbox[0])
    component_width = max(0.1, component_bbox[2] - component_bbox[0])
    horizontal_gap = anchor_bbox[0] - body_bbox[2]
    right_tolerance = max(0.5, 0.1 * effective_height)
    minimum_gap = max(2.5 * effective_height, 0.08 * component_width)

    return (
        anchor_bbox[0] >= body_bbox[2]
        and anchor_bbox[2] >= component_bbox[2] - right_tolerance
        and anchor_width <= 2.0 * effective_height
        and horizontal_gap > minimum_gap
    )


def _split_trailing_formula_number(text: str) -> tuple[str, str] | None:
    """拆出右缘文本末尾的圆括号公式序号，并保留序号前的标点或正文。"""

    match = _FORMULA_NUMBER_SUFFIX_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    return match.group("prefix").rstrip(), match.group("marker").strip()


def _formula_members_to_block(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    angle: int,
    *,
    anchor_source_index: int,
) -> dict[str, Any] | None:
    """把公式空间分量按视觉行聚类，并将非末视觉行的离散右侧 sidecar 后置。"""

    heights = [_line_effective_height(line, bbox) for line, bbox in members]
    median_height = statistics.median(heights) if heights else 1.0
    row_tolerance = max(1.5, 0.35 * median_height)
    rows: list[list[tuple[_LineItem, BBox]]] = []
    for member in sorted(members, key=lambda item: (_bbox_center_y(item[1]), item[1][0], item[0].source_index)):
        if not rows:
            rows.append([member])
            continue
        row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in rows[-1])
        if abs(_bbox_center_y(member[1]) - row_center) <= row_tolerance:
            rows[-1].append(member)
        else:
            rows.append([member])

    trailing_sidecar_content: str | None = None
    # 右侧 sidecar 按视觉 y 常落在分式中部；仅在其后仍有公式行时转为逻辑末行。
    for row_index, row in enumerate(rows[:-1]):
        anchor_member = next(
            (member for member in row if member[0].source_index == anchor_source_index),
            None,
        )
        if anchor_member is None:
            continue
        formula_number_parts = _split_trailing_formula_number(anchor_member[0].text)
        if formula_number_parts is not None:
            prefix, marker = formula_number_parts
            rows[row_index] = [
                (
                    (replace(member[0], text=prefix), member[1])
                    if member[0].source_index == anchor_source_index and prefix
                    else member
                )
                for member in row
                if member[0].source_index != anchor_source_index or prefix
            ]
            trailing_sidecar_content = marker
        elif _is_detached_formula_sidecar(anchor_member, members, median_height):
            rows[row_index] = [
                member for member in row if member[0].source_index != anchor_source_index
            ]
            trailing_sidecar_content = anchor_member[0].text.strip()
        break

    row_contents = [_join_formula_visual_row(row, page_size) for row in rows if row]
    if trailing_sidecar_content is not None:
        row_contents.append(trailing_sidecar_content)
    content = _sanitize_pdf_control_text("\n".join(filter(None, row_contents)), preserve_newlines=True)
    if not content.strip():
        return None
    return {
        "type": "equation",
        "bbox": _bbox_union_many([line.bbox for line, _bbox in members]),
        "angle": angle,
        "content": content,
    }


def _join_formula_visual_row(
    row: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
) -> str:
    """将一个公式视觉行按局部 x 排序，并按字宽估计几何空格。"""

    ordered = sorted(row, key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not ordered:
        return ""
    parts = [ordered[0][0].text.strip()]
    for previous, current in zip(ordered, ordered[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        gap = current_bbox[0] - previous_bbox[2]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(current_line, current_bbox),
        )
        glyph_widths = [
            width
            for line in (previous_line, current_line)
            if (width := _median_native_glyph_width(line, page_size)) is not None
        ]
        glyph_width = statistics.median(glyph_widths) if glyph_widths else max(1.0, 0.5 * pair_height)
        if gap <= max(0.5, 0.2 * pair_height):
            separator = ""
        else:
            separator = " " * max(1, min(8, int(round(gap / glyph_width))))
        parts.extend([separator, current_line.text.strip()])
    return "".join(parts).strip()


def _restore_dense_split_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在公式认领后恢复同一栏带内被均匀大空格拆开的密集原生视觉行。"""

    if len(lines) < 3:
        return list(lines)
    lane_keys: dict[int, tuple[int, int]] = {}
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane_index, lane in enumerate(lanes):
            if lane.is_span:
                continue
            for line, _bbox in lane.lines:
                lane_keys[line.source_index] = (angle, lane_index)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    restored_lines: list[_LineItem] = []
    for members in row_groups.values():
        if not _can_restore_dense_split_visual_row(
            members,
            page_size,
            table_bboxes,
            lane_keys,
        ):
            continue
        restored_lines.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(restored_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _merge_title_resolved_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """在标题判定后合并同行标题或已降级的混合字体正文 run。"""

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    merged_rows: list[_LineItem] = []
    for members in row_groups.values():
        if len(members) < 2 or not all(member.split_from_row for member in members):
            continue
        semantic_types = {member.semantic_type for member in members}
        if len(semantic_types) != 1:
            continue
        semantic_type = next(iter(semantic_types))
        font_signatures = {member.font_signature for member in members}
        dense_same_font_text = _is_dense_same_font_two_run_row(
            members,
            page_size,
        )
        if semantic_type != "paragraph_title" and not (
            semantic_type is None
            and (len(font_signatures) > 1 or dense_same_font_text)
        ):
            continue
        local_geometry = [
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ]
        local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
        if any(
            not _same_baseline_geometry(
                previous[1],
                _line_effective_height(*previous),
                current[1],
                _line_effective_height(*current),
                maximum_gap=3.0
                * max(
                    _line_effective_height(*previous),
                    _line_effective_height(*current),
                ),
            )
            for previous, current in zip(local_geometry, local_geometry[1:])
        ):
            continue
        merged_rows.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [
        line for line in lines if line.source_index not in consumed_source_indices
    ]
    output.extend(merged_rows)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _is_dense_same_font_two_run_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> bool:
    """检查两个普通文本 run 是否为同字体且占用充分的完整视觉行。"""

    if (
        len(members) != 2
        or not all(member.split_from_row for member in members)
        or any(member.semantic_type is not None for member in members)
        or any(member.font_signature is None for member in members)
        or any(member.font_coverage < 0.75 for member in members)
    ):
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != [0, 1]:
        return False
    if ordered[0].font_signature != ordered[1].font_signature:
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(
        key=lambda item: (item[1][0], item[1][1], item[0].source_index)
    )
    first, second = local_geometry
    pair_height = max(
        _line_effective_height(*first),
        _line_effective_height(*second),
    )
    if not _same_baseline_geometry(
        first[1],
        _line_effective_height(*first),
        second[1],
        _line_effective_height(*second),
        maximum_gap=3.0 * pair_height,
    ):
        return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(
        bbox[2] - bbox[0] for _member, bbox in local_geometry
    )
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= 0.85


def _can_restore_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
    lane_keys: dict[int, tuple[int, int]],
) -> bool:
    """检查 hard-split run 是否构成同栏、同字体且占用充分的完整视觉行。"""

    if len(members) < 3 or not all(member.split_from_row for member in members):
        return False
    if len({member.semantic_type for member in members}) != 1:
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != list(range(len(ordered))):
        return False
    member_lane_keys = {lane_keys.get(member.source_index) for member in ordered}
    if None in member_lane_keys or len(member_lane_keys) != 1:
        return False
    font_signatures = {member.font_signature for member in ordered}
    if len(font_signatures) != 1:
        return False
    if any(
        _bbox_intersects(member.bbox, table_bbox)
        for member in ordered
        for table_bbox in table_bboxes
    ):
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    heights = [_line_effective_height(member, bbox) for member, bbox in local_geometry]
    median_height = statistics.median(heights)
    glyph_widths = [
        width
        for member, _bbox in local_geometry
        if (width := _median_native_glyph_width(member, page_size)) is not None
    ]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.0
    gap_limit = max(12.0, 1.75 * median_height, 3.0 * median_glyph_width)
    for previous, current in zip(local_geometry, local_geometry[1:]):
        if not _same_baseline_geometry(
            previous[1],
            _line_effective_height(*previous),
            current[1],
            _line_effective_height(*current),
            maximum_gap=gap_limit,
        ):
            return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= 0.65


def _merge_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序恢复密集视觉行，正净空使用单空格，重叠片段直接连接。"""

    ordered_geometry = sorted(
        (
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ),
        key=lambda item: (item[1][0], item[1][1], item[0].source_index),
    )
    content_parts = [ordered_geometry[0][0].text.strip()]
    for previous, current in zip(ordered_geometry, ordered_geometry[1:]):
        separator = "" if current[1][0] <= previous[1][2] else " "
        content_parts.extend([separator, current[0].text.strip()])

    ordered_members = [member for member, _bbox in ordered_geometry]
    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in ordered_members]),
        angle=ordered_members[0].angle,
        source_index=min(member.source_index for member in ordered_members),
        chars=[char for member in ordered_members for char in member.chars],
        visual_row_id=ordered_members[0].visual_row_id,
        run_index=0,
        effective_height=statistics.median(
            _line_effective_height(member, bbox)
            for member, bbox in ordered_geometry
        ),
        font_signature=ordered_members[0].font_signature,
        font_coverage=min(member.font_coverage for member in ordered_members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight
            for member in ordered_members
            if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in ordered_members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width
            for member in ordered_members
            if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in ordered_members)
        else None,
        split_from_row=False,
        semantic_type=ordered_members[0].semantic_type,
    )
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _build_hanging_indent_group_map(
    lane: _TextLane,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> dict[int, int]:
    """仅按重复的左突首行和稳定续行缩进识别悬挂缩进条目。"""

    if lane.is_span or len(lane.lines) < 4:
        return {}
    rows = sorted(
        (item for item in lane.lines if item[0].semantic_type is None),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if len(rows) < 4:
        return {}
    median_height = statistics.median(
        _line_effective_height(line, bbox) for line, bbox in rows
    )
    start_tolerance = max(5.0, 0.65 * median_height)
    minimum_indent = max(7.0, 0.8 * median_height)
    continuation_tolerance = max(4.0, 0.55 * median_height)

    def rows_are_adjacent(
        previous: tuple[_LineItem, BBox],
        current: tuple[_LineItem, BBox],
    ) -> bool:
        """检查相邻行的净空和几何障碍是否允许组成同一缩进序列。"""

        effective_gap = _effective_text_row_gap(previous, current)
        if not -0.6 * median_height <= effective_gap <= 1.3 * median_height:
            return False
        if _connection_crosses_table(
            previous[0].bbox,
            current[0].bbox,
            table_bboxes,
        ):
            return False
        return not _horizontal_rule_separates_rows(
            previous[1],
            current[1],
            lane,
            axis_lines,
        )

    def consume_entry(
        start_index: int,
        start_left: float,
        expected_continuation_left: float | None,
        *,
        require_next_start: bool,
    ) -> tuple[int, float] | None:
        """消费一个左突首行及其续行，并返回下一条首行位置。"""

        continuation_index = start_index + 1
        if continuation_index >= len(rows):
            return None
        first_continuation = rows[continuation_index]
        if not rows_are_adjacent(rows[start_index], first_continuation):
            return None
        continuation_left = first_continuation[1][0]
        if continuation_left < start_left + minimum_indent:
            return None
        if (
            expected_continuation_left is not None
            and abs(continuation_left - expected_continuation_left) > continuation_tolerance
        ):
            return None

        continuation_index += 1
        while continuation_index < len(rows):
            previous = rows[continuation_index - 1]
            current = rows[continuation_index]
            current_left = current[1][0]
            if not rows_are_adjacent(previous, current):
                break
            if current_left < start_left + minimum_indent:
                break
            if abs(current_left - continuation_left) > continuation_tolerance:
                break
            continuation_index += 1

        if not require_next_start:
            return continuation_index, continuation_left
        if continuation_index >= len(rows):
            return None
        if not rows_are_adjacent(rows[continuation_index - 1], rows[continuation_index]):
            return None
        if abs(rows[continuation_index][1][0] - start_left) > start_tolerance:
            return None
        return continuation_index, continuation_left

    group_map: dict[int, int] = {}
    group_index = 0
    row_index = 0
    while row_index < len(rows) - 3:
        start_left = rows[row_index][1][0]
        first_entry = consume_entry(
            row_index,
            start_left,
            None,
            require_next_start=True,
        )
        if first_entry is None:
            row_index += 1
            continue

        next_start_index, continuation_left = first_entry
        start_indices = [row_index, next_start_index]
        current_start_index = next_start_index
        while True:
            next_entry = consume_entry(
                current_start_index,
                start_left,
                continuation_left,
                require_next_start=True,
            )
            if next_entry is None:
                break
            next_start_index, _continuation_left = next_entry
            start_indices.append(next_start_index)
            current_start_index = next_start_index

        final_entry = consume_entry(
            start_indices[-1],
            start_left,
            continuation_left,
            require_next_start=False,
        )
        if final_entry is None:
            row_index += 1
            continue
        end_index, _continuation_left = final_entry

        entry_ranges = [
            (start, end)
            for start, end in zip(
                start_indices,
                [*start_indices[1:], end_index],
                strict=True,
            )
        ]
        for start, end in entry_ranges:
            for line, _bbox in rows[start:end]:
                group_map[line.source_index] = group_index
            group_index += 1
        row_index = end_index

    return group_map


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    drawing_lines: list[_AxisLine] | None = None,
) -> list[dict[str, Any]]:
    """按类型屏障、局部栏带和自然段边界聚合剩余文本行。"""

    blocks: list[dict[str, Any]] = []
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        local_axis_lines = _transform_axis_lines(drawing_lines or [], page_size, angle)

        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            if not lane.lines:
                continue
            regular_gap, gap_mad = _estimate_lane_gap(lane)
            hanging_indent_groups = _build_hanging_indent_group_map(
                lane,
                table_bboxes,
                local_axis_lines,
            )
            component: list[tuple[_LineItem, BBox]] = [lane.lines[0]]
            components: list[list[tuple[_LineItem, BBox]]] = []
            for previous, current in zip(lane.lines, lane.lines[1:]):
                previous_type = previous[0].semantic_type
                current_type = current[0].semantic_type
                if previous_type != current_type:
                    should_connect = False
                elif previous_type is not None:
                    should_connect = _should_connect_semantic_rows(
                        previous,
                        current,
                        lane,
                        regular_gap,
                        table_bboxes,
                        local_axis_lines,
                    )
                else:
                    previous_group = hanging_indent_groups.get(previous[0].source_index)
                    current_group = hanging_indent_groups.get(current[0].source_index)
                    if previous_group is not None or current_group is not None:
                        should_connect = previous_group is not None and previous_group == current_group
                    else:
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                if should_connect:
                    component.append(current)
                else:
                    components.append(component)
                    component = [current]
            components.append(component)

            for component_geometry in components:
                component_lines = [item[0] for item in component_geometry]
                content = _merge_text_line_content([line.text for line in component_lines])
                if not content:
                    continue
                visual_row_ids = {
                    line.visual_row_id for line in component_lines if line.visual_row_id is not None
                }
                single_run_row_id = (
                    component_lines[0].visual_row_id
                    if len(component_lines) == 1
                    and component_lines[0].split_from_row
                    and component_lines[0].visual_row_id is not None
                    else None
                )
                blocks.append(
                    {
                        "type": component_lines[0].semantic_type or "text",
                        "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                        "angle": angle,
                        "content": content,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                    }
                )
    return blocks


def _should_connect_semantic_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """只用几何、字体和障碍连接同类型语义行，避免标题内容影响聚合。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    if previous_line.semantic_type != current_line.semantic_type:
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
        return False
    vertical_gap = _effective_text_row_gap(previous, current)
    if not -0.25 * pair_height <= vertical_gap <= max(1.25 * pair_height, regular_gap + 0.75 * pair_height):
        return False
    if (
        previous_line.semantic_type == "paragraph_title"
        and vertical_gap > 0.65 * pair_height
    ):
        return False
    if (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
    ):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    centered_pair = (
        abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) <= 0.15 * lane_width
    )
    aligned_pair = abs(previous_bbox[0] - current_bbox[0]) <= 0.75 * pair_height
    overlapping_pair = _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") >= 0.35
    return centered_pair or aligned_pair or overlapping_pair


def _line_effective_height(line: _LineItem, local_bbox: BBox) -> float:
    """返回字符统计得到的有效行高，缺失时回退到局部 bbox 高度。"""

    return max(0.1, line.effective_height or (local_bbox[3] - local_bbox[1]))


def _effective_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """按前一行顶边与有效行高计算净空，避免高数学字形拉长 bbox 底边。"""

    previous_line, previous_bbox = previous
    _current_line, current_bbox = current
    return current_bbox[1] - (previous_bbox[1] + _line_effective_height(previous_line, previous_bbox))


def _infer_text_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
) -> list[_TextLane]:
    """从重复左右边缘推断稳定栏带，并把跨栏行放入独立 span lane。"""

    anchor_tolerance = max(3.0, 0.75 * median_height)
    regular_lines = [
        item
        for item in line_geometry
        if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.15 * local_page_width)
    ]
    left_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(regular_lines, key=lambda value: value[1][0]):
        if not left_clusters:
            left_clusters.append([item])
            continue
        cluster_left = statistics.median(member[1][0] for member in left_clusters[-1])
        if abs(item[1][0] - cluster_left) <= anchor_tolerance:
            left_clusters[-1].append(item)
        else:
            left_clusters.append([item])

    supported_intervals = [
        (
            statistics.median(item[1][0] for item in cluster),
            statistics.median(item[1][2] for item in cluster),
            len(cluster),
        )
        for cluster in left_clusters
        if len(cluster) >= 3
    ]
    supported_intervals.sort(key=lambda interval: interval[0])
    filtered_intervals: list[tuple[float, float, int]] = []
    for interval in supported_intervals:
        if not filtered_intervals:
            filtered_intervals.append(interval)
            continue
        previous = filtered_intervals[-1]
        minimum_gutter = max(6.0, 0.75 * median_height)
        if interval[0] - previous[1] >= minimum_gutter:
            filtered_intervals.append(interval)
        elif interval[2] > previous[2]:
            filtered_intervals[-1] = interval

    if not filtered_intervals:
        source = regular_lines or line_geometry
        filtered_intervals = [
            (
                min(item[1][0] for item in source),
                max(item[1][2] for item in source),
                len(source),
            )
        ]

    lanes = [_TextLane(left=left, right=right) for left, right, _support in filtered_intervals]
    span_lines: list[tuple[_LineItem, BBox]] = []
    for item in line_geometry:
        bbox = item[1]
        line_width = max(0.1, bbox[2] - bbox[0])
        scored_lanes = [
            (
                max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) / line_width,
                lane,
            )
            for lane in lanes
        ]
        coverage_scores = sorted(
            (coverage for coverage, _lane in scored_lanes),
            reverse=True,
        )
        # 同时覆盖两个稳定正文栏的短尾行仍属于跨栏内容，不能按中心点落回单栏。
        if len(coverage_scores) > 1 and coverage_scores[1] >= 0.2:
            span_lines.append(item)
            continue
        best_coverage, best_lane = max(scored_lanes, key=lambda value: value[0])
        if len(lanes) == 1 or (
            best_coverage >= 0.5
            and best_lane.left - anchor_tolerance
            <= _bbox_center_x(bbox)
            <= best_lane.right + anchor_tolerance
        ):
            best_lane.lines.append(item)
        else:
            span_lines.append(item)

    if span_lines:
        lanes.append(
            _TextLane(
                left=min(item[1][0] for item in span_lines),
                right=max(item[1][2] for item in span_lines),
                lines=span_lines,
                is_span=True,
            )
        )
    _reattach_span_lane_continuations(lanes, median_height)
    return lanes


def _reattach_span_lane_continuations(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把紧随稳定跨栏多行之后的单栏宽短尾行迁回对应 span lane。"""

    regular_lanes = [lane for lane in lanes if not lane.is_span]
    span_lanes = [lane for lane in lanes if lane.is_span]
    if len(regular_lanes) < 2 or not span_lanes:
        return

    for span_lane in span_lanes:
        span_lane.lines.sort(
            key=lambda item: (item[1][1], item[1][0], item[0].source_index)
        )
        if len(span_lane.lines) < 2:
            continue
        previous, last = span_lane.lines[-2:]
        previous_height = _line_effective_height(*previous)
        last_height = _line_effective_height(*last)
        if (
            previous[0].semantic_type != last[0].semantic_type
            or abs(previous[1][0] - last[1][0]) > 0.75 * median_height
            or max(previous_height, last_height) / min(previous_height, last_height) > 1.35
            or not _title_fonts_compatible(previous[0], last[0])
            or not -0.25 * median_height
            <= _effective_text_row_gap(previous, last)
            <= 0.75 * median_height
        ):
            continue

        while True:
            candidates: list[
                tuple[
                    float,
                    float,
                    _TextLane,
                    tuple[_LineItem, BBox],
                ]
            ] = []
            for regular_lane in regular_lanes:
                for candidate in regular_lane.lines:
                    candidate_line, candidate_bbox = candidate
                    if (
                        candidate_line.semantic_type != last[0].semantic_type
                        or candidate_bbox[1] <= last[1][1]
                    ):
                        continue
                    gap = _effective_text_row_gap(last, candidate)
                    candidate_height = _line_effective_height(*candidate)
                    if (
                        not -0.25 * median_height <= gap <= 0.75 * median_height
                        or abs(candidate_bbox[0] - last[1][0]) > 0.75 * median_height
                        or max(last_height, candidate_height)
                        / min(last_height, candidate_height)
                        > 1.35
                        or not _title_fonts_compatible(last[0], candidate_line)
                    ):
                        continue
                    has_parallel_peer = any(
                        other_line is not candidate_line
                        and _bbox_axis_overlap_ratio(
                            candidate_bbox,
                            other_bbox,
                            axis="y",
                        )
                        >= 0.5
                        for lane in regular_lanes
                        for other_line, other_bbox in lane.lines
                    )
                    if has_parallel_peer:
                        continue
                    candidates.append(
                        (
                            max(0.0, gap),
                            candidate_bbox[1],
                            regular_lane,
                            candidate,
                        )
                    )
            if not candidates:
                break
            _gap, _top, regular_lane, candidate = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            regular_lane.lines.remove(candidate)
            span_lane.lines.append(candidate)
            span_lane.left = min(span_lane.left, candidate[1][0])
            span_lane.right = max(span_lane.right, candidate[1][2])
            span_lane.lines.sort(
                key=lambda item: (item[1][1], item[1][0], item[0].source_index)
            )
            last = candidate
            last_height = _line_effective_height(*last)


def _estimate_lane_gap(lane: _TextLane) -> tuple[float, float]:
    """从栏带内兼容相邻行的较小间隙簇估计常规净空和 MAD。"""

    lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    heights = [_line_effective_height(line, bbox) for line, bbox in lane.lines]
    median_height = statistics.median(heights) if heights else 1.0
    gaps: list[float] = []
    for previous, current in zip(lane.lines, lane.lines[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if previous_line.visual_row_id == current_line.visual_row_id and (
            previous_line.split_from_row or current_line.split_from_row
        ):
            continue
        previous_height = _line_effective_height(previous_line, previous_bbox)
        current_height = _line_effective_height(current_line, current_bbox)
        if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
            continue
        pair_height = max(previous_height, current_height)
        gap = _effective_text_row_gap(previous, current)
        if gap < -0.25 * pair_height or gap > 2.0 * pair_height:
            continue
        if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
            previous_bbox[0] - current_bbox[0]
        ) > 1.5 * median_height:
            continue
        # PDF 字符框常在相邻基线间产生极小重叠；按零净空计入常规行距统计。
        gaps.append(max(0.0, gap))

    if not gaps:
        return 0.35 * median_height, 0.0
    sorted_gaps = sorted(gaps)
    lower_count = max(1, math.ceil(len(sorted_gaps) * 0.6))
    lower_gaps = sorted_gaps[:lower_count]
    regular_gap = statistics.median(lower_gaps)
    gap_mad = statistics.median(abs(gap - regular_gap) for gap in lower_gaps)
    return regular_gap, gap_mad


def _should_connect_text_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """综合局部间距、首行缩进、字体和障碍判断两个相邻视觉行是否同段。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    lane_width = max(0.1, lane.right - lane.left)
    previous_width = previous_bbox[2] - previous_bbox[0]
    current_width = current_bbox[2] - current_bbox[0]
    vertical_gap = _effective_text_row_gap(previous, current)
    if previous_line.visual_row_id == current_line.visual_row_id and (
        previous_line.split_from_row or current_line.split_from_row
    ):
        return False
    if (
        current_height < 0.88 * previous_height
        and vertical_gap > regular_gap + 0.25 * previous_height
    ):
        return False
    height_ratio = max(previous_height, current_height) / min(previous_height, current_height)
    if height_ratio > 1.35:
        both_fill_lane = previous_width >= 0.8 * lane_width and current_width >= 0.8 * lane_width
        aligned_left_edges = abs(previous_bbox[0] - current_bbox[0]) <= 0.5 * pair_height
        font_style_changed = (
            previous_line.font_signature is not None
            and current_line.font_signature is not None
            and previous_line.font_signature[1] != current_line.font_signature[1]
        )
        # 满栏混合字体可跨字号续接，但显式正体/斜体等样式边界仍保持原分段语义。
        if not both_fill_lane or not aligned_left_edges or font_style_changed:
            return False

    if vertical_gap < -0.25 * pair_height:
        return False
    if _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5 and abs(
        previous_bbox[0] - current_bbox[0]
    ) > 1.5 * pair_height:
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False

    gap_limit = max(
        regular_gap + max(0.5 * pair_height, 3.0 * gap_mad),
        1.1 * pair_height,
    )
    # 排版断词可以跳过缩进、字体和短行规则，但仍须限制在邻近物理行内，
    # 避免页内远距离的 “cross-” 与后续标题被误拼为同一段。
    if is_hyphen_at_line_end(previous_line.text):
        return vertical_gap <= max(gap_limit, 1.8 * pair_height)
    if vertical_gap > gap_limit:
        return False

    terminal_previous = bool(re.search(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$", previous_line.text.rstrip()))
    if terminal_previous and vertical_gap > regular_gap + 0.5 * pair_height:
        return False

    # 局部版心可能比整栏推断边界更靠左，缩进需同时参考上一物理行。
    local_lane_left = min(lane.left, previous_bbox[0])
    local_lane_width = max(0.1, lane.right - local_lane_left)
    next_indent = current_bbox[0] - local_lane_left
    previous_fill = max(0.0, previous_bbox[2] - local_lane_left) / local_lane_width
    if next_indent >= max(5.0, 0.65 * pair_height) and (previous_fill <= 0.8 or terminal_previous):
        return False

    abnormal_gap = vertical_gap > regular_gap + 0.25 * pair_height
    if (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
        and (abnormal_gap or min(previous_width, current_width) <= 0.7 * lane_width)
    ):
        return False
    if abnormal_gap and min(previous_width, current_width) <= 0.65 * lane_width:
        return False
    return True


def _horizontal_rule_separates_rows(
    previous_bbox: BBox,
    current_bbox: BBox,
    lane: _TextLane,
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """检查两个相邻文本行之间是否存在覆盖当前栏带的长水平规则线。"""

    if current_bbox[1] <= previous_bbox[3]:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    for axis_line in axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        line_y = _bbox_center_y(axis_line.bbox)
        if not previous_bbox[3] <= line_y <= current_bbox[1]:
            continue
        overlap = max(0.0, min(axis_line.bbox[2], lane.right) - max(axis_line.bbox[0], lane.left))
        if overlap / lane_width >= 0.6:
            return True
    return False


def _merge_text_line_content(line_texts: Sequence[str]) -> str:
    """按 Hybrid 语言与行末连字规则折叠普通文本行。"""

    normalized_lines = [_normalize_native_run_text(str(text or "")) for text in line_texts]
    normalized_lines = [text for text in normalized_lines if text]
    if not normalized_lines:
        return ""
    try:
        block_language = detect_lang("".join(normalized_lines))
    except Exception:
        block_language = ""
    content_parts = [normalized_lines[0]]
    for current_line in normalized_lines[1:]:
        processed_previous, separator = resolve_text_line_boundary(
            content_parts[-1],
            block_language=block_language,
            next_starts_with_lowercase=current_line[0].islower(),
        )
        content_parts[-1] = processed_previous
        content_parts.extend([separator, current_line])
    return "".join(content_parts).strip()


def _connection_crosses_table(
    first_bbox: BBox,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """检查两行中心连接区域是否穿过已确认表格。"""

    first_center = (_bbox_center_x(first_bbox), _bbox_center_y(first_bbox))
    second_center = (_bbox_center_x(second_bbox), _bbox_center_y(second_bbox))
    connector = _coerce_bbox(
        (
            min(first_center[0], second_center[0]) - 0.1,
            min(first_center[1], second_center[1]) - 0.1,
            max(first_center[0], second_center[0]) + 0.1,
            max(first_center[1], second_center[1]) + 0.1,
        )
    )
    return connector is not None and any(_bbox_intersects(connector, table_bbox) for table_bbox in table_bboxes)


def _sort_blocks_with_visual_row_groups(
    blocks: list[dict[str, Any]],
    page_size: tuple[float, float],
) -> list[dict[str, Any]]:
    """把同一粗行拆出的单行 block 包装成虚拟项排序，再按局部 x 顺序展开。"""

    grouped_indices: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        row_id = block.get("_single_run_row_id")
        if isinstance(row_id, int):
            grouped_indices.setdefault(row_id, []).append(index)

    virtual_groups: dict[int, dict[str, Any]] = {}
    consumed_indices: set[int] = set()
    for row_id, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        members = [blocks[index] for index in indices]
        virtual_group = {
            "type": "_xycut_visual_row_group",
            "bbox": _bbox_union_many([member["bbox"] for member in members]),
            "angle": members[0].get("angle", 0),
            "content": "",
            "_members": members,
        }
        virtual_groups[row_id] = virtual_group
        consumed_indices.update(indices)

    sortable_blocks = [block for index, block in enumerate(blocks) if index not in consumed_indices]
    sortable_blocks.extend(virtual_groups.values())
    sorted_payloads = sort_entries(sortable_blocks)
    output: list[dict[str, Any]] = []
    for payload in sorted_payloads:
        members = payload.get("_members")
        if not isinstance(members, list):
            output.append(payload)
            continue
        angle = int(payload.get("angle", 0) or 0) % 360
        members.sort(
            key=lambda member: (
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[0],
                _rotate_bbox_to_upright(member["bbox"], page_size, angle)[1],
            )
        )
        output.extend(members)
    return output


def _normalize_output_block(
    block: dict[str, Any],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """在排序完成后将绝对 bbox 裁剪并归一化为 model_list 坐标。"""

    page_width, page_height = page_size
    bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
    if bbox is None or page_width <= 0 or page_height <= 0:
        return None
    content = block.get("content")
    if not isinstance(content, str):
        return None
    content = _sanitize_pdf_control_text(content, preserve_newlines=True)
    block_type = block.get("type")
    normalized_type = block_type if block_type in _OUTPUT_BLOCK_TYPES else "text"
    if normalized_type != "image" and not content.strip():
        return None
    normalized_bbox = _normalize_bbox_to_thousandths(bbox, page_size)
    return {
        "type": normalized_type,
        "bbox": normalized_bbox,
        "angle": 0 if normalized_type == "image" else int(block.get("angle", 0) or 0) % 360,
        "content": content,
    }


def _normalize_bbox_to_thousandths(
    bbox: BBox,
    page_size: tuple[float, float],
) -> list[float]:
    """将绝对 bbox 转成千分位刻度，并保证舍入后的宽高至少各占一个刻度。"""

    page_width, page_height = page_size
    ticks = [
        max(0, min(1000, int(round(bbox[0] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[1] / page_height * 1000)))),
        max(0, min(1000, int(round(bbox[2] / page_width * 1000)))),
        max(0, min(1000, int(round(bbox[3] / page_height * 1000)))),
    ]
    if ticks[2] <= ticks[0]:
        if ticks[0] < 1000:
            ticks[2] = ticks[0] + 1
        else:
            ticks[0] = max(0, ticks[2] - 1)
    if ticks[3] <= ticks[1]:
        if ticks[1] < 1000:
            ticks[3] = ticks[1] + 1
        else:
            ticks[1] = max(0, ticks[3] - 1)
    return [tick / 1000 for tick in ticks]


def _rotate_bbox_to_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将页面 bbox 转到当前文本方向的正向局部坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (page_height - y1, x0, page_height - y0, x1)
    if angle == 90:
        return (y0, page_width - x1, y1, page_width - x0)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _rotate_bbox_from_upright(
    bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
) -> BBox:
    """将正向局部 bbox 逆变换回 PDF 页面坐标。"""

    page_width, page_height = page_size
    x0, y0, x1, y1 = bbox
    if angle == 270:
        return (y0, page_height - x1, y1, page_height - x0)
    if angle == 90:
        return (page_width - y1, x0, page_width - y0, x1)
    if angle == 180:
        return (page_width - x1, page_height - y1, page_width - x0, page_height - y0)
    return bbox


def _median_fragment_height(fragments: list[_Fragment]) -> float:
    """返回正向文本片段高度的中位数。"""

    heights = [
        fragment.local_bbox[3] - fragment.local_bbox[1]
        for fragment in fragments
        if fragment.local_bbox[3] > fragment.local_bbox[1]
    ]
    return max(0.1, float(statistics.median(heights)) if heights else 1.0)


def _coerce_bbox(value: Any) -> BBox | None:
    """将任意四元 bbox 规范成非退化浮点坐标。"""

    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if not all(math.isfinite(item) for item in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _clip_bbox(
    bbox: BBox | None,
    page_size: tuple[float, float],
) -> BBox | None:
    """将 bbox 裁剪到页面范围，退化框返回 None。"""

    if bbox is None:
        return None
    page_width, page_height = page_size
    return _coerce_bbox(
        (
            max(0.0, min(page_width, bbox[0])),
            max(0.0, min(page_height, bbox[1])),
            max(0.0, min(page_width, bbox[2])),
            max(0.0, min(page_height, bbox[3])),
        )
    )


def _bbox_union(first: BBox, second: BBox) -> BBox:
    """返回两个 bbox 的外接并集框。"""

    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bbox_union_many(bboxes: Sequence[BBox]) -> BBox:
    """返回非空 bbox 序列的外接并集框。"""

    if not bboxes:
        raise ValueError("bbox sequence must not be empty")
    result = bboxes[0]
    for bbox in bboxes[1:]:
        result = _bbox_union(result, bbox)
    return result


def _expand_bbox(bbox: BBox, margin: float) -> BBox:
    """向四周扩展 bbox，仅供几何容差判定使用。"""

    return (
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin,
    )


def _bbox_center_x(bbox: BBox) -> float:
    """返回 bbox 的水平中心。"""

    return (bbox[0] + bbox[2]) / 2.0


def _bbox_center_y(bbox: BBox) -> float:
    """返回 bbox 的垂直中心。"""

    return (bbox[1] + bbox[3]) / 2.0


def _bbox_intersects(first: BBox, second: BBox) -> bool:
    """检查两个 bbox 是否存在正面积交叠。"""

    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def _bbox_distance(first: BBox, second: BBox) -> float:
    """返回两个 bbox 的欧氏净空距离，交叠或相接时为零。"""

    horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _bbox_overlap_in_smaller(first: BBox, second: BBox) -> float:
    """计算交集面积占较小 bbox 面积的比例。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _bbox_axis_overlap_ratio(
    first: BBox,
    second: BBox,
    *,
    axis: Literal["x", "y"],
) -> float:
    """计算指定轴上的交叠长度占较短轴长的比例。"""

    if axis == "x":
        first_start, first_end = first[0], first[2]
        second_start, second_end = second[0], second[2]
    else:
        first_start, first_end = first[1], first[3]
        second_start, second_end = second[1], second[3]
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    shorter = min(first_end - first_start, second_end - second_start)
    return overlap / shorter if shorter > 0 else 0.0


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    """检查点是否位于 bbox 内部或边界上。"""

    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def doc_analyze(
    pdf_bytes: bytes,
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    page_index_map: list[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """使用 Flash 提取原生文本，OCR 文档则委托 Hybrid low。"""

    if parse_mode not in {"auto", "txt", "ocr"}:
        raise ValueError(f"parse_mode {parse_mode} is not supported")

    with PDFDocument(pdf_bytes) as pdf_doc:
        page_count = pdf_doc.page_count
        if page_index_map and len(page_index_map) != page_count:
            raise ValueError(
                f"Flash page_index_map length mismatch: page_count={page_count}, page_index_map={len(page_index_map)}"
            )

        resolved_mode: Literal["txt", "ocr"]
        if parse_mode == "auto":
            resolved_mode = pdf_doc.classify()
        else:
            resolved_mode = parse_mode

        if resolved_mode == "txt":
            return _analyze_native_document(pdf_doc)

    # OCR 路径延迟加载 Hybrid，保证原生 Flash 不引入本地视觉模型运行时。
    from mineru.backend.hybrid.analyze import doc_analyze as hybrid_doc_analyze

    _middle_json, model_list = hybrid_doc_analyze(
        pdf_bytes,
        effort="low",
        parse_mode="ocr",
        page_index_map=page_index_map,
    )
    return model_list


if __name__ == '__main__':
    # pdf_path = "/Users/myhloli/pdf/截断合并/demo1-3.pdf"
    # pdf_path = "/Users/myhloli/pdf/png/seal4.png"  # shubiao.png
    pdf_path = "/Users/myhloli/pdf/demo1.pdf"
    pdf_bytes = read_fn(pdf_path)
    model_list = doc_analyze(pdf_bytes, parse_mode="auto")
    logger.info(f"model_list: {model_list}")
