# Copyright (c) Opendatalab. All rights reserved.

"""为 Flash 原生文本生成 loose/tight/origin 协商后的 canonical 几何。"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Sequence, TypeAlias

from ....types import BBox
from .document import PDFPageTextGeometry
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _rotate_origin_to_upright,
)
from .models import _LineItem


X_RELIABLE_PAIR_MIN = 30
X_STRONG_MEDIAN_RATIO = 1.30
X_STRONG_RATIO_THRESHOLD = 1.35
X_STRONG_RATIO_SHARE = 0.30
X_STRONG_NEXT_TIGHT_OVERLAP_SHARE = 0.30
X_SIBLING_PAIR_MIN = 10
X_SIBLING_MEDIAN_RATIO = 1.15
X_SIBLING_NEXT_TIGHT_OVERLAP_SHARE = 0.50
X_SIBLING_P90_RATIO = 1.50

ANCHOR_MIN_TIGHT_HEIGHT_RATIO = 0.38
Y_MIN_ANCHOR_COUNT = 4
Y_MIN_GEOMETRY_COVERAGE = 0.80
Y_DOMINANT_ROW_SHARE = 0.80
Y_NEIGHBOR_CORE_INTRUSION = 0.25
Y_MIN_REPEATED_LINES = 3
Y_MIN_REPEATED_SHARE = 0.50
Y_HEALTHY_LOOSE_SAMPLE_MIN = 20
Y_DOCUMENT_RISK_P95_RATIO = 2.20
STYLE_INFLATION_LOOSE_FONT_RATIO = 1.50
STYLE_INFLATION_LOOSE_TIGHT_RATIO = 1.80
STYLE_INFLATION_MIN_LINE_COUNT = 3
STYLE_INFLATION_MIN_LINE_SHARE = 0.50
STYLE_INFLATION_MIN_PAGE_COUNT = 2
STYLE_INFLATION_MIN_SCALE = 4.0
STYLE_TIER_MIN_MEMBER_COUNT = 2
STYLE_TIER_MIN_MEMBER_SHARE = 0.15
STYLE_TIER_GAP_RATIO = 1.35

RunKey: TypeAlias = tuple[str, float, int, int, int, str]
LineKey: TypeAlias = tuple[int, int]
CharKey: TypeAlias = tuple[int, int]
LooseTierSample: TypeAlias = tuple[float, float, float, float]


@dataclass(slots=True)
class CharLayoutGeometry:
    """保存一个实际需要修复的字符几何及分轴状态。"""

    source_bbox: BBox
    tight_bbox: BBox
    origin: tuple[float, float]
    layout_bbox: BBox
    ink_bbox: BBox
    baseline: float
    advance: float | None
    em_height: float
    x_state: Literal["healthy", "abnormal", "unknown"]
    y_state: Literal["healthy", "abnormal", "unknown"]
    confidence: float


@dataclass(slots=True)
class LineGeometryRepair:
    """保存一条 legacy line 的 canonical 修复和 shadow 诊断。"""

    source_bbox: BBox
    layout_bbox: BBox
    ink_bbox: BBox | None
    baseline: float | None
    em_height: float
    state: Literal["healthy", "repair_x", "trim_y", "repair_xy", "uncertain"] = "healthy"
    confidence: float = 1.0
    split_y_candidate: bool = False
    repaired_char_count: int = 0
    y_intrusion_ratio: float = 0.0
    run_key: RunKey | None = None


@dataclass(slots=True)
class DocumentGeometryPlan:
    """保存文档级 run 结论、局部字符修复和逐行 canonical 几何。"""

    char_repairs: dict[CharKey, CharLayoutGeometry] = field(default_factory=dict)
    line_repairs: dict[LineKey, LineGeometryRepair] = field(default_factory=dict)
    line_style_scales: dict[LineKey, float] = field(default_factory=dict)
    line_ink_bboxes: dict[LineKey, BBox] = field(default_factory=dict)
    line_baselines: dict[LineKey, float] = field(default_factory=dict)
    style_inflated_runs: set[RunKey] = field(default_factory=set)
    run_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    document_style_anomaly: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为 review 脚本可序列化的稳定诊断。"""

        return {
            "document_style_anomaly": self.document_style_anomaly,
            "run_diagnostics": self.run_diagnostics,
            "char_repairs": [
                {
                    "page_index": page_index,
                    "char_idx": char_idx,
                    "source_bbox": list(repair.source_bbox),
                    "tight_bbox": list(repair.tight_bbox),
                    "origin": list(repair.origin),
                    "layout_bbox": list(repair.layout_bbox),
                    "advance": repair.advance,
                    "em_height": repair.em_height,
                    "x_state": repair.x_state,
                    "y_state": repair.y_state,
                    "confidence": repair.confidence,
                }
                for (page_index, char_idx), repair in sorted(self.char_repairs.items())
            ],
            "line_repairs": [
                {
                    "page_index": page_index,
                    "source_index": source_index,
                    "source_bbox": list(repair.source_bbox),
                    "layout_bbox": list(repair.layout_bbox),
                    "ink_bbox": list(repair.ink_bbox) if repair.ink_bbox is not None else None,
                    "baseline": repair.baseline,
                    "em_height": repair.em_height,
                    "state": repair.state,
                    "confidence": repair.confidence,
                    "split_y_candidate": repair.split_y_candidate,
                    "repaired_char_count": repair.repaired_char_count,
                    "y_intrusion_ratio": repair.y_intrusion_ratio,
                    "run_key": list(repair.run_key) if repair.run_key is not None else None,
                }
                for (page_index, source_index), repair in sorted(self.line_repairs.items())
                if repair.state != "healthy" or repair.split_y_candidate
            ],
        }


@dataclass(frozen=True, slots=True)
class _DocumentGeometryRisk:
    """区分需要完整字符几何的布局风险与仅需字号校准的样式风险。"""

    layout: bool = False
    style: bool = False

    @property
    def any(self) -> bool:
        """返回当前文档是否需要进入完整字符样本收集。"""

        return self.layout or self.style


@dataclass(slots=True)
class _CharSample:
    """保存分析阶段使用的局部字符样本。"""

    page_index: int
    line: _LineItem
    position: int
    char_idx: int
    text: str
    source_bbox: BBox
    tight_bbox: BBox
    origin: tuple[float, float]
    local_source_bbox: BBox
    local_tight_bbox: BBox
    local_origin: tuple[float, float]
    run_key: RunKey
    font_size: float
    is_anchor: bool = False


@dataclass(slots=True)
class _RunStats:
    """保存一个字体 run 的 origin-advance 与覆盖统计。"""

    key: RunKey
    pair_ratios: list[float] = field(default_factory=list)
    pair_overlaps: list[bool] = field(default_factory=list)
    advances: list[float] = field(default_factory=list)
    tight_left_bearings: list[float] = field(default_factory=list)
    samples: list[_CharSample] = field(default_factory=list)
    strong_x_bad: bool = False
    sibling_x_bad: bool = False
    style_y_bad: bool = False
    median_advance: float | None = None
    median_tight_left_bearing: float = 0.0


@dataclass(slots=True)
class _LineAnalysis:
    """保存 Y 轴 row 聚类和邻行侵入判定所需信息。"""

    key: LineKey
    line: _LineItem
    samples: list[_CharSample]
    anchors: list[_CharSample]
    dominant: list[_CharSample]
    baseline: float
    tight_core: BBox
    local_source_bbox: BBox
    explicit_all_source_bbox: BBox
    legacy_local_bbox: BBox
    run_key: RunKey
    geometry_coverage: float
    dominant_share: float
    split_y_candidate: bool
    neighbors: list[_LineAnalysis] = field(default_factory=list)
    intrusion_ratio: float = 0.0
    y_candidate: bool = False


def _quantile(values: list[float], fraction: float) -> float:
    """返回确定性的线性插值分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _line_loose_tier_offsets(
    samples: list[LooseTierSample],
    em_height: float,
) -> tuple[float, float] | None:
    """把归一化 loose 高度分档，并返回最大异常档回缩到次档后的 ascent/descent。"""

    if len(samples) < 2 * STYLE_TIER_MIN_MEMBER_COUNT or em_height <= 0:
        return None
    normalized = sorted(
        (
            (ascent + descent) / max(font_size, tight_height, 0.1),
            ascent,
            descent,
            tight_height,
            font_size,
        )
        for ascent, descent, tight_height, font_size in samples
    )
    tiers: list[list[tuple[float, float, float, float, float]]] = []
    for item in normalized:
        if (
            not tiers
            or item[0]
            > STYLE_TIER_GAP_RATIO
            * statistics.median(member[0] for member in tiers[-1])
        ):
            tiers.append([item])
        else:
            tiers[-1].append(item)
    if len(tiers) < 2:
        return None
    upper_tier = tiers[-1]
    reference_tier = tiers[-2]
    minimum_members = max(
        STYLE_TIER_MIN_MEMBER_COUNT,
        math.ceil(STYLE_TIER_MIN_MEMBER_SHARE * len(samples)),
    )
    if (
        len(upper_tier) < minimum_members
        or len(reference_tier) < STYLE_TIER_MIN_MEMBER_COUNT
    ):
        return None
    upper_loose_heights = [member[1] + member[2] for member in upper_tier]
    upper_font_sizes = [member[4] for member in upper_tier if member[4] > 0]
    upper_tight_heights = [member[3] for member in upper_tier]
    if (
        statistics.median(upper_loose_heights)
        <= STYLE_INFLATION_LOOSE_FONT_RATIO
        * (
            statistics.median(upper_font_sizes)
            if upper_font_sizes
            else 0.0
        )
        or statistics.median(upper_loose_heights)
        <= STYLE_INFLATION_LOOSE_TIGHT_RATIO
        * max(0.1, _quantile(upper_tight_heights, 0.75))
    ):
        return None
    reference_ascent_ratios = [
        member[1] / max(member[4], member[3], 0.1)
        for member in reference_tier
    ]
    reference_descent_ratios = [
        member[2] / max(member[4], member[3], 0.1)
        for member in reference_tier
    ]
    return (
        statistics.median(reference_ascent_ratios) * em_height,
        statistics.median(reference_descent_ratios) * em_height,
    )


def _coerce_origin(value: Any) -> tuple[float, float] | None:
    """把 side map 中的 origin 收敛为有限二维坐标。"""

    try:
        origin = (float(value[0]), float(value[1]))
    except (IndexError, TypeError, ValueError):
        return None
    return origin if all(math.isfinite(item) for item in origin) else None


@lru_cache(maxsize=512)
def _normalized_font_family(name: str) -> str:
    """移除 PDF 子集前缀并归一化字体族名称。"""

    value = re.sub(r"^[A-Z]{6}\+", "", name)
    return re.sub(r"[\s_-]+", "", value).casefold() or "<unknown>"


@lru_cache(maxsize=8192)
def _script_group(text: str) -> str:
    """把字符归入字体 run 使用的宽粒度文字类别。"""

    if text.isascii() and text.isalpha():
        return "latin"
    if text.isdigit():
        return "digit"
    codepoint = ord(text[0]) if text else 0
    if (
        0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    ):
        return "cjk"
    category = unicodedata.category(text[0]) if text else "Cn"
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    return "other"


def _font_run_key(char: dict[str, Any], angle: int, text: str) -> tuple[RunKey, float]:
    """构造字体族、字号、字重、方向和文字类别组成的 run key。"""

    font = char.get("font") or {}
    font_name = str(font.get("name") or "<unknown>")
    try:
        font_size = float(font.get("size") or 0.0)
    except (TypeError, ValueError):
        font_size = 0.0
    try:
        flags = int(font.get("flags") or 0)
    except (TypeError, ValueError):
        flags = 0
    try:
        weight = int(round(float(font.get("weight") or 0.0) / 100.0) * 100)
    except (TypeError, ValueError):
        weight = 0
    return _cached_run_key(
        font_name,
        font_size,
        flags,
        weight,
        angle,
        _script_group(text),
    ), font_size


@lru_cache(maxsize=4096)
def _cached_run_key(
    font_name: str,
    font_size: float,
    flags: int,
    weight: int,
    angle: int,
    script: str,
) -> RunKey:
    """缓存同字体样式反复出现的规范化 run key。"""

    return (
        _normalized_font_family(font_name),
        round(font_size * 4.0) / 4.0,
        flags,
        weight,
        angle,
        script,
    )


def _is_anchor_text(text: str) -> bool:
    """仅让字母、数字和 CJK 等完整字形参与主要基线统计。"""

    if not text or text.isspace() or not text.isprintable():
        return False
    category = unicodedata.category(text[0])
    return category.startswith("L") or category.startswith("N")


def _source_bbox(
    char: dict[str, Any],
    geometry: PDFPageTextGeometry,
    char_idx: int,
) -> BBox | None:
    """按提取契约读取 loose：零旋转使用 char bbox，旋转字符才接受 side-map。"""

    raw_bbox = _coerce_bbox(char.get("bbox"))
    try:
        rotation = float(char.get("rotation") or 0.0)
    except (TypeError, ValueError):
        rotation = math.nan
    if math.isfinite(rotation) and abs(rotation) <= 1e-9:
        return raw_bbox
    side_bbox = _coerce_bbox(geometry.loose_bboxes.get(char_idx))
    if side_bbox is None or raw_bbox is None:
        return side_bbox or raw_bbox
    tight_bbox = _coerce_bbox(geometry.tight_bboxes.get(char_idx))
    stable_width = max(
        raw_bbox[2] - raw_bbox[0],
        (tight_bbox[2] - tight_bbox[0]) if tight_bbox is not None else 0.0,
        0.1,
    )
    if side_bbox[2] - side_bbox[0] > 1.6 * stable_width:
        return raw_bbox
    return side_bbox


def _style_line_is_inflated(
    source_height: float,
    font_sizes: Sequence[float],
    tight_heights: Sequence[float],
) -> bool:
    """按统一阈值判断一行 loose 高度是否显著偏离字号与 tight 字形。"""

    if not tight_heights:
        return False
    style_scale = max(
        statistics.median(font_sizes) if font_sizes else 0.0,
        _quantile(tight_heights, 0.75),
    )
    if style_scale < STYLE_INFLATION_MIN_SCALE:
        return False
    tight_height = max(0.1, _quantile(tight_heights, 0.75))
    return (
        source_height > STYLE_INFLATION_LOOSE_FONT_RATIO * style_scale
        and source_height > STYLE_INFLATION_LOOSE_TIGHT_RATIO * tight_height
    )


def _document_requires_full_geometry(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
) -> _DocumentGeometryRisk:
    """流式识别布局与样式风险，避免健康文档保留整本字符样本。"""

    x_ratios: dict[RunKey, list[float]] = defaultdict(list)
    x_overlaps: dict[RunKey, int] = defaultdict(int)
    y_ratios: list[float] = []
    y_extreme_runs: Counter[RunKey] = Counter()
    style_line_counts: Counter[RunKey] = Counter()
    style_inflated_lines: dict[RunKey, set[LineKey]] = defaultdict(set)
    for page_index, (lines, geometry, page_size) in enumerate(
        zip(lines_by_page, geometries, page_sizes, strict=True),
    ):
        for line in lines:
            entries: list[tuple[int, str, BBox, BBox, tuple[float, float], RunKey, float]] = []
            for position, char in enumerate(line.chars):
                text = str(char.get("char") or "")
                char_idx = char.get("char_idx")
                if not _is_anchor_text(text) or isinstance(char_idx, bool) or not isinstance(char_idx, int):
                    continue
                source = _clip_bbox(_source_bbox(char, geometry, char_idx), page_size)
                tight = _clip_bbox(_coerce_bbox(geometry.tight_bboxes.get(char_idx)), page_size)
                origin = _coerce_origin(geometry.origins.get(char_idx))
                if source is None or tight is None or origin is None:
                    continue
                run_key, font_size = _font_run_key(char, line.angle, text)
                entries.append(
                    (
                        position,
                        text,
                        _rotate_bbox_to_upright(source, page_size, line.angle),
                        _rotate_bbox_to_upright(tight, page_size, line.angle),
                        _rotate_origin_to_upright(origin, page_size, line.angle),
                        run_key,
                        font_size,
                    )
                )
            if not entries:
                continue
            height_q75 = _quantile([entry[3][3] - entry[3][1] for entry in entries], 0.75)
            anchors = [entry for entry in entries if entry[3][3] - entry[3][1] >= ANCHOR_MIN_TIGHT_HEIGHT_RATIO * height_q75]
            for current, following in zip(anchors, anchors[1:]):
                if current[5] != following[5]:
                    continue
                tight_height = max(current[3][3] - current[3][1], following[3][3] - following[3][1])
                if abs(current[4][1] - following[4][1]) > max(0.5, 0.25 * tight_height):
                    continue
                advance = following[4][0] - current[4][0]
                tight_width = max(current[3][2] - current[3][0], following[3][2] - following[3][0])
                if not 0.1 < advance <= max(5.0 * max(current[6], 1.0), 8.0 * tight_width):
                    continue
                ratio = (current[2][2] - current[2][0]) / advance
                x_ratios[current[5]].append(ratio)
                following_width = following[3][2] - following[3][0]
                if current[2][2] - following[3][0] >= 0.05 * max(following_width, 0.1):
                    x_overlaps[current[5]] += 1

            anchors_by_run: dict[RunKey, list[tuple[int, str, BBox, BBox, tuple[float, float], RunKey, float]]] = defaultdict(
                list
            )
            for entry in anchors:
                anchors_by_run[entry[5]].append(entry)
            for run_key, run_entries in anchors_by_run.items():
                style_line_counts[run_key] += 1
                if _style_line_is_inflated(
                    line.effective_height,
                    [entry[6] for entry in run_entries if entry[6] > 0],
                    [entry[3][3] - entry[3][1] for entry in run_entries],
                ):
                    style_inflated_lines[run_key].add(
                        (page_index, line.source_index),
                    )

            # 四锚点门槛只约束 Y 分析；短行仍须为文档级 X 统计贡献相邻字符对。
            if len(anchors) < Y_MIN_ANCHOR_COUNT:
                continue
            if line.angle != 0 or line.formula_candidate_only or line.restored_inline_cluster or line.compact_formula_cluster:
                continue
            baseline_entries = sorted(anchors, key=lambda entry: entry[4][1])
            tolerance = max(0.5, 0.25 * height_q75)
            clusters: list[list[tuple[int, str, BBox, BBox, tuple[float, float], RunKey, float]]] = []
            for entry in baseline_entries:
                if not clusters or abs(entry[4][1] - statistics.median(item[4][1] for item in clusters[-1])) > tolerance:
                    clusters.append([entry])
                else:
                    clusters[-1].append(entry)
            supported = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
            if len(supported) >= 2:
                return _DocumentGeometryRisk(layout=True)
            dominant = max(clusters, key=len)
            if len(dominant) / len(anchors) < Y_DOMINANT_ROW_SHARE:
                continue
            source_union = _bbox_union_many([entry[2] for entry in dominant])
            tight_union = _bbox_union_many([entry[3] for entry in dominant])
            tight_height = tight_union[3] - tight_union[1]
            if tight_height <= 0:
                continue
            ratio = (source_union[3] - source_union[1]) / tight_height
            y_ratios.append(ratio)
            if ratio >= 3.0:
                y_extreme_runs[Counter(entry[5] for entry in dominant).most_common(1)[0][0]] += 1

    layout_risk = False
    for run_key, ratios in x_ratios.items():
        pair_count = len(ratios)
        if pair_count < X_RELIABLE_PAIR_MIN:
            continue
        if (
            statistics.median(ratios) >= X_STRONG_MEDIAN_RATIO
            and sum(value > X_STRONG_RATIO_THRESHOLD for value in ratios) / pair_count >= X_STRONG_RATIO_SHARE
            and x_overlaps[run_key] / pair_count >= X_STRONG_NEXT_TIGHT_OVERLAP_SHARE
        ):
            layout_risk = True
            break
    layout_risk = (
        layout_risk
        or bool(y_ratios)
        and (
            _quantile(y_ratios, 0.95) >= Y_DOCUMENT_RISK_P95_RATIO
            or any(count >= Y_MIN_REPEATED_LINES for count in y_extreme_runs.values())
        )
    )
    style_risk = any(
        len(inflated_lines) >= STYLE_INFLATION_MIN_LINE_COUNT
        and len(inflated_lines) / style_line_counts[run_key] >= STYLE_INFLATION_MIN_LINE_SHARE
        and len({page_index for page_index, _source_index in inflated_lines}) >= STYLE_INFLATION_MIN_PAGE_COUNT
        for run_key, inflated_lines in style_inflated_lines.items()
        if style_line_counts[run_key] >= STYLE_INFLATION_MIN_LINE_COUNT
    )
    return _DocumentGeometryRisk(
        layout=layout_risk,
        style=style_risk,
    )


def _collect_samples(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
) -> tuple[list[_CharSample], dict[LineKey, list[_CharSample]]]:
    """收集具有合法 loose/tight/origin 的可见字符样本。"""

    samples: list[_CharSample] = []
    by_line: dict[LineKey, list[_CharSample]] = defaultdict(list)
    for page_index, (lines, geometry, page_size) in enumerate(zip(lines_by_page, geometries, page_sizes, strict=True)):
        for line in lines:
            for position, char in enumerate(line.chars):
                text = str(char.get("char") or "")
                char_idx = char.get("char_idx")
                if (
                    not text
                    or not text.isprintable()
                    or text.isspace()
                    or isinstance(char_idx, bool)
                    or not isinstance(char_idx, int)
                ):
                    continue
                source_bbox = _clip_bbox(_source_bbox(char, geometry, char_idx), page_size)
                tight_bbox = _clip_bbox(_coerce_bbox(geometry.tight_bboxes.get(char_idx)), page_size)
                origin = _coerce_origin(geometry.origins.get(char_idx))
                if source_bbox is None or tight_bbox is None or origin is None:
                    continue
                run_key, font_size = _font_run_key(char, line.angle, text)
                sample = _CharSample(
                    page_index=page_index,
                    line=line,
                    position=position,
                    char_idx=char_idx,
                    text=text,
                    source_bbox=source_bbox,
                    tight_bbox=tight_bbox,
                    origin=origin,
                    local_source_bbox=_rotate_bbox_to_upright(source_bbox, page_size, line.angle),
                    local_tight_bbox=_rotate_bbox_to_upright(tight_bbox, page_size, line.angle),
                    local_origin=_rotate_origin_to_upright(origin, page_size, line.angle),
                    run_key=run_key,
                    font_size=font_size,
                )
                samples.append(sample)
                by_line[(page_index, line.source_index)].append(sample)
        # canonical sample 已持有所需 source bbox；后续表格、脚本和字符回填只读取
        # tight/origin，逐页释放 loose side-map 可限制长文档峰值内存。
        geometry.loose_bboxes.clear()

    for line_samples in by_line.values():
        anchor_heights = [
            sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in line_samples if _is_anchor_text(sample.text)
        ]
        height_q75 = _quantile(anchor_heights, 0.75)
        for sample in line_samples:
            tight_height = sample.local_tight_bbox[3] - sample.local_tight_bbox[1]
            sample.is_anchor = (
                _is_anchor_text(sample.text) and height_q75 > 0 and tight_height >= ANCHOR_MIN_TIGHT_HEIGHT_RATIO * height_q75
            )
    return samples, by_line


def _build_run_stats(
    samples: list[_CharSample],
    by_line: dict[LineKey, list[_CharSample]],
) -> dict[RunKey, _RunStats]:
    """统计同 run 相邻 origin advance 与 loose 覆盖下一 tight 的比例。"""

    runs = {key: _RunStats(key=key) for key in {sample.run_key for sample in samples}}
    for sample in samples:
        run = runs[sample.run_key]
        run.samples.append(sample)
        run.tight_left_bearings.append(sample.local_tight_bbox[0] - sample.local_origin[0])

    for line_samples in by_line.values():
        anchors = [sample for sample in line_samples if sample.is_anchor]
        anchors.sort(key=lambda sample: sample.position)
        for current, following in zip(anchors, anchors[1:]):
            if current.run_key != following.run_key:
                continue
            tight_height = max(
                current.local_tight_bbox[3] - current.local_tight_bbox[1],
                following.local_tight_bbox[3] - following.local_tight_bbox[1],
            )
            if abs(current.local_origin[1] - following.local_origin[1]) > max(0.5, 0.25 * tight_height):
                continue
            advance = following.local_origin[0] - current.local_origin[0]
            tight_width = max(
                current.local_tight_bbox[2] - current.local_tight_bbox[0],
                following.local_tight_bbox[2] - following.local_tight_bbox[0],
            )
            limit = max(5.0 * max(current.font_size, 1.0), 8.0 * tight_width)
            if not 0.1 < advance <= limit:
                continue
            source_width = current.local_source_bbox[2] - current.local_source_bbox[0]
            if source_width <= 0:
                continue
            overlap = current.local_source_bbox[2] - following.local_tight_bbox[0]
            following_width = following.local_tight_bbox[2] - following.local_tight_bbox[0]
            run = runs[current.run_key]
            run.pair_ratios.append(source_width / advance)
            run.pair_overlaps.append(overlap >= 0.05 * max(following_width, 0.1))
            run.advances.append(advance)

    for run in runs.values():
        run.median_advance = statistics.median(run.advances) if run.advances else None
        run.median_tight_left_bearing = statistics.median(run.tight_left_bearings) if run.tight_left_bearings else 0.0
        pair_count = len(run.pair_ratios)
        if pair_count < X_RELIABLE_PAIR_MIN:
            continue
        median_ratio = statistics.median(run.pair_ratios)
        large_share = sum(value > X_STRONG_RATIO_THRESHOLD for value in run.pair_ratios) / pair_count
        overlap_share = sum(run.pair_overlaps) / pair_count
        run.strong_x_bad = (
            median_ratio >= X_STRONG_MEDIAN_RATIO
            and large_share >= X_STRONG_RATIO_SHARE
            and overlap_share >= X_STRONG_NEXT_TIGHT_OVERLAP_SHARE
        )

    bad_families = {run.key[0] for run in runs.values() if run.strong_x_bad}
    for run in runs.values():
        if run.strong_x_bad or run.key[0] not in bad_families or len(run.pair_ratios) < X_SIBLING_PAIR_MIN:
            continue
        median_ratio = statistics.median(run.pair_ratios)
        overlap_share = sum(run.pair_overlaps) / len(run.pair_overlaps)
        run.sibling_x_bad = (
            median_ratio >= X_SIBLING_MEDIAN_RATIO and overlap_share >= X_SIBLING_NEXT_TIGHT_OVERLAP_SHARE
        ) or _quantile(run.pair_ratios, 0.9) >= X_SIBLING_P90_RATIO
    return runs


def _mark_style_inflated_runs(
    runs: dict[RunKey, _RunStats],
) -> set[RunKey]:
    """标记跨页重复出现且 loose 高度同时偏离字号与 tight 的字体 run。"""

    output: set[RunKey] = set()
    for run in runs.values():
        samples_by_line: dict[LineKey, list[_CharSample]] = defaultdict(list)
        for sample in run.samples:
            if sample.is_anchor:
                samples_by_line[(sample.page_index, sample.line.source_index)].append(sample)
        if len(samples_by_line) < STYLE_INFLATION_MIN_LINE_COUNT:
            continue
        inflated_lines: list[LineKey] = []
        for line_key, line_samples in samples_by_line.items():
            if _style_line_is_inflated(
                line_samples[0].line.effective_height,
                [sample.font_size for sample in line_samples if sample.font_size > 0],
                [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in line_samples],
            ):
                inflated_lines.append(line_key)
        if (
            len(inflated_lines) >= STYLE_INFLATION_MIN_LINE_COUNT
            and len(inflated_lines) / len(samples_by_line) >= STYLE_INFLATION_MIN_LINE_SHARE
            and len({page_index for page_index, _source_index in inflated_lines}) >= STYLE_INFLATION_MIN_PAGE_COUNT
        ):
            run.style_y_bad = True
            output.add(run.key)
    return output


def _document_uses_global_style_calibration(
    style_inflated_runs: set[RunKey],
) -> bool:
    """只要存在已通过跨页重复证据的异常 run，就启用全文统一字体尺度。"""

    return bool(style_inflated_runs)


def _apply_style_scale_repairs(
    plan: DocumentGeometryPlan,
    style_inflated_runs: set[RunKey],
    by_line: dict[LineKey, list[_CharSample]],
) -> None:
    """异常文档触发后统一写入逐行 canonical 字号，不改变公开来源 bbox。"""

    if not style_inflated_runs:
        return
    for line_key, line_samples in by_line.items():
        anchor_samples = [
            sample
            for sample in line_samples
            if sample.is_anchor
        ]
        if not anchor_samples:
            continue
        dominant_run = Counter(
            sample.run_key
            for sample in anchor_samples
        ).most_common(1)[0][0]
        style_samples = [
            sample
            for sample in anchor_samples
            if sample.run_key == dominant_run
        ]
        font_sizes = [
            sample.font_size
            for sample in style_samples
            if sample.font_size > 0
        ]
        tight_heights = [
            sample.local_tight_bbox[3]
            - sample.local_tight_bbox[1]
            for sample in style_samples
        ]
        style_scale = max(
            statistics.median(font_sizes)
            if font_sizes
            else 0.0,
            _quantile(tight_heights, 0.75),
            0.1,
        )
        if style_scale < STYLE_INFLATION_MIN_SCALE:
            continue
        plan.line_style_scales[line_key] = style_scale


def _line_uses_repaired_style_scale(
    line: _LineItem,
    style_inflated_runs: set[RunKey],
) -> bool:
    """按字体族、字号、flags 和方向把异常 run 证据投影到当前视觉行。"""

    if line.font_signature is None or line.em_height <= 0:
        return False
    family = _normalized_font_family(line.font_signature[0])
    style_size = round(line.em_height * 4.0) / 4.0
    flags = line.font_signature[1]
    return any(
        family == run_key[0]
        and abs(style_size - run_key[1]) <= 0.25
        and flags == run_key[2]
        and line.angle == run_key[4]
        for run_key in style_inflated_runs
    )


def _restore_stable_legacy_source_bboxes(
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """全文异常成立后改用原始字符 bbox，隔离仅存在于 loose side-map 的扰动。"""

    for (page_index, _source_index), line_samples in by_line.items():
        page_size = page_sizes[page_index]
        for sample in line_samples:
            raw_bbox = _clip_bbox(
                _coerce_bbox(sample.line.chars[sample.position].get("bbox")),
                page_size,
            )
            if raw_bbox is None:
                continue
            sample.source_bbox = raw_bbox
            sample.local_source_bbox = _rotate_bbox_to_upright(
                raw_bbox,
                page_size,
                sample.line.angle,
            )


def _next_compatible_sample(
    current: _CharSample,
    line_samples: list[_CharSample],
) -> _CharSample | None:
    """返回同 run、同基线且 origin 正向的下一可见字符。"""

    for candidate in line_samples:
        if candidate.position <= current.position or candidate.run_key != current.run_key:
            continue
        tight_height = max(
            current.local_tight_bbox[3] - current.local_tight_bbox[1],
            candidate.local_tight_bbox[3] - candidate.local_tight_bbox[1],
        )
        if abs(current.local_origin[1] - candidate.local_origin[1]) > max(0.5, 0.25 * tight_height):
            continue
        advance = candidate.local_origin[0] - current.local_origin[0]
        if advance > 0.1:
            return candidate
    return None


def _repair_x_chars(
    plan: DocumentGeometryPlan,
    runs: dict[RunKey, _RunStats],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """仅对强异常或同族高置信 run 重建前进方向 cell。"""

    for line_key, line_samples in by_line.items():
        page_index, _source_index = line_key
        page_size = page_sizes[page_index]
        for sample in line_samples:
            run = runs[sample.run_key]
            if not (run.strong_x_bad or run.sibling_x_bad):
                continue
            next_sample = _next_compatible_sample(sample, line_samples)
            advance = next_sample.local_origin[0] - sample.local_origin[0] if next_sample is not None else run.median_advance
            if advance is None or advance <= 0:
                continue
            left_bearing = run.median_tight_left_bearing
            left = min(sample.local_tight_bbox[0], sample.local_origin[0] + left_bearing)
            right = max(sample.local_tight_bbox[2], sample.local_origin[0] + advance)
            if next_sample is not None:
                right = min(right, max(sample.local_tight_bbox[2], next_sample.local_origin[0]))
            if right <= left:
                continue
            local_layout = (
                left,
                sample.local_source_bbox[1],
                right,
                sample.local_source_bbox[3],
            )
            layout_bbox = _clip_bbox(
                _rotate_bbox_from_upright(local_layout, page_size, sample.line.angle),
                page_size,
            )
            if layout_bbox is None:
                continue
            plan.char_repairs[(page_index, sample.char_idx)] = CharLayoutGeometry(
                source_bbox=sample.source_bbox,
                tight_bbox=sample.tight_bbox,
                origin=sample.origin,
                layout_bbox=layout_bbox,
                ink_bbox=sample.tight_bbox,
                baseline=sample.local_origin[1],
                advance=advance,
                em_height=max(sample.font_size, sample.local_tight_bbox[3] - sample.local_tight_bbox[1]),
                x_state="abnormal",
                y_state="healthy",
                confidence=min(1.0, len(run.pair_ratios) / X_RELIABLE_PAIR_MIN),
            )


def _baseline_clusters(samples: list[_CharSample]) -> tuple[list[list[_CharSample]], float]:
    """按 origin-v 将 anchor 聚为独立视觉基线。"""

    tight_heights = [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in samples]
    tolerance = max(0.5, 0.25 * _quantile(tight_heights, 0.75))
    clusters: list[list[_CharSample]] = []
    for sample in sorted(samples, key=lambda item: item.local_origin[1]):
        target = next(
            (
                cluster
                for cluster in clusters
                if abs(sample.local_origin[1] - statistics.median(item.local_origin[1] for item in cluster)) <= tolerance
            ),
            None,
        )
        if target is None:
            clusters.append([sample])
        else:
            target.append(sample)
    return clusters, tolerance


def _record_line_canonical_metrics(
    plan: DocumentGeometryPlan,
    by_line: dict[LineKey, list[_CharSample]],
) -> None:
    """为无需改写 bbox 的普通行保存 tight 字形并集和 dominant origin 基线。"""

    for line_key, line_samples in by_line.items():
        if not line_samples:
            continue
        line = line_samples[0].line
        plan.line_ink_bboxes[line_key] = _bbox_union_many(
            [sample.tight_bbox for sample in line_samples],
        )
        if line.formula_candidate_only or line.compact_formula_cluster:
            continue
        anchors = [sample for sample in line_samples if sample.is_anchor]
        if len(anchors) < 2:
            continue
        clusters, _tolerance = _baseline_clusters(anchors)
        dominant = max(
            clusters,
            key=lambda cluster: (
                sum(sample.local_tight_bbox[2] - sample.local_tight_bbox[0] for sample in cluster),
                len(cluster),
            ),
        )
        if len(dominant) / len(anchors) < 0.5:
            continue
        plan.line_baselines[line_key] = statistics.median(sample.local_origin[1] for sample in dominant)


def _analyze_lines(
    lines_by_page: list[list[_LineItem]],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> dict[LineKey, _LineAnalysis]:
    """分析 legacy line 的 dominant anchor row 与 split shadow 候选。"""

    analyses: dict[LineKey, _LineAnalysis] = {}
    for page_index, lines in enumerate(lines_by_page):
        page_size = page_sizes[page_index]
        for line in lines:
            if line.angle != 0 or line.formula_candidate_only or line.restored_inline_cluster or line.compact_formula_cluster:
                continue
            line_key = (page_index, line.source_index)
            line_samples = by_line.get(line_key, [])
            visible_count = sum(
                1
                for char in line.chars
                if str(char.get("char") or "").isprintable() and not str(char.get("char") or "").isspace()
            )
            anchors = [sample for sample in line_samples if sample.is_anchor]
            if visible_count == 0 or len(anchors) < Y_MIN_ANCHOR_COUNT:
                continue
            clusters, _tolerance = _baseline_clusters(anchors)
            dominant = max(
                clusters,
                key=lambda cluster: (
                    sum(item.local_tight_bbox[2] - item.local_tight_bbox[0] for item in cluster),
                    len(cluster),
                ),
            )
            dominant_share = len(dominant) / len(anchors)
            geometry_coverage = len(line_samples) / visible_count
            supported_clusters = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
            baseline = statistics.median(sample.local_origin[1] for sample in dominant)
            tight_core = _bbox_union_many([sample.local_tight_bbox for sample in dominant])
            source_envelope = _bbox_union_many([sample.local_source_bbox for sample in anchors])
            all_source_envelope = _bbox_union_many([sample.local_source_bbox for sample in line_samples])
            run_key = Counter(sample.run_key for sample in dominant).most_common(1)[0][0]
            analyses[line_key] = _LineAnalysis(
                key=line_key,
                line=line,
                samples=line_samples,
                anchors=anchors,
                dominant=dominant,
                baseline=baseline,
                tight_core=tight_core,
                local_source_bbox=source_envelope,
                explicit_all_source_bbox=all_source_envelope,
                legacy_local_bbox=_rotate_bbox_to_upright(line.bbox, page_size, line.angle),
                run_key=run_key,
                geometry_coverage=geometry_coverage,
                dominant_share=dominant_share,
                split_y_candidate=len(supported_clusters) >= 2,
            )
    return analyses


def _has_document_y_risk(by_line: dict[LineKey, list[_CharSample]]) -> bool:
    """用稳定单基线行的尾部分布实现正常文档 Y 分析快速否决。"""

    ratios: list[float] = []
    extreme_by_run: Counter[RunKey] = Counter()
    for line_samples in by_line.values():
        line = line_samples[0].line if line_samples else None
        if (
            line is None
            or line.angle != 0
            or line.formula_candidate_only
            or line.restored_inline_cluster
            or line.compact_formula_cluster
        ):
            continue
        anchors = [sample for sample in line_samples if sample.is_anchor]
        if len(anchors) < Y_MIN_ANCHOR_COUNT:
            continue
        clusters, _tolerance = _baseline_clusters(anchors)
        supported_clusters = [cluster for cluster in clusters if len(cluster) >= 3 and len(cluster) / len(anchors) >= 0.20]
        if len(supported_clusters) >= 2:
            return True
        dominant = max(clusters, key=len)
        if len(dominant) / len(anchors) < Y_DOMINANT_ROW_SHARE:
            continue
        source_union = _bbox_union_many([sample.local_source_bbox for sample in dominant])
        tight_union = _bbox_union_many([sample.local_tight_bbox for sample in dominant])
        tight_height = tight_union[3] - tight_union[1]
        if tight_height <= 0:
            continue
        ratio = (source_union[3] - source_union[1]) / tight_height
        ratios.append(ratio)
        if ratio >= 3.0:
            run_key = Counter(sample.run_key for sample in dominant).most_common(1)[0][0]
            extreme_by_run[run_key] += 1
    return bool(ratios) and (
        _quantile(ratios, 0.95) >= Y_DOCUMENT_RISK_P95_RATIO
        or any(count >= Y_MIN_REPEATED_LINES for count in extreme_by_run.values())
    )


def _same_lane(first: _LineAnalysis, second: _LineAnalysis) -> bool:
    """用水平覆盖和左边缘近似判断两行是否属于同一栏。"""

    height = max(
        first.tight_core[3] - first.tight_core[1],
        second.tight_core[3] - second.tight_core[1],
    )
    return _bbox_axis_overlap_ratio(first.local_source_bbox, second.local_source_bbox, axis="x") >= 0.35 or abs(
        first.local_source_bbox[0] - second.local_source_bbox[0]
    ) <= 2.0 * max(height, 1.0)


def _assign_neighbors(analyses: dict[LineKey, _LineAnalysis]) -> None:
    """为每条 eligible line 选择上下最近的同栏正文行。"""

    by_page: dict[int, list[_LineAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        by_page[analysis.key[0]].append(analysis)
    for page_analyses in by_page.values():
        for current in page_analyses:
            candidates = [
                other
                for other in page_analyses
                if other is not current
                and _same_lane(current, other)
                and abs(other.baseline - current.baseline) >= max(2.0, 0.6 * (current.tight_core[3] - current.tight_core[1]))
            ]
            above = [other for other in candidates if other.baseline < current.baseline]
            below = [other for other in candidates if other.baseline > current.baseline]
            if above:
                current.neighbors.append(max(above, key=lambda other: other.baseline))
            if below:
                current.neighbors.append(min(below, key=lambda other: other.baseline))


def _intrusion_ratio(source_bbox: BBox, tight_core: BBox) -> float:
    """计算 loose envelope 侵入邻行 tight core 的纵向比例。"""

    overlap = max(0.0, min(source_bbox[3], tight_core[3]) - max(source_bbox[1], tight_core[1]))
    height = tight_core[3] - tight_core[1]
    return overlap / height if height > 0 else 0.0


def _mark_y_candidates(analyses: dict[LineKey, _LineAnalysis]) -> set[RunKey]:
    """按局部侵入和文档级重复支持确认允许 trim_y 的 run。"""

    _assign_neighbors(analyses)
    eligible_by_run: dict[RunKey, list[_LineAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        if (
            len(analysis.anchors) < Y_MIN_ANCHOR_COUNT
            or analysis.geometry_coverage < Y_MIN_GEOMETRY_COVERAGE
            or analysis.dominant_share < Y_DOMINANT_ROW_SHARE
        ):
            continue
        analysis.intrusion_ratio = max(
            (_intrusion_ratio(analysis.local_source_bbox, neighbor.tight_core) for neighbor in analysis.neighbors),
            default=0.0,
        )
        analysis.y_candidate = analysis.intrusion_ratio >= Y_NEIGHBOR_CORE_INTRUSION
        eligible_by_run[analysis.run_key].append(analysis)

    confirmed: set[RunKey] = set()
    for run_key, members in eligible_by_run.items():
        candidates = [member for member in members if member.y_candidate]
        if len(candidates) >= Y_MIN_REPEATED_LINES and len(candidates) / len(members) >= Y_MIN_REPEATED_SHARE:
            confirmed.add(run_key)
    return confirmed


def _healthy_loose_offsets_by_run(
    analyses: dict[LineKey, _LineAnalysis],
    confirmed_runs: set[RunKey],
) -> dict[RunKey, tuple[list[float], list[float]]]:
    """单次遍历收集各确认 run 的健康 loose ascent/descent。"""

    values: dict[RunKey, tuple[list[float], list[float]]] = {run_key: ([], []) for run_key in confirmed_runs}
    for analysis in analyses.values():
        if analysis.run_key not in confirmed_runs or analysis.y_candidate:
            continue
        ascents, descents = values[analysis.run_key]
        for sample in analysis.dominant:
            ascents.append(max(0.0, sample.local_origin[1] - sample.local_source_bbox[1]))
            descents.append(max(0.0, sample.local_source_bbox[3] - sample.local_origin[1]))
    return values


def _repair_y_lines(
    plan: DocumentGeometryPlan,
    analyses: dict[LineKey, _LineAnalysis],
    confirmed_runs: set[RunKey],
    x_bad_runs: set[RunKey],
    page_sizes: list[tuple[float, float]],
) -> None:
    """为确认异常的单基线行生成 baseline 锚定的 Y envelope。"""

    healthy_offsets = _healthy_loose_offsets_by_run(analyses, confirmed_runs)
    for line_key, analysis in analyses.items():
        current = plan.line_repairs.get(line_key)
        source_bbox = analysis.line.source_bbox or analysis.line.bbox
        should_trim = (
            analysis.run_key in confirmed_runs
            and (analysis.y_candidate or analysis.run_key in x_bad_runs)
            and len(analysis.anchors) >= Y_MIN_ANCHOR_COUNT
            and analysis.geometry_coverage >= Y_MIN_GEOMETRY_COVERAGE
            and analysis.dominant_share >= Y_DOMINANT_ROW_SHARE
        )
        if current is None and not analysis.split_y_candidate and not should_trim:
            continue
        if current is None:
            current = LineGeometryRepair(
                source_bbox=source_bbox,
                layout_bbox=analysis.line.bbox,
                ink_bbox=_bbox_union_many([sample.tight_bbox for sample in analysis.samples]) if analysis.samples else None,
                baseline=analysis.baseline,
                em_height=analysis.line.effective_height,
                split_y_candidate=analysis.split_y_candidate,
                run_key=analysis.run_key,
            )
            plan.line_repairs[line_key] = current
        else:
            current.split_y_candidate = analysis.split_y_candidate
            current.baseline = analysis.baseline
            current.run_key = analysis.run_key
        if not should_trim:
            continue

        local_tight_boxes = [sample.local_tight_bbox for sample in analysis.samples]
        tight_union = _bbox_union_many(local_tight_boxes)
        font_sizes = [sample.font_size for sample in analysis.dominant if sample.font_size > 0]
        tight_heights = [sample.local_tight_bbox[3] - sample.local_tight_bbox[1] for sample in analysis.dominant]
        em_height = max(
            statistics.median(font_sizes) if font_sizes else 0.0,
            _quantile(tight_heights, 0.75),
            0.1,
        )
        healthy_ascents, healthy_descents = healthy_offsets[analysis.run_key]
        healthy_loose_height = (
            statistics.median(healthy_ascents) + statistics.median(healthy_descents)
            if healthy_ascents and healthy_descents
            else math.inf
        )
        tier_offsets = _line_loose_tier_offsets(
            [
                (
                    max(
                        0.0,
                        sample.local_origin[1]
                        - sample.local_source_bbox[1],
                    ),
                    max(
                        0.0,
                        sample.local_source_bbox[3]
                        - sample.local_origin[1],
                    ),
                    sample.local_tight_bbox[3]
                    - sample.local_tight_bbox[1],
                    sample.font_size,
                )
                for sample in analysis.dominant
                if sample.run_key == analysis.run_key
            ],
            em_height,
        )
        if tier_offsets is not None:
            ascent, descent = tier_offsets
        elif len(healthy_ascents) >= Y_HEALTHY_LOOSE_SAMPLE_MIN and healthy_loose_height <= 1.5 * em_height:
            ascent = statistics.median(healthy_ascents)
            descent = statistics.median(healthy_descents)
        else:
            ascent = _quantile(
                [max(0.0, sample.local_origin[1] - sample.local_tight_bbox[1]) for sample in analysis.dominant],
                0.9,
            )
            descent = _quantile(
                [max(0.0, sample.local_tight_bbox[3] - sample.local_origin[1]) for sample in analysis.dominant],
                0.9,
            )
            padding = max(0.5, 0.08 * em_height)
            ascent += padding
            descent += padding

        top = min(analysis.baseline - ascent, tight_union[1])
        bottom = max(analysis.baseline + descent, tight_union[3])
        for neighbor in analysis.neighbors:
            midpoint = (analysis.baseline + neighbor.baseline) / 2.0
            if neighbor.baseline < analysis.baseline:
                top = max(top, min(midpoint, tight_union[1]))
            else:
                bottom = min(bottom, max(midpoint, tight_union[3]))
        legacy_height = analysis.legacy_local_bbox[3] - analysis.legacy_local_bbox[1]
        provenance_tolerance = max(1.05, 0.05 * legacy_height)
        preserve_legacy_typography = (
            abs(analysis.explicit_all_source_bbox[1] - analysis.legacy_local_bbox[1]) > provenance_tolerance
            or abs(analysis.explicit_all_source_bbox[3] - analysis.legacy_local_bbox[3]) > provenance_tolerance
        )
        if preserve_legacy_typography:
            # explicit loose 与 legacy char bbox 来源不同时保留已经健康的 legacy envelope；
            # 该分支也让 review 扰动只验证 side-map 修复，不重写既有 line membership。
            top = analysis.legacy_local_bbox[1]
            bottom = analysis.legacy_local_bbox[3]
        page_size = page_sizes[line_key[0]]
        local_current = _rotate_bbox_to_upright(current.layout_bbox, page_size, analysis.line.angle)
        layout_bbox = _clip_bbox(
            _rotate_bbox_from_upright(
                (local_current[0], top, local_current[2], bottom),
                page_size,
                analysis.line.angle,
            ),
            page_size,
        )
        if layout_bbox is None:
            continue
        current.layout_bbox = layout_bbox
        current.ink_bbox = _bbox_union_many([sample.tight_bbox for sample in analysis.samples])
        current.em_height = (
            analysis.line.effective_height
            if preserve_legacy_typography
            else max(0.1, bottom - top)
        )
        current.state = "repair_xy" if current.state == "repair_x" else "trim_y"
        current.confidence = min(analysis.geometry_coverage, analysis.dominant_share)
        current.y_intrusion_ratio = analysis.intrusion_ratio


def _build_line_repairs_from_x(
    plan: DocumentGeometryPlan,
    lines_by_page: list[list[_LineItem]],
    by_line: dict[LineKey, list[_CharSample]],
    page_sizes: list[tuple[float, float]],
) -> None:
    """把已修复字符并集投影为只改变 X 的 canonical line。"""

    for page_index, lines in enumerate(lines_by_page):
        page_size = page_sizes[page_index]
        for line in lines:
            line_key = (page_index, line.source_index)
            line_samples = by_line.get(line_key, [])
            repaired = [
                plan.char_repairs[(page_index, sample.char_idx)]
                for sample in line_samples
                if (page_index, sample.char_idx) in plan.char_repairs
            ]
            if not repaired:
                continue
            local_source = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
            explicit_source = _bbox_union_many([sample.local_source_bbox for sample in line_samples])
            if abs(explicit_source[0] - local_source[0]) > 0.25 or abs(explicit_source[2] - local_source[2]) > 0.25:
                # 旋转字符或 shadow 扰动的 explicit loose 与 legacy bbox 来源不同，
                # 首版保留 legacy X，避免把诊断 side-map 变化误写入公开输出。
                continue
            local_boxes = []
            for sample in line_samples:
                repair = plan.char_repairs.get((page_index, sample.char_idx))
                page_bbox = repair.layout_bbox if repair is not None else sample.source_bbox
                local_boxes.append(_rotate_bbox_to_upright(page_bbox, page_size, line.angle))
            local_union = _bbox_union_many(local_boxes)
            layout_bbox = _clip_bbox(
                _rotate_bbox_from_upright(
                    (local_union[0], local_source[1], local_union[2], local_source[3]),
                    page_size,
                    line.angle,
                ),
                page_size,
            )
            if layout_bbox is None:
                continue
            ink_bbox = _bbox_union_many([sample.tight_bbox for sample in line_samples]) if line_samples else None
            plan.line_repairs[line_key] = LineGeometryRepair(
                source_bbox=line.bbox,
                layout_bbox=layout_bbox,
                ink_bbox=ink_bbox,
                baseline=None,
                em_height=line.effective_height,
                state="repair_x",
                confidence=min(repair.confidence for repair in repaired),
                repaired_char_count=len(repaired),
            )


def _run_diagnostics(runs: dict[RunKey, _RunStats]) -> list[dict[str, Any]]:
    """生成不依赖字符对象的 run 级可序列化诊断。"""

    output = []
    for run in sorted(runs.values(), key=lambda item: item.key):
        pair_count = len(run.pair_ratios)
        output.append(
            {
                "run_key": list(run.key),
                "sample_count": len(run.samples),
                "reliable_pair_count": pair_count,
                "median_ratio": statistics.median(run.pair_ratios) if pair_count else None,
                "p90_ratio": _quantile(run.pair_ratios, 0.9) if pair_count else None,
                "ratio_gt_1_35_share": (
                    sum(value > X_STRONG_RATIO_THRESHOLD for value in run.pair_ratios) / pair_count if pair_count else 0.0
                ),
                "next_tight_overlap_share": sum(run.pair_overlaps) / pair_count if pair_count else 0.0,
                "strong_x_bad": run.strong_x_bad,
                "sibling_x_bad": run.sibling_x_bad,
                "style_y_bad": run.style_y_bad,
            }
        )
    return output


def build_document_geometry_plan(
    lines_by_page: list[list[_LineItem]],
    geometries: list[PDFPageTextGeometry],
    page_sizes: list[tuple[float, float]],
) -> DocumentGeometryPlan:
    """构建文档级 X 修复、Y trim 与 split shadow 计划。"""

    plan = DocumentGeometryPlan()
    if not any(geometry.tight_bboxes and geometry.origins for geometry in geometries):
        return plan
    risk = _document_requires_full_geometry(
        lines_by_page,
        geometries,
        page_sizes,
    )
    if not risk.any:
        for geometry in geometries:
            geometry.loose_bboxes.clear()
        return plan
    samples, by_line = _collect_samples(lines_by_page, geometries, page_sizes)
    if risk.layout:
        # 只有布局风险才记录公开输出候选；仅样式异常时只校准内部字号。
        _record_line_canonical_metrics(plan, by_line)
    runs = _build_run_stats(samples, by_line)
    style_inflated_runs = _mark_style_inflated_runs(runs)
    plan.style_inflated_runs = style_inflated_runs
    plan.document_style_anomaly = _document_uses_global_style_calibration(
        style_inflated_runs,
    )
    _apply_style_scale_repairs(
        plan,
        style_inflated_runs if plan.document_style_anomaly else set(),
        by_line,
    )
    if plan.document_style_anomaly and risk.layout:
        _restore_stable_legacy_source_bboxes(by_line, page_sizes)
        runs = _build_run_stats(samples, by_line)
        for run in runs.values():
            run.style_y_bad = run.key in style_inflated_runs
    plan.run_diagnostics = _run_diagnostics(runs)
    if not risk.layout:
        return plan
    x_bad_runs = {run.key for run in runs.values() if run.strong_x_bad or run.sibling_x_bad}
    if x_bad_runs:
        _repair_x_chars(plan, runs, by_line, page_sizes)
        _build_line_repairs_from_x(plan, lines_by_page, by_line, page_sizes)
    if not _has_document_y_risk(by_line):
        return plan
    analyses = _analyze_lines(lines_by_page, by_line, page_sizes)
    confirmed_runs = _mark_y_candidates(analyses)
    _repair_y_lines(plan, analyses, confirmed_runs, x_bad_runs, page_sizes)
    return plan


def apply_line_geometry_repairs(
    lines: list[_LineItem],
    *,
    page_index: int,
    plan: DocumentGeometryPlan,
    allow_y_trim: bool,
) -> None:
    """把文档计划应用到当前仍可参与 Flash 布局的行。"""

    for line in lines:
        canonical_ink_bbox = plan.line_ink_bboxes.get(
            (page_index, line.source_index),
        )
        if canonical_ink_bbox is not None:
            line.ink_bbox = canonical_ink_bbox
        canonical_baseline = plan.line_baselines.get(
            (page_index, line.source_index),
        )
        if canonical_baseline is not None:
            line.baseline = canonical_baseline
        style_scale = plan.line_style_scales.get(
            (page_index, line.source_index),
        )
        if style_scale is not None:
            line.em_height = style_scale
        line.style_scale_repaired = _line_uses_repaired_style_scale(
            line,
            plan.style_inflated_runs,
        )
        repair = plan.line_repairs.get((page_index, line.source_index))
        if repair is None:
            continue
        if repair.state in {"trim_y", "repair_xy"} and not allow_y_trim:
            if repair.state == "trim_y":
                continue
            local_state = "repair_x"
            source = repair.source_bbox
            layout = (repair.layout_bbox[0], source[1], repair.layout_bbox[2], source[3])
        else:
            local_state = repair.state
            layout = repair.layout_bbox
        line.source_bbox = repair.source_bbox
        line.ink_bbox = repair.ink_bbox
        if repair.baseline is not None:
            line.baseline = repair.baseline
        line.geometry_state = local_state
        line.geometry_confidence = repair.confidence
        line.split_y_candidate = repair.split_y_candidate
        line.bbox = layout
        line.em_height = (
            style_scale
            if style_scale is not None
            else repair.em_height
        )
        if allow_y_trim and repair.state in {"trim_y", "repair_xy"}:
            line.effective_height = repair.em_height


__all__ = [
    "CharLayoutGeometry",
    "DocumentGeometryPlan",
    "LineGeometryRepair",
    "apply_line_geometry_repairs",
    "build_document_geometry_plan",
]
