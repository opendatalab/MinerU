# Copyright (c) Opendatalab. All rights reserved.
"""编排 Native PDF 表格多候选生成、欠分割诊断和高置信仲裁。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .candidate import serialize_candidate_html
from .contracts import (
    NativeTableCandidate,
    NativeTableInput,
    NativeTableRectangle,
    NativeTableResult,
    NativeTableRule,
    NativeTableText,
)
from .geometry import bbox_intersection, normalize_bbox
from .sparse_hybrid import (
    build_sparse_hybrid_candidates,
    diagnose_sparse_hybrid_candidate_builds,
)
from .text import build_native_table_text
from .text_grid import build_text_candidates, diagnose_text_candidate_builds
from .vector import (
    MAX_PRIMITIVES_PER_TABLE,
    build_vector_candidates,
    diagnose_vector_candidate_builds,
)

MIN_TOPOLOGY_SCORE_GAP = 0.05
_SOURCE_PRIORITY = {
    "vector_grid": 5,
    "sparse_hybrid": 4,
    "sparse_grid": 3,
    "key_value": 2,
    "text_grid": 1,
}
_VERIFIED_SCORE_BY_SOURCE = {
    "vector_grid": 0.95,
    "sparse_hybrid": 0.98,
    "sparse_grid": 0.95,
    "key_value": 0.95,
    "text_grid": 0.95,
}


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    """保存一次生产决策及调试工具需要的中间候选。"""

    primitive_count: int
    text: NativeTableText | None
    generated_candidates: tuple[NativeTableCandidate, ...]
    candidates: tuple[NativeTableCandidate, ...]
    selected: NativeTableCandidate | None
    physical_topology_conflict: bool
    first_rejection_gate: str | None


def _is_verified_line_candidate(candidate: NativeTableCandidate) -> bool:
    """判断候选是否由无歧义 drawing 网格独立验证。"""

    return (
        candidate.source == "vector_grid"
        and candidate.score >= 0.95
        and "evidence=line_grid" in candidate.issues
        and "ambiguous_separator_ratio=0.0000" in candidate.issues
    )


def _is_verified_rect_candidate(candidate: NativeTableCandidate) -> bool:
    """判断候选是否由无歧义矩形晶格独立验证。"""

    return (
        candidate.source == "vector_grid"
        and candidate.score >= 0.95
        and "evidence=rect_grid" in candidate.issues
        and "ambiguous_separator_ratio=0.0000" in candidate.issues
    )


def _has_line_rect_topology_conflict(
    candidates: Iterable[NativeTableCandidate],
) -> bool:
    """判断两类独立物理证据是否给出不同拓扑。"""

    materialized = list(candidates)
    line_candidates = [candidate for candidate in materialized if _is_verified_line_candidate(candidate)]
    rect_candidates = [candidate for candidate in materialized if _is_verified_rect_candidate(candidate)]
    return any(line.topology != rect.topology for line in line_candidates for rect in rect_candidates)


def _has_attempted_line_rect_grid_conflict(
    attempts: Iterable[dict[str, Any]],
) -> bool:
    """判断已恢复轨道的 line/rect 物理假设是否在行列数上冲突。"""

    materialized = list(attempts)
    line_grids = [attempt.get("grid") for attempt in materialized if attempt.get("evidence") == "line_grid"]
    rect_grids = [attempt.get("grid") for attempt in materialized if attempt.get("evidence") == "rect_grid"]
    return any(
        isinstance(line, dict)
        and isinstance(rect, dict)
        and (line.get("rows"), line.get("cols")) != (rect.get("rows"), rect.get("cols"))
        for line in line_grids
        for rect in rect_grids
    )


def _resolved_rect_undercount_candidate(
    candidates: Iterable[NativeTableCandidate],
    attempts: Iterable[dict[str, Any]],
    text: NativeTableText,
) -> NativeTableCandidate | None:
    """在线网格明确漏行且矩形晶格逐行吻合时允许 rect 独立胜出。"""

    rect_candidates = [candidate for candidate in candidates if _is_verified_rect_candidate(candidate)]
    if len(rect_candidates) != 1:
        return None
    rect_candidate = rect_candidates[0]
    line_attempts = [attempt for attempt in attempts if attempt.get("evidence") == "line_grid"]
    has_matching_undercount = any(
        attempt.get("first_rejection_gate") == "physical_row_undercount"
        and isinstance(attempt.get("grid"), dict)
        and attempt["grid"].get("cols") == rect_candidate.cols
        and attempt["grid"].get("rows", 0) < rect_candidate.rows
        for attempt in line_attempts
    )
    if not has_matching_undercount:
        return None
    if rect_candidate.rows != len(text.rows) or rect_candidate.text_capture < 1.0 or rect_candidate.order_consistency < 1.0:
        return None
    return rect_candidate


def _passes_verified_threshold(candidate: NativeTableCandidate) -> bool:
    """按候选来源应用独立校准的 verified 可靠度门槛。"""

    return candidate.score >= _VERIFIED_SCORE_BY_SOURCE[candidate.source]


def _read_value(item: object, name: str, default: Any = None) -> Any:
    """同时读取普通对象属性和字典字段，供页面原语适配使用。"""

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def coerce_native_table_rules(
    drawing_lines: Iterable[object],
) -> tuple[NativeTableRule, ...]:
    """把 PDFDocument 或 Flash drawing 结果转换成共享横竖线契约。"""

    rules: list[NativeTableRule] = []
    for drawing_line in drawing_lines:
        bbox = normalize_bbox(_read_value(drawing_line, "bbox"))
        orientation = str(_read_value(drawing_line, "orientation", ""))
        if bbox is None or orientation not in {"horizontal", "vertical"}:
            continue
        try:
            width = max(0.0, float(_read_value(drawing_line, "width", 0.0) or 0.0))
        except (TypeError, ValueError):
            width = 0.0
        rules.append(
            NativeTableRule(
                bbox=bbox,
                width=width,
                orientation=orientation,  # type: ignore[arg-type]
            )
        )
    return tuple(rules)


def coerce_native_table_rectangles(
    path_infos: Iterable[object],
) -> tuple[NativeTableRectangle, ...]:
    """把 PDF Path 摘要转换成共享矩形路径契约。"""

    rectangles: list[NativeTableRectangle] = []
    for path_info in path_infos:
        bbox = normalize_bbox(_read_value(path_info, "bbox"))
        if bbox is None:
            continue
        try:
            segment_count = int(_read_value(path_info, "segment_count", 0) or 0)
            form_depth = int(_read_value(path_info, "form_depth", 0) or 0)
        except (TypeError, ValueError):
            continue
        rectangles.append(
            NativeTableRectangle(
                bbox=bbox,
                segment_count=segment_count,
                fill_visible=bool(_read_value(path_info, "fill_visible", False)),
                stroke_visible=bool(_read_value(path_info, "stroke_visible", False)),
                form_depth=form_depth,
            )
        )
    return tuple(rectangles)


def _remove_undercounted_vector_candidates(
    candidates: list[NativeTableCandidate],
) -> list[NativeTableCandidate]:
    """当稳定文本候选显著多出行列时，剔除欠分割矢量候选。"""

    text_candidates = [
        candidate
        for candidate in candidates
        if candidate.source != "vector_grid" and candidate.row_stability >= 0.80 and candidate.column_stability >= 0.80
    ]
    if not text_candidates:
        return candidates
    output: list[NativeTableCandidate] = []
    for candidate in candidates:
        if candidate.source != "vector_grid":
            output.append(candidate)
            continue
        if _is_verified_line_candidate(candidate):
            output.append(candidate)
            continue
        undercounted = any(
            text_candidate.rows >= math.ceil(1.5 * candidate.rows) or text_candidate.cols >= math.ceil(1.3 * candidate.cols)
            for text_candidate in text_candidates
        )
        if not undercounted:
            output.append(candidate)
    return output


def _has_alias_affected_physical_blank_row(
    vector_attempts: tuple[dict[str, Any], ...],
) -> bool:
    """判断强线框空白行是否因自身 alias 风险而禁止文本候选绕过。"""

    for attempt in vector_attempts:
        if attempt.get("evidence") != "line_grid":
            continue
        hypotheses = [attempt, *attempt.get("track_hypotheses", [])]
        for hypothesis in hypotheses:
            if hypothesis.get("first_rejection_gate") != "empty_row":
                continue
            empty_rows = set(hypothesis.get("empty_rows", []))
            affected_rows = set(hypothesis.get("alias_affected_rows", []))
            if empty_rows.intersection(affected_rows):
                return True
    return False


def _has_physical_row_undercount(
    vector_attempts: tuple[dict[str, Any], ...],
) -> bool:
    """判断 line-grid 及其有限轨道假设是否已发现物理行欠分割。"""

    for attempt in vector_attempts:
        if attempt.get("evidence") != "line_grid":
            continue
        hypotheses = [attempt, *attempt.get("track_hypotheses", [])]
        if any(hypothesis.get("first_rejection_gate") == "physical_row_undercount" for hypothesis in hypotheses):
            return True
    return False


def _table_primitive_count(table_input: NativeTableInput) -> int:
    """统计实际与目标表格相交的 drawing 和矩形数量。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return 0
    count = 0
    for primitive in (*table_input.drawing_lines, *table_input.rectangles):
        primitive_bbox = normalize_bbox(primitive.bbox)
        if primitive_bbox is not None and bbox_intersection(primitive_bbox, table_bbox) is not None:
            count += 1
    return count


def _select_candidate(
    candidates: list[NativeTableCandidate],
) -> NativeTableCandidate | None:
    """选择达到生产门槛且未与近分异构候选冲突的最佳结果。"""

    accepted = [candidate for candidate in candidates if _passes_verified_threshold(candidate)]
    if not accepted:
        return None
    if _has_line_rect_topology_conflict(accepted):
        return None
    verified_line_candidates = [candidate for candidate in accepted if _is_verified_line_candidate(candidate)]
    if len(verified_line_candidates) == 1:
        return verified_line_candidates[0]
    accepted.sort(
        key=lambda candidate: (
            candidate.score,
            _SOURCE_PRIORITY[candidate.source],
        ),
        reverse=True,
    )
    best = accepted[0]
    for competitor in accepted[1:]:
        if competitor.topology == best.topology:
            continue
        if best.score - competitor.score < MIN_TOPOLOGY_SCORE_GAP:
            return None
    return best


def _evaluate_native_pdf_table(
    table_input: NativeTableInput,
) -> _CandidateEvaluation:
    """执行共享生产判定，并保留候选生成到仲裁的完整阶段结果。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    page_width, page_height = table_input.page_size
    primitive_count = _table_primitive_count(table_input)
    if (
        table_bbox is None
        or page_width <= 0
        or page_height <= 0
        or table_bbox[0] < 0
        or table_bbox[1] < 0
        or table_bbox[2] > page_width
        or table_bbox[3] > page_height
    ):
        return _CandidateEvaluation(
            primitive_count,
            None,
            (),
            (),
            None,
            False,
            "input_geometry",
        )
    if primitive_count > MAX_PRIMITIVES_PER_TABLE:
        return _CandidateEvaluation(
            primitive_count,
            None,
            (),
            (),
            None,
            False,
            "primitive_limit",
        )
    text = build_native_table_text(table_input)
    if text is None:
        return _CandidateEvaluation(
            primitive_count,
            None,
            (),
            (),
            None,
            False,
            "native_text",
        )
    vector_candidates = build_vector_candidates(table_input, text)
    vector_attempts: tuple[dict[str, Any], ...] = ()
    physical_topology_conflict = False
    if any(_is_verified_rect_candidate(candidate) for candidate in vector_candidates):
        vector_attempts = diagnose_vector_candidate_builds(
            table_input,
            text,
        )
        physical_topology_conflict = _has_attempted_line_rect_grid_conflict(vector_attempts)
        if (
            physical_topology_conflict
            and _resolved_rect_undercount_candidate(
                vector_candidates,
                vector_attempts,
                text,
            )
            is not None
        ):
            physical_topology_conflict = False
    vector_selection = None if physical_topology_conflict else _select_candidate(vector_candidates)
    sparse_hybrid_allowed = len(text.rows) >= 2 and vector_selection is None and not physical_topology_conflict
    if sparse_hybrid_allowed:
        if not vector_attempts:
            vector_attempts = diagnose_vector_candidate_builds(
                table_input,
                text,
            )
        if _has_alias_affected_physical_blank_row(vector_attempts):
            sparse_hybrid_allowed = False
    sparse_hybrid_candidates = build_sparse_hybrid_candidates(table_input, text) if sparse_hybrid_allowed else []
    sparse_hybrid_selection = _select_candidate(sparse_hybrid_candidates)
    text_candidates = (
        build_text_candidates(table_input, text)
        if len(text.rows) >= 2 and vector_selection is None and sparse_hybrid_selection is None
        else []
    )
    if text_candidates and (not vector_candidates or physical_topology_conflict):
        if not vector_attempts:
            vector_attempts = diagnose_vector_candidate_builds(
                table_input,
                text,
            )
        if _has_alias_affected_physical_blank_row(vector_attempts) or _has_physical_row_undercount(vector_attempts):
            text_candidates = []
    generated_candidates = [
        *vector_candidates,
        *sparse_hybrid_candidates,
        *text_candidates,
    ]
    candidates = _remove_undercounted_vector_candidates(generated_candidates)
    selected = None if physical_topology_conflict else _select_candidate(candidates)
    if selected is not None:
        first_rejection_gate = None
    elif not generated_candidates:
        first_rejection_gate = "candidate_generation"
    elif not candidates:
        first_rejection_gate = "undercount_guard"
    elif not any(_passes_verified_threshold(candidate) for candidate in candidates):
        first_rejection_gate = "verified_threshold"
    else:
        first_rejection_gate = "topology_conflict"
    return _CandidateEvaluation(
        primitive_count,
        text,
        tuple(generated_candidates),
        tuple(candidates),
        selected,
        physical_topology_conflict,
        first_rejection_gate,
    )


def diagnose_native_pdf_table(table_input: NativeTableInput) -> dict[str, Any]:
    """返回仅供测试评测使用、不会进入用户结果的候选诊断。"""

    evaluation = _evaluate_native_pdf_table(table_input)
    vector_attempts = (
        diagnose_vector_candidate_builds(
            table_input,
            evaluation.text,
        )
        if evaluation.text is not None
        else ()
    )
    text_attempts = (
        diagnose_text_candidate_builds(
            table_input,
            evaluation.text,
        )
        if evaluation.text is not None
        else ()
    )
    sparse_hybrid_attempts = (
        diagnose_sparse_hybrid_candidate_builds(
            table_input,
            evaluation.text,
        )
        if evaluation.text is not None
        else ()
    )
    first_rejection_gate = evaluation.first_rejection_gate
    if first_rejection_gate == "candidate_generation":
        text_attempt = next(
            (
                attempt
                for attempt in text_attempts
                if attempt.get("first_rejection_gate")
                in {
                    "dense_row_ambiguity",
                    "header_requires_rowspan",
                    "token_split",
                }
            ),
            None,
        )
        line_attempt = next(
            (attempt for attempt in vector_attempts if attempt.get("evidence") == "line_grid"),
            None,
        )
        if text_attempt is not None:
            first_rejection_gate = "text_" + str(text_attempt["first_rejection_gate"])
        elif sparse_attempt := next(
            (attempt for attempt in sparse_hybrid_attempts if attempt.get("first_rejection_gate")),
            None,
        ):
            first_rejection_gate = "sparse_hybrid_" + str(sparse_attempt["first_rejection_gate"])
        elif line_attempt is not None and line_attempt.get("first_rejection_gate"):
            first_rejection_gate = "vector_" + str(line_attempt["first_rejection_gate"])

    def candidate_record(
        candidate: NativeTableCandidate,
    ) -> dict[str, Any]:
        """把内部候选转换成稳定且不包含单元格全文的诊断记录。"""

        return {
            "source": candidate.source,
            "rows": candidate.rows,
            "cols": candidate.cols,
            "tracks": {
                "x": candidate.cols + 1,
                "y": candidate.rows + 1,
            },
            "score": candidate.score,
            "verified": _passes_verified_threshold(candidate),
            "score_components": {
                "text_capture": candidate.text_capture,
                "structure_support": candidate.structure_support,
                "row_stability": candidate.row_stability,
                "column_stability": candidate.column_stability,
                "order_consistency": candidate.order_consistency,
            },
            "span_signature": [
                [cell.row, cell.col, cell.rowspan, cell.colspan]
                for cell in candidate.cells
                if cell.rowspan > 1 or cell.colspan > 1
            ],
            "issues": list(candidate.issues),
        }

    retained_ids = {id(candidate) for candidate in evaluation.candidates}
    removed_candidates = [
        candidate_record(candidate) for candidate in evaluation.generated_candidates if id(candidate) not in retained_ids
    ]
    counterfactual_best = max(
        evaluation.generated_candidates,
        key=lambda candidate: (
            candidate.score,
            _SOURCE_PRIORITY[candidate.source],
        ),
        default=None,
    )
    return {
        "primitive_count": evaluation.primitive_count,
        "glyph_count": len(evaluation.text.glyphs) if evaluation.text else 0,
        "visual_text_rows": len(evaluation.text.rows) if evaluation.text else 0,
        "first_rejection_gate": first_rejection_gate,
        "line_rect_topology_conflict": (
            evaluation.physical_topology_conflict or _has_line_rect_topology_conflict(evaluation.candidates)
        ),
        "vector_attempts": list(vector_attempts),
        "sparse_hybrid_attempts": list(sparse_hybrid_attempts),
        "text_attempts": list(text_attempts),
        "generated_candidates": [candidate_record(candidate) for candidate in evaluation.generated_candidates],
        "removed_by_undercount": removed_candidates,
        "counterfactual_best": (candidate_record(counterfactual_best) if counterfactual_best is not None else None),
        "adopted": (candidate_record(evaluation.selected) if evaluation.selected is not None else None),
    }


def recover_native_pdf_table(
    table_input: NativeTableInput,
) -> NativeTableResult | None:
    """对一个已知表格区域运行全部候选并返回高置信 HTML 结果。"""

    evaluation = _evaluate_native_pdf_table(table_input)
    selected = evaluation.selected
    if selected is None:
        return None
    diagnostics = tuple(
        [
            f"candidate={selected.source}",
            f"score={selected.score:.4f}",
            f"text_capture={selected.text_capture:.4f}",
            f"structure_support={selected.structure_support:.4f}",
            f"row_stability={selected.row_stability:.4f}",
            f"column_stability={selected.column_stability:.4f}",
            f"order_consistency={selected.order_consistency:.4f}",
        ]
        + list(selected.issues)
    )
    return NativeTableResult(
        html=serialize_candidate_html(selected),
        rows=selected.rows,
        cols=selected.cols,
        cells=selected.cells,
        source=selected.source,
        confidence=selected.score,
        diagnostics=diagnostics,
    )


__all__ = [
    "coerce_native_table_rectangles",
    "coerce_native_table_rules",
    "recover_native_pdf_table",
]
