from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


CONTROL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>|<[a-z]+>")
LAYOUT_RE = re.compile(
    r"<\|box_start\|>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*"
    r"<\|box_end\|>\s*<\|ref_start\|>\s*([^<]+?)\s*<\|ref_end\|>"
)
LAYOUT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class LayoutBlock:
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    baseline_ok: bool
    candidate_ok: bool
    baseline_chars: int
    candidate_chars: int
    char_ratio: float | None
    similarity: float | None
    candidate_control_token_ratio: float
    candidate_max_control_repeat: int
    layout_baseline_boxes: int | None = None
    layout_candidate_boxes: int | None = None
    layout_matched_boxes: int | None = None
    layout_precision: float | None = None
    layout_recall: float | None = None
    layout_f1: float | None = None


@dataclass(frozen=True)
class QualityThresholds:
    min_similarity: float | None = None
    similarity_cases: tuple[str, ...] = ()
    max_control_token_ratio: float | None = None
    max_control_repeat: int | None = None


class QualityGateError(RuntimeError):
    pass


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _control_tokens(text: str) -> list[str]:
    return CONTROL_TOKEN_RE.findall(text)


def control_token_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(len(token) for token in _control_tokens(text)) / len(text)


def max_consecutive_control_repeat(text: str) -> int:
    max_repeat = 0
    current_token: str | None = None
    current_count = 0
    for token in _control_tokens(text):
        if token == current_token:
            current_count += 1
        else:
            current_token = token
            current_count = 1
        max_repeat = max(max_repeat, current_count)
    return max_repeat


def parse_layout_blocks(text: str) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for match in LAYOUT_RE.finditer(text):
        x1, y1, x2, y2 = (float(value) for value in match.group(1, 2, 3, 4))
        if any(value < 0 or value > 1000 for value in (x1, y1, x2, y2)):
            continue
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if left == right or top == bottom:
            continue
        blocks.append(
            LayoutBlock(
                label=match.group(5).strip().lower(),
                bbox=(left, top, right, bottom),
            )
        )
    return blocks


def _bbox_iou(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> float:
    left = max(lhs[0], rhs[0])
    top = max(lhs[1], rhs[1])
    right = min(lhs[2], rhs[2])
    bottom = min(lhs[3], rhs[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    lhs_area = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1])
    rhs_area = (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])
    union = lhs_area + rhs_area - intersection
    return intersection / union if union else 0.0


def compare_layout_blocks(
    baseline_blocks: list[LayoutBlock],
    candidate_blocks: list[LayoutBlock],
    *,
    iou_threshold: float = LAYOUT_IOU_THRESHOLD,
) -> tuple[int, float, float, float]:
    pairs: list[tuple[float, int, int]] = []
    for baseline_index, baseline_block in enumerate(baseline_blocks):
        for candidate_index, candidate_block in enumerate(candidate_blocks):
            if baseline_block.label != candidate_block.label:
                continue
            iou = _bbox_iou(baseline_block.bbox, candidate_block.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, baseline_index, candidate_index))

    matched_baseline: set[int] = set()
    matched_candidate: set[int] = set()
    matches = 0
    for _iou, baseline_index, candidate_index in sorted(pairs, reverse=True):
        if baseline_index in matched_baseline or candidate_index in matched_candidate:
            continue
        matched_baseline.add(baseline_index)
        matched_candidate.add(candidate_index)
        matches += 1

    precision = matches / len(candidate_blocks) if candidate_blocks else 0.0
    recall = matches / len(baseline_blocks) if baseline_blocks else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return matches, precision, recall, f1


def compare_case(
    case_id: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> CaseComparison:
    baseline_text = baseline.get("output_text", "") if baseline else ""
    candidate_text = candidate.get("output_text", "") if candidate else ""
    baseline_chars = len(baseline_text)
    candidate_chars = len(candidate_text)
    char_ratio = (
        candidate_chars / baseline_chars if baseline_chars and candidate_chars else None
    )
    similarity = None
    if baseline_text and candidate_text:
        similarity = SequenceMatcher(
            None,
            _normalize_text(baseline_text),
            _normalize_text(candidate_text),
            autojunk=False,
        ).ratio()
    layout_baseline_boxes = None
    layout_candidate_boxes = None
    layout_matched_boxes = None
    layout_precision = None
    layout_recall = None
    layout_f1 = None
    if case_id == "layout":
        baseline_blocks = parse_layout_blocks(baseline_text)
        candidate_blocks = parse_layout_blocks(candidate_text)
        layout_baseline_boxes = len(baseline_blocks)
        layout_candidate_boxes = len(candidate_blocks)
        (
            layout_matched_boxes,
            layout_precision,
            layout_recall,
            layout_f1,
        ) = compare_layout_blocks(baseline_blocks, candidate_blocks)
    return CaseComparison(
        case_id=case_id,
        baseline_ok=bool(baseline and baseline.get("ok")),
        candidate_ok=bool(candidate and candidate.get("ok")),
        baseline_chars=baseline_chars,
        candidate_chars=candidate_chars,
        char_ratio=char_ratio,
        similarity=similarity,
        candidate_control_token_ratio=control_token_ratio(candidate_text),
        candidate_max_control_repeat=max_consecutive_control_repeat(candidate_text),
        layout_baseline_boxes=layout_baseline_boxes,
        layout_candidate_boxes=layout_candidate_boxes,
        layout_matched_boxes=layout_matched_boxes,
        layout_precision=layout_precision,
        layout_recall=layout_recall,
        layout_f1=layout_f1,
    )


def compare_results(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
) -> list[CaseComparison]:
    case_ids = sorted(set(baseline_results) | set(candidate_results))
    return [
        compare_case(
            case_id,
            baseline_results.get(case_id),
            candidate_results.get(case_id),
        )
        for case_id in case_ids
    ]


def summarize_comparisons(comparisons: list[CaseComparison]) -> dict[str, Any]:
    similarities = [
        item.similarity for item in comparisons if item.similarity is not None
    ]
    ratios = [item.char_ratio for item in comparisons if item.char_ratio is not None]
    layout_f1s = [item.layout_f1 for item in comparisons if item.layout_f1 is not None]
    return {
        "num_cases": len(comparisons),
        "num_candidate_ok": sum(item.candidate_ok for item in comparisons),
        "mean_similarity": (
            sum(similarities) / len(similarities) if similarities else None
        ),
        "min_similarity": min(similarities) if similarities else None,
        "mean_char_ratio": sum(ratios) / len(ratios) if ratios else None,
        "max_control_token_ratio": max(
            (item.candidate_control_token_ratio for item in comparisons),
            default=0.0,
        ),
        "max_control_repeat": max(
            (item.candidate_max_control_repeat for item in comparisons),
            default=0,
        ),
        "mean_layout_f1": (
            sum(layout_f1s) / len(layout_f1s) if layout_f1s else None
        ),
        "min_layout_f1": min(layout_f1s) if layout_f1s else None,
    }


def assert_quality_thresholds(
    comparisons: list[CaseComparison],
    thresholds: QualityThresholds,
) -> None:
    failures: list[str] = []
    if thresholds.min_similarity is not None:
        cases = set(thresholds.similarity_cases)
        for item in comparisons:
            if cases and item.case_id not in cases:
                continue
            if item.similarity is None or item.similarity < thresholds.min_similarity:
                failures.append(
                    f"{item.case_id}: similarity {item.similarity} "
                    f"< min_similarity {thresholds.min_similarity}"
                )
    if thresholds.max_control_token_ratio is not None:
        for item in comparisons:
            if item.candidate_control_token_ratio > thresholds.max_control_token_ratio:
                failures.append(
                    f"{item.case_id}: control_token_ratio "
                    f"{item.candidate_control_token_ratio} "
                    f"> max_control_token_ratio "
                    f"{thresholds.max_control_token_ratio}"
                )
    if thresholds.max_control_repeat is not None:
        for item in comparisons:
            if item.candidate_max_control_repeat > thresholds.max_control_repeat:
                failures.append(
                    f"{item.case_id}: max_control_repeat "
                    f"{item.candidate_max_control_repeat} "
                    f"> max_control_repeat {thresholds.max_control_repeat}"
                )
    if failures:
        raise QualityGateError("; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument(
        "--similarity-cases",
        default="",
        help="comma-separated case ids to apply --min-similarity to",
    )
    parser.add_argument("--max-control-token-ratio", type=float, default=None)
    parser.add_argument("--max-control-repeat", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = compare_results(read_results(args.baseline), read_results(args.candidate))
    payload = {
        "summary": summarize_comparisons(comparisons),
        "cases": [asdict(item) for item in comparisons],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    similarity_cases = tuple(
        case_id.strip()
        for case_id in args.similarity_cases.split(",")
        if case_id.strip()
    )
    assert_quality_thresholds(
        comparisons,
        QualityThresholds(
            min_similarity=args.min_similarity,
            similarity_cases=similarity_cases,
            max_control_token_ratio=args.max_control_token_ratio,
            max_control_repeat=args.max_control_repeat,
        ),
    )


if __name__ == "__main__":
    main()
