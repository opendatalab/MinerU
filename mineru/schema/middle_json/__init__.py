# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...types import BLOCK_TYPES, EMPTY_BBOX, Block, BlockType, ContentType, Line, PageInfo, Span
from .envelope import MIDDLE_JSON_SCHEMA_VERSION

__all__ = [
    "MIDDLE_JSON_SCHEMA_VERSION",
    "ValidationIssue",
    "bbox_known",
    "validate_pages",
]

IssueSeverity = Literal["error", "warning"]

KNOWN_BLOCK_TYPES = BLOCK_TYPES | {"list_item"}
TEXT_SPAN_TYPES = {ContentType.TEXT}
IMAGE_SPAN_TYPES = {ContentType.IMAGE}
TABLE_SPAN_TYPES = {ContentType.TABLE}
EQUATION_SPAN_TYPES = {ContentType.EQUATION, ContentType.INLINE_EQUATION, ContentType.INTERLINE_EQUATION}


@dataclass(frozen=True)
class ValidationIssue:
    severity: IssueSeverity
    code: str
    path: str
    message: str


def bbox_known(bbox: object) -> bool:
    return _is_bbox(bbox) and tuple(float(v) for v in bbox) != EMPTY_BBOX


def validate_pages(pages: list[PageInfo]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(pages, list):
        return [
            ValidationIssue(
                severity="error",
                code="pages_invalid",
                path="pages",
                message="pages must be a list.",
            )
        ]

    for page_index, page in enumerate(pages):
        page_path = f"pages[{page_index}]"
        if not isinstance(page, PageInfo):
            issues.append(_invalid_type(page_path, "PageInfo"))
            continue

        if not _has_field(page, "page_idx"):
            issues.append(_missing(f"{page_path}.page_idx"))
        elif not _is_int(page.page_idx):
            issues.append(_invalid_type(f"{page_path}.page_idx", "int"))
        elif page.page_idx < 0:
            issues.append(_invalid_value(f"{page_path}.page_idx", "page_idx must be non-negative."))

        page_size: tuple[float, float] | None = None
        if _has_field(page, "page_size") and page.page_size is not None:
            _validate_page_size(page.page_size, f"{page_path}.page_size", issues)
            if _is_page_size(page.page_size):
                page_size = (float(page.page_size[0]), float(page.page_size[1]))

        if getattr(page, "_backend", None) is not None:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="legacy_backend",
                    path=f"{page_path}._backend",
                    message="_backend is a legacy/internal page field.",
                )
            )

        for attr in ("preproc_blocks", "para_blocks", "discarded_blocks"):
            if not _has_field(page, attr):
                continue
            blocks = getattr(page, attr)
            if not isinstance(blocks, list):
                issues.append(_invalid_type(f"{page_path}.{attr}", "list"))
                continue
            _validate_block_list(blocks, f"{page_path}.{attr}", issues, page_size)
    return issues


def _validate_block_list(
    blocks: list[object],
    path: str,
    issues: list[ValidationIssue],
    page_size: tuple[float, float] | None,
) -> None:
    seen_indexes: set[int] = set()
    previous_index: int | None = None
    for block_index, block in enumerate(blocks):
        block_path = f"{path}[{block_index}]"
        if not isinstance(block, Block):
            issues.append(_invalid_type(block_path, "Block"))
            continue

        if _has_field(block, "index") and _is_int(block.index):
            if block.index in seen_indexes:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="block_index_duplicate",
                        path=f"{block_path}.index",
                        message=f"block index {block.index} is duplicated within {path}.",
                    )
                )
            seen_indexes.add(block.index)
            if previous_index is not None and block.index < previous_index:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="block_index_out_of_order",
                        path=f"{block_path}.index",
                        message="block indexes should be sorted in ascending order.",
                    )
                )
            previous_index = block.index

        _validate_block(block, block_path, issues, page_size)


def _validate_block(block: Block, path: str, issues: list[ValidationIssue], page_size: tuple[float, float] | None) -> None:
    for field_name in ("index", "type", "bbox"):
        if not _has_field(block, field_name):
            issues.append(_missing(f"{path}.{field_name}"))

    if _has_field(block, "index") and not _is_int(block.index):
        issues.append(_invalid_type(f"{path}.index", "int"))
    elif _has_field(block, "index") and block.index < 0:
        issues.append(_invalid_value(f"{path}.index", "block index must be non-negative."))

    if _has_field(block, "type") and not isinstance(block.type, str):
        issues.append(_invalid_type(f"{path}.type", "str"))
    elif _has_field(block, "type") and block.type not in KNOWN_BLOCK_TYPES:
        issues.append(
            ValidationIssue(
                severity="error",
                code="block_type_unknown",
                path=f"{path}.type",
                message=f"unknown block type: {block.type}",
            )
        )
    elif _has_field(block, "type") and block.type == BlockType.TITLE:
        _validate_title_level(block, f"{path}.level", issues)

    if _has_field(block, "bbox"):
        _validate_bbox(block.bbox, f"{path}.bbox", issues, page_size)

    if not _has_field(block, "lines") and not _has_field(block, "blocks"):
        issues.append(_missing(f"{path}.lines_or_blocks"))

    if _has_field(block, "lines"):
        if not isinstance(block.lines, list):
            issues.append(_invalid_type(f"{path}.lines", "list"))
        else:
            for line_index, line in enumerate(block.lines):
                line_path = f"{path}.lines[{line_index}]"
                if not isinstance(line, Line):
                    issues.append(_invalid_type(line_path, "Line"))
                    continue
                _validate_line(line, line_path, issues, page_size)

    if _has_field(block, "blocks"):
        if not isinstance(block.blocks, list):
            issues.append(_invalid_type(f"{path}.blocks", "list"))
        else:
            _validate_block_list(block.blocks, f"{path}.blocks", issues, page_size)


def _validate_line(line: Line, path: str, issues: list[ValidationIssue], page_size: tuple[float, float] | None) -> None:
    if not _has_field(line, "bbox"):
        issues.append(_missing(f"{path}.bbox"))
    else:
        _validate_bbox(line.bbox, f"{path}.bbox", issues, page_size)

    if not _has_field(line, "spans"):
        issues.append(_missing(f"{path}.spans"))
        return
    if not isinstance(line.spans, list):
        issues.append(_invalid_type(f"{path}.spans", "list"))
        return
    for span_index, span in enumerate(line.spans):
        span_path = f"{path}.spans[{span_index}]"
        if not isinstance(span, Span):
            issues.append(_invalid_type(span_path, "Span"))
            continue
        _validate_span(span, span_path, issues, page_size)


def _validate_span(span: Span, path: str, issues: list[ValidationIssue], page_size: tuple[float, float] | None) -> None:
    for field_name in ("type", "bbox"):
        if not _has_field(span, field_name):
            issues.append(_missing(f"{path}.{field_name}"))

    if _has_field(span, "type") and not isinstance(span.type, str):
        issues.append(_invalid_type(f"{path}.type", "str"))
    if _has_field(span, "bbox"):
        _validate_bbox(span.bbox, f"{path}.bbox", issues, page_size)
    if _has_field(span, "type") and isinstance(span.type, str):
        _validate_span_content(span, path, issues)


def _validate_bbox(
    bbox: object,
    path: str,
    issues: list[ValidationIssue],
    page_size: tuple[float, float] | None = None,
) -> None:
    if not _is_bbox(bbox):
        issues.append(
            ValidationIssue(
                severity="error",
                code="bbox_invalid",
                path=path,
                message="bbox must contain four numeric coordinates.",
            )
        )
        return

    x0, y0, x1, y1 = (float(v) for v in bbox)
    if (x0, y0, x1, y1) == EMPTY_BBOX:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="bbox_unknown",
                path=path,
                message="bbox is the unknown bbox sentinel.",
            )
        )
        return
    if x1 < x0 or y1 < y0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="bbox_invalid",
                path=path,
                message="bbox coordinates must satisfy x1 >= x0 and y1 >= y0.",
            )
        )
        return
    if page_size is not None:
        page_w, page_h = page_size
        if x0 < 0 or y0 < 0 or x1 > page_w or y1 > page_h:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="bbox_out_of_page",
                    path=path,
                    message="bbox is outside page_size bounds.",
                )
            )


def _validate_page_size(page_size: object, path: str, issues: list[ValidationIssue]) -> None:
    if not _is_page_size(page_size):
        issues.append(
            ValidationIssue(
                severity="error",
                code="page_size_invalid",
                path=path,
                message="page_size must contain width and height.",
            )
        )
        return
    if float(page_size[0]) <= 0 or float(page_size[1]) <= 0:
        issues.append(_invalid_value(path, "page_size width and height must be positive."))


def _validate_title_level(block: Block, path: str, issues: list[ValidationIssue]) -> None:
    if block.level is None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="title_level_missing",
                path=path,
                message="title block should provide a positive integer level.",
            )
        )
        return
    if not _is_int(block.level) or block.level < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="title_level_invalid",
                path=path,
                message="title level must be a positive integer.",
            )
        )


def _validate_span_content(span: Span, path: str, issues: list[ValidationIssue]) -> None:
    if span.type in TEXT_SPAN_TYPES and not span.content:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="span_content_missing",
                path=f"{path}.content",
                message="text span should provide content.",
            )
        )
    elif span.type in IMAGE_SPAN_TYPES and not (span.image_path or span.image_base64 or span.html):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="span_image_missing",
                path=path,
                message="image span should provide image_path, image_base64, or html.",
            )
        )
    elif span.type in TABLE_SPAN_TYPES and not (span.html or span.content):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="span_table_missing",
                path=path,
                message="table span should provide html or content.",
            )
        )
    elif span.type in EQUATION_SPAN_TYPES and not (span.content or span.latex):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="span_equation_missing",
                path=path,
                message="equation span should provide content or latex.",
            )
        )


def _is_bbox(bbox: object) -> bool:
    return (
        isinstance(bbox, tuple | list)
        and len(bbox) == 4
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bbox)
    )


def _is_page_size(page_size: object) -> bool:
    return (
        isinstance(page_size, tuple | list)
        and len(page_size) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in page_size)
    )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_field(obj: object, name: str) -> bool:
    return hasattr(obj, name)


def _missing(path: str) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        code="missing_required_field",
        path=path,
        message=f"{path} is required.",
    )


def _invalid_type(path: str, expected: str) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        code="invalid_type",
        path=path,
        message=f"{path} must be {expected}.",
    )


def _invalid_value(path: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        code="invalid_value",
        path=path,
        message=message,
    )
