from _span_test_utils import inline as _inline
from dataclasses import dataclass
from typing import Literal

from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import (
    BLOCK_TYPES,
    BlockBase,
    BlockType,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    InlineContentBlock,
    PageInfo,
    ParagraphTitleBlock,
    TextBlock,
)

IssueSeverity = Literal["error", "warning"]

KNOWN_BLOCK_TYPES = BLOCK_TYPES
STRING_CONTENT_BLOCK_TYPES = {
    BlockType.EQUATION,
    BlockType.TABLE_BODY,
    BlockType.CODE_BODY,
    BlockType.IMAGE_BODY,
    BlockType.CHART_BODY,
}
VISUAL_PARENT_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
}
VISUAL_BODY_TYPE_BY_PARENT = {
    BlockType.IMAGE: BlockType.IMAGE_BODY,
    BlockType.TABLE: BlockType.TABLE_BODY,
    BlockType.CHART: BlockType.CHART_BODY,
    BlockType.CODE: BlockType.CODE_BODY,
}
CONTAINER_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
    BlockType.LIST,
    BlockType.INDEX,
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: IssueSeverity
    code: str
    path: str
    message: str


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

        if not _has_field(page, "blocks"):
            issues.append(_missing(f"{page_path}.blocks"))
            continue
        if not isinstance(page.blocks, list):
            issues.append(_invalid_type(f"{page_path}.blocks", "list"))
            continue

        _validate_block_list(page.blocks, f"{page_path}.blocks", issues)
    return issues


def _validate_block_list(
    blocks: list[object],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    seen_indexes: set[int] = set()
    previous_index: int | None = None
    for block_index, block in enumerate(blocks):
        block_path = f"{path}[{block_index}]"
        if not isinstance(block, BlockBase):
            issues.append(_invalid_type(block_path, "BlockBase"))
            continue

        if _has_field(block, "index") and block.index is not None and _is_int(block.index):
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

        _validate_block(block, block_path, issues)


def _validate_block(block: BlockBase, path: str, issues: list[ValidationIssue]) -> None:
    block_type = getattr(block, "type", None)
    if not _has_field(block, "type"):
        issues.append(_missing(f"{path}.type"))
    elif not isinstance(block_type, str):
        issues.append(_invalid_type(f"{path}.type", "str"))
    elif block_type not in KNOWN_BLOCK_TYPES:
        issues.append(
            ValidationIssue(
                severity="error",
                code="block_type_unknown",
                path=f"{path}.type",
                message=f"unknown block type: {block_type}",
            )
        )
    elif block_type in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
        _validate_title_level(block, f"{path}.level", issues)

    if _has_field(block, "index") and block.index is not None and not _is_int(block.index):
        issues.append(_invalid_type(f"{path}.index", "int"))
    elif _has_field(block, "index") and block.index is not None and block.index < 0:
        issues.append(_invalid_value(f"{path}.index", "block index must be non-negative."))

    if _has_field(block, "bbox") and block.bbox is not None:
        _validate_bbox(block.bbox, f"{path}.bbox", issues)

    if isinstance(block, InlineContentBlock):
        _validate_inline_content(block, f"{path}.content", issues)
    elif block_type in STRING_CONTENT_BLOCK_TYPES:
        _validate_string_content(block, f"{path}.content", issues)

    if block_type in VISUAL_PARENT_TYPES:
        _validate_visual_block_children(block, f"{path}.content", issues)

    if block_type in CONTAINER_BLOCK_TYPES:
        content = getattr(block, "content", None)
        if isinstance(content, list):
            _validate_block_list(content, f"{path}.content", issues)


def _validate_string_content(block: BlockBase, path: str, issues: list[ValidationIssue]) -> None:
    content = getattr(block, "content", None)
    block_type = getattr(block, "type", None)
    if content is None:
        issues.append(_missing(path))
        return
    if not isinstance(content, str):
        issues.append(_invalid_type(path, "str"))
        return
    if not content and block_type == BlockType.CODE_BODY:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="block_content_missing",
                path=path,
                message=f"{block_type} block should provide content.",
            )
        )


def _validate_inline_content(block: InlineContentBlock, path: str, issues: list[ValidationIssue]) -> None:
    """验证行内内容是非空 Span 列表。"""
    content = getattr(block, "content", None)
    if not isinstance(content, list):
        issues.append(_invalid_type(path, "list[InlineSpan]"))
        return
    if not content:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="block_content_missing",
                path=path,
                message=f"{block.type} block should provide content.",
            )
        )


def _validate_visual_block_children(block: BlockBase, path: str, issues: list[ValidationIssue]) -> None:
    """Validate visual parent block has exactly one body, and body index equals parent index."""
    content = getattr(block, "content", None)
    block_type = getattr(block, "type", None)
    if not isinstance(content, list):
        issues.append(_invalid_type(path, "list"))
        return

    body_type = VISUAL_BODY_TYPE_BY_PARENT.get(block_type)
    if body_type is None:
        return

    bodies = [child for child in content if isinstance(child, BlockBase) and getattr(child, "type", None) == body_type]
    if len(bodies) != 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="visual_block_body_count",
                path=path,
                message=f"{block_type} must contain exactly one {body_type}.",
            )
        )
        return

    body = bodies[0]
    if block.index is not None and body.index is not None and block.index != body.index:
        issues.append(
            ValidationIssue(
                severity="error",
                code="visual_block_body_index_mismatch",
                path=f"{path}[body].index",
                message=f"{block_type} body index must equal parent index.",
            )
        )


def _validate_bbox(
    bbox: object,
    path: str,
    issues: list[ValidationIssue],
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
    if x1 <= x0 or y1 <= y0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="bbox_invalid",
                path=path,
                message="bbox coordinates must satisfy x1 > x0 and y1 > y0.",
            )
        )
        return
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="bbox_out_of_bounds",
                path=path,
                message="bbox values should be normalized to [0, 1].",
            )
        )


def _validate_title_level(block: BlockBase, path: str, issues: list[ValidationIssue]) -> None:
    level = getattr(block, "level", None)
    if level is None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="title_level_missing",
                path=path,
                message="title block should provide a positive integer level.",
            )
        )
        return
    if not _is_int(level) or level < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="title_level_invalid",
                path=path,
                message="title level must be a positive integer.",
            )
        )


def _is_bbox(bbox: object) -> bool:
    return (
        isinstance(bbox, tuple | list)
        and len(bbox) == 4
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bbox)
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


def _issue_keys(issues):
    return {(issue.severity, issue.code, issue.path) for issue in issues}


def _valid_page() -> PageInfo:
    return PageInfo(
        page_idx=0,
        blocks=[
            TextBlock(type=BlockType.TEXT, index=0, bbox=(0.1, 0.1, 0.2, 0.2), content=_inline("hello")),
        ],
    )


def test_middle_json_schema_version_is_public_constant() -> None:
    assert MIDDLE_JSON_SCHEMA_VERSION == "2.0"


def test_validate_pages_accepts_valid_page_tree() -> None:
    assert validate_pages([_valid_page()]) == []


def test_validate_pages_reports_missing_required_page_and_block_fields() -> None:
    page = _valid_page()
    page_no_idx = PageInfo.model_construct(blocks=page.blocks)
    block_no_type = TextBlock.model_construct(index=0, content="hello", bbox=(0.1, 0.1, 0.2, 0.2))
    page_no_type = PageInfo.model_construct(page_idx=0, blocks=[block_no_type])

    issues = validate_pages([page_no_idx, page_no_type])

    assert ("error", "missing_required_field", "pages[0].page_idx") in _issue_keys(issues)
    assert ("error", "missing_required_field", "pages[1].blocks[0].type") in _issue_keys(issues)


def test_validate_pages_distinguishes_unknown_and_invalid_bbox() -> None:
    """验证 bbox 坐标 x1 <= x0 或 y1 <= y0 时报 error。"""
    invalid_bbox_block = TextBlock.model_construct(
        type=BlockType.TEXT,
        index=0,
        content="hello",
        bbox=(0.5, 0.1, 0.2, 0.2),  # x1 < x0
    )
    page = PageInfo.model_construct(page_idx=0, blocks=[invalid_bbox_block])

    issues = validate_pages([page])

    assert ("error", "bbox_invalid", "pages[0].blocks[0].bbox") in _issue_keys(issues)


def test_validate_pages_reports_bbox_out_of_bounds() -> None:
    """验证 bbox 值超出 [0, 1] 范围时报 warning。"""
    out_of_bounds_block = TextBlock.model_construct(
        type=BlockType.TEXT,
        index=0,
        content="hello",
        bbox=(0.1, 0.1, 1.5, 0.2),  # x1 > 1.0
    )
    page = PageInfo.model_construct(page_idx=0, blocks=[out_of_bounds_block])

    issues = validate_pages([page])

    assert ("warning", "bbox_out_of_bounds", "pages[0].blocks[0].bbox") in _issue_keys(issues)


def test_validate_pages_recurses_into_child_blocks() -> None:
    """验证 visual parent block 的 content 子块也会被递归校验。"""
    image_block = ImageBlock(
        type=BlockType.IMAGE,
        index=0,
        bbox=(0.0, 0.0, 0.5, 0.5),
        content=[
            ImageBodyBlock(
                type=BlockType.IMAGE_BODY,
                index=0,
                bbox=(0.0, 0.0, 0.5, 0.5),
                content="",
            ),
        ],
    )
    page = PageInfo(page_idx=0, blocks=[image_block])

    issues = validate_pages([page])

    # image_body 允许空 content，不应产生 content 警告
    assert ("warning", "block_content_missing", "pages[0].blocks[0].content[0].content") not in _issue_keys(issues)


def test_validate_pages_reports_wrong_node_types() -> None:
    page = _valid_page()
    page.blocks.append(object())  # type: ignore[arg-type]

    issues = validate_pages([object(), page])  # type: ignore[list-item]

    assert ("error", "invalid_type", "pages[0]") in _issue_keys(issues)
    assert ("error", "invalid_type", "pages[1].blocks[1]") in _issue_keys(issues)


def test_validate_pages_reports_unknown_block_type_and_bad_title_level() -> None:
    bad_type_block = TextBlock.model_construct(type="unknown_type", index=0, content="hello", bbox=(0.1, 0.1, 0.2, 0.2))
    bad_level_block = ParagraphTitleBlock.model_construct(
        type=BlockType.PARAGRAPH_TITLE,
        index=1,
        content="section",
        bbox=(0.1, 0.1, 0.2, 0.2),
        level=0,
    )
    page = PageInfo.model_construct(page_idx=0, blocks=[bad_type_block, bad_level_block])

    issues = validate_pages([page])

    assert ("error", "block_type_unknown", "pages[0].blocks[0].type") in _issue_keys(issues)
    assert ("error", "title_level_invalid", "pages[0].blocks[1].level") in _issue_keys(issues)


def test_validate_pages_reports_block_index_order_and_duplicates() -> None:
    """验证 block index 乱序和重复会被报告。

    PageInfo 顶层 index 在构造时已强制 unique+ascending，这里用 model_construct 绕过
    Pydantic 校验来测试 validator 自身的检测能力。
    """
    page = PageInfo.model_construct(
        page_idx=0,
        blocks=[
            TextBlock(type=BlockType.TEXT, index=2, bbox=(0.1, 0.1, 0.2, 0.2), content=_inline("a")),
            TextBlock(type=BlockType.TEXT, index=1, bbox=(0.1, 0.1, 0.2, 0.2), content=_inline("b")),
            TextBlock(type=BlockType.TEXT, index=1, bbox=(0.1, 0.1, 0.2, 0.2), content=_inline("c")),
        ],
    )

    issues = validate_pages([page])

    assert ("warning", "block_index_out_of_order", "pages[0].blocks[1].index") in _issue_keys(issues)
    assert ("error", "block_index_duplicate", "pages[0].blocks[2].index") in _issue_keys(issues)


def test_validate_pages_reports_inline_content_contracts() -> None:
    """验证 Span 内容 block 的空 content 会触发 warning。"""
    empty_text = TextBlock.model_construct(type=BlockType.TEXT, index=0, content=[], bbox=(0.1, 0.1, 0.2, 0.2))
    page = PageInfo(page_idx=0, blocks=[empty_text])

    issues = validate_pages([page])

    assert ("warning", "block_content_missing", "pages[0].blocks[0].content") in _issue_keys(issues)


def test_validate_pages_reports_visual_block_body_count_mismatch() -> None:
    """验证视觉父块必须有且仅有一个 body。

    ImageBlock 构造时已强制 exactly-one-body，用 model_construct 绕过以测试 validator。
    """
    image_block = ImageBlock.model_construct(
        type=BlockType.IMAGE,
        index=0,
        bbox=(0.0, 0.0, 0.5, 0.5),
        content=[
            ImageAnnotationBlock.model_construct(
                type=BlockType.IMAGE_CAPTION,
                index=1,
                content=_inline("caption"),
                bbox=(0.0, 0.0, 0.5, 0.5),
            ),
        ],
    )
    page = PageInfo.model_construct(page_idx=0, blocks=[image_block])

    issues = validate_pages([page])

    assert ("error", "visual_block_body_count", "pages[0].blocks[0].content") in _issue_keys(issues)


def test_validate_pages_reports_visual_block_body_index_mismatch() -> None:
    """验证视觉父块 body index 必须等于 parent index。

    ImageBlock 构造时已强制 body index == parent index，用 model_construct 绕过以测试 validator。
    """
    image_block = ImageBlock.model_construct(
        type=BlockType.IMAGE,
        index=0,
        bbox=(0.0, 0.0, 0.5, 0.5),
        content=[
            ImageBodyBlock.model_construct(type=BlockType.IMAGE_BODY, index=5, content="", bbox=(0.0, 0.0, 0.5, 0.5)),
        ],
    )
    page = PageInfo.model_construct(page_idx=0, blocks=[image_block])

    issues = validate_pages([page])

    assert ("error", "visual_block_body_index_mismatch", "pages[0].blocks[0].content[body].index") in _issue_keys(issues)
