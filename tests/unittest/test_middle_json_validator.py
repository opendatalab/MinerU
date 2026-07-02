from mineru.schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION, validate_pages
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span


def _issue_keys(issues):
    return {(issue.severity, issue.code, issue.path) for issue in issues}


def _valid_page() -> PageInfo:
    return PageInfo(
        page_idx=0,
        page_size=(100, 200),
        para_blocks=[
            Block(
                index=0,
                type="text",
                bbox=(1.0, 2.0, 20.0, 30.0),
                lines=[
                    Line(
                        bbox=(1.0, 2.0, 20.0, 30.0),
                        spans=[Span(type="text", bbox=(1.0, 2.0, 20.0, 10.0), content="hello")],
                    )
                ],
            )
        ],
    )


def test_middle_json_schema_version_is_public_constant() -> None:
    assert MIDDLE_JSON_SCHEMA_VERSION == "1.0"


def test_validate_pages_accepts_valid_page_tree() -> None:
    assert validate_pages([_valid_page()]) == []


def test_validate_pages_reports_missing_required_page_and_block_fields() -> None:
    page = _valid_page()
    delattr(page, "page_idx")
    delattr(page.para_blocks[0], "type")

    issues = validate_pages([page])

    assert ("error", "missing_required_field", "pages[0].page_idx") in _issue_keys(issues)
    assert ("error", "missing_required_field", "pages[0].para_blocks[0].type") in _issue_keys(issues)


def test_validate_pages_distinguishes_unknown_and_invalid_bbox() -> None:
    page = _valid_page()
    page.para_blocks[0].bbox = (0.0, 0.0, 0.0, 0.0)
    page.para_blocks[0].lines[0].bbox = (3.0, 2.0, 1.0, 4.0)

    issues = validate_pages([page])

    assert ("warning", "bbox_unknown", "pages[0].para_blocks[0].bbox") in _issue_keys(issues)
    assert ("error", "bbox_invalid", "pages[0].para_blocks[0].lines[0].bbox") in _issue_keys(issues)


def test_validate_pages_recurses_into_child_blocks() -> None:
    page = _valid_page()
    page.para_blocks[0].blocks.append(
        Block(
            index=1,
            type="text",
            bbox=(2.0, 2.0, 10.0, 10.0),
            lines=[Line(bbox=(0.0, 0.0, 0.0, 0.0), spans=[])],
        )
    )

    issues = validate_pages([page])

    assert ("warning", "bbox_unknown", "pages[0].para_blocks[0].blocks[0].lines[0].bbox") in _issue_keys(issues)


def test_validate_pages_reports_wrong_node_types() -> None:
    page = _valid_page()
    page.para_blocks[0].lines.append(object())  # type: ignore[arg-type]

    issues = validate_pages([object(), page])  # type: ignore[list-item]

    assert ("error", "invalid_type", "pages[0]") in _issue_keys(issues)
    assert ("error", "invalid_type", "pages[1].para_blocks[0].lines[1]") in _issue_keys(issues)


def test_validate_pages_reports_unknown_block_type_and_bad_title_level() -> None:
    page = _valid_page()
    page.para_blocks[0].type = "unknown_type"
    page.para_blocks.append(
        Block(
            index=1,
            type=BlockType.TITLE,
            bbox=(2.0, 2.0, 20.0, 20.0),
            lines=[Line(bbox=(2.0, 2.0, 20.0, 20.0), spans=[])],
            level=0,
        )
    )

    issues = validate_pages([page])

    assert ("error", "block_type_unknown", "pages[0].para_blocks[0].type") in _issue_keys(issues)
    assert ("error", "title_level_invalid", "pages[0].para_blocks[1].level") in _issue_keys(issues)


def test_validate_pages_reports_block_index_order_and_duplicates() -> None:
    page = _valid_page()
    page.para_blocks.extend(
        [
            Block(index=2, type=BlockType.TEXT, bbox=(1.0, 1.0, 5.0, 5.0)),
            Block(index=1, type=BlockType.TEXT, bbox=(1.0, 1.0, 5.0, 5.0)),
            Block(index=1, type=BlockType.TEXT, bbox=(1.0, 1.0, 5.0, 5.0)),
        ]
    )

    issues = validate_pages([page])

    assert ("warning", "block_index_out_of_order", "pages[0].para_blocks[2].index") in _issue_keys(issues)
    assert ("error", "block_index_duplicate", "pages[0].para_blocks[3].index") in _issue_keys(issues)


def test_validate_pages_reports_bbox_outside_page_size() -> None:
    page = _valid_page()
    page.para_blocks[0].bbox = (1.0, 2.0, 200.0, 30.0)

    issues = validate_pages([page])

    assert ("warning", "bbox_out_of_page", "pages[0].para_blocks[0].bbox") in _issue_keys(issues)


def test_validate_pages_reports_span_content_contracts() -> None:
    page = PageInfo(
        page_idx=0,
        para_blocks=[
            Block(
                index=0,
                type=BlockType.TEXT,
                bbox=(1.0, 1.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(1.0, 1.0, 10.0, 10.0),
                        spans=[
                            Span(type=ContentType.TEXT, bbox=(1.0, 1.0, 2.0, 2.0)),
                            Span(type=ContentType.IMAGE, bbox=(2.0, 1.0, 3.0, 2.0)),
                            Span(type=ContentType.TABLE, bbox=(3.0, 1.0, 4.0, 2.0)),
                            Span(type=ContentType.INTERLINE_EQUATION, bbox=(4.0, 1.0, 5.0, 2.0)),
                        ],
                    )
                ],
            )
        ],
    )

    issues = validate_pages([page])

    assert ("warning", "span_content_missing", "pages[0].para_blocks[0].lines[0].spans[0].content") in _issue_keys(issues)
    assert ("warning", "span_image_missing", "pages[0].para_blocks[0].lines[0].spans[1]") in _issue_keys(issues)
    assert ("warning", "span_table_missing", "pages[0].para_blocks[0].lines[0].spans[2]") in _issue_keys(issues)
    assert ("warning", "span_equation_missing", "pages[0].para_blocks[0].lines[0].spans[3]") in _issue_keys(issues)


def test_validate_pages_warns_legacy_page_backend() -> None:
    page = _valid_page()
    page._backend = "hybrid"

    issues = validate_pages([page])

    assert ("warning", "legacy_backend", "pages[0]._backend") in _issue_keys(issues)
