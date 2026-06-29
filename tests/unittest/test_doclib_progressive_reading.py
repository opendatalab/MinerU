import pytest

from mineru.doclib.server import _next_content_request, _normalize_content_page_range, _page_markdown_blocks, _render_progressive_markdown
from mineru.doclib.types import ContentRange
from mineru.errors import InvalidRequestError
from mineru.types import Block, BlockType, ContentType, Line, PageInfo, Span


def _page(page_idx: int, *texts: str) -> PageInfo:
    blocks = [
        Block(
            index=index,
            type=BlockType.TEXT,
            bbox=(0, 0, 10, 10),
            lines=[Line(bbox=(0, 0, 10, 10), spans=[Span(type=ContentType.TEXT, bbox=(0, 0, 10, 10), content=text)])],
        )
        for index, text in enumerate(texts)
    ]
    return PageInfo(page_idx=page_idx, page_size=(100, 100), para_blocks=blocks, _backend="pipeline")


def test_paginated_default_content_pages_are_first_ten() -> None:
    doc = {"page_count": 38}

    assert _normalize_content_page_range(None, None, doc) == "1~10"
    assert _normalize_content_page_range("all", None, doc) == "1~38"
    assert _normalize_content_page_range("1~5,-5~-1", None, doc) == "1~5,34~38"


def test_content_page_range_uses_available_subset_and_merges_ranges() -> None:
    doc = {"page_count": 5}

    assert _normalize_content_page_range("1~10,3,4~5", None, doc) == "1~5"


def test_content_page_range_rejects_empty_available_subset() -> None:
    doc = {"page_count": 5}

    with pytest.raises(InvalidRequestError) as exc_info:
        _normalize_content_page_range("6~10", None, doc)

    assert exc_info.value.code == "page_range_invalid"


def test_paginated_page_boundary_truncation_suggests_pages_only() -> None:
    pages = [_page(index, f"page {index + 1}") for index in range(10)]
    rendered = _render_progressive_markdown(pages, short_id="ab12cd3", tier="standard", after=None, limit=40, add_markers=False)
    next_request = _next_content_request(
        rendered=rendered,
        request_page_range="1~10",
        after=None,
        page_count=38,
        paginated=True,
    )

    assert rendered.truncated is True
    assert rendered.content_ranges[0].page_range == "1~5"
    assert next_request is not None
    assert next_request.page_range == "6~10"
    assert next_request.after is None


def test_paginated_long_single_page_truncation_keeps_pages_with_after() -> None:
    pages = [_page(6, "word " * 100)]
    rendered = _render_progressive_markdown(pages, short_id="ab12cd3", tier="standard", after=None, limit=40, add_markers=False)
    next_request = _next_content_request(
        rendered=rendered,
        request_page_range="7~10",
        after=None,
        page_count=38,
        paginated=True,
    )

    assert rendered.truncated is True
    assert rendered.content_ranges[-1].end.startswith("doc:ab12cd3/tier:standard/page:7/block:1/char:")
    assert next_request is not None
    assert next_request.page_range == "7~10"
    assert next_request.after == rendered.content_ranges[-1].end


def test_non_paginated_truncation_can_use_after_only() -> None:
    rendered = _render_progressive_markdown(
        [_page(0, "block 1", "block 2", "block 3")],
        short_id="ab12cd3",
        tier="standard",
        after=None,
        limit=15,
        add_markers=False,
    )
    next_request = _next_content_request(
        rendered=rendered,
        request_page_range=None,
        after=None,
        page_count=1,
        paginated=False,
    )

    assert rendered.truncated is True
    assert next_request is not None
    assert next_request.page_range is None
    assert next_request.after == rendered.content_ranges[-1].end


def test_non_contiguous_pages_suggest_after_last_requested_page() -> None:
    rendered = _render_progressive_markdown(
        [_page(0, "page 1"), _page(4, "page 5"), _page(19, "page 20"), _page(24, "page 25")],
        short_id="ab12cd3",
        tier="standard",
        after=None,
        limit=30000,
        add_markers=False,
    )
    next_request = _next_content_request(
        rendered=rendered,
        request_page_range="1~5,20~25",
        after=None,
        page_count=38,
        paginated=True,
    )

    assert next_request is not None
    assert next_request.page_range == "26~35"


def test_truncated_false_can_still_have_next_request() -> None:
    rendered = _render_progressive_markdown(
        [_page(index, f"page {index + 1}") for index in range(10)],
        short_id="ab12cd3",
        tier="standard",
        after=None,
        limit=30000,
        add_markers=False,
    )
    next_request = _next_content_request(
        rendered=rendered,
        request_page_range="1~10",
        after=None,
        page_count=38,
        paginated=True,
    )

    assert rendered.truncated is False
    assert next_request is not None
    assert next_request.page_range == "11~20"


def test_paginated_next_request_is_never_after_only() -> None:
    next_request = _next_content_request(
        rendered=type(
            "Rendered",
            (),
            {
                "content_ranges": [
                    ContentRange(
                        page_range="7",
                        start="doc:ab12cd3/tier:standard/page:7/block:1/char:0",
                        end="doc:ab12cd3/tier:standard/page:7/block:1/char:40",
                    )
                ],
                "truncated": True,
                "last_page_no": 7,
                "next_page_no": 7,
                "cut_inside_page": True,
            },
        )(),
        request_page_range="7~10",
        after=None,
        page_count=38,
        paginated=True,
    )

    assert next_request is not None
    assert next_request.page_range == "7~10"
    assert next_request.after is not None


def test_page_markdown_blocks_prefers_markdown_table_in_doclib() -> None:
    html = """
    <table>
      <tr><th>Name</th><th>Score</th></tr>
      <tr><td>Alice</td><td>90</td></tr>
    </table>
    """.strip()
    page = PageInfo(
        page_idx=0,
        page_size=(100, 100),
        para_blocks=[
            Block(
                index=0,
                type=BlockType.TABLE,
                bbox=(0, 0, 10, 10),
                blocks=[
                    Block(
                        index=0,
                        type=BlockType.TABLE_BODY,
                        bbox=(0, 0, 10, 10),
                        lines=[
                            Line(
                                bbox=(0, 0, 10, 10),
                                spans=[Span(type=ContentType.TABLE, bbox=(0, 0, 10, 10), content=html)],
                            )
                        ],
                    )
                ],
            )
        ],
        _backend="pipeline",
    )

    assert _page_markdown_blocks(page) == [
        (
            0,
            "\n".join(
                [
                    "| Name | Score |",
                    "| --- | --- |",
                    "| Alice | 90 |",
                ]
            ),
        )
    ]
