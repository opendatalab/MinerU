from mineru.parser.base import ParseResult
from mineru.types import Block, ContentType, Line, PageInfo, Span
from mineru.utils import title_level_postprocess


def test_finalize_client_side_pages_uses_explicit_backend(monkeypatch) -> None:
    calls: list[list[PageInfo]] = []
    pages = [PageInfo(page_idx=0, _backend="pipeline")]
    monkeypatch.setattr(
        "mineru.backend.pipeline.model_output_to_middle_json.finalize_middle_json_from_preproc",
        lambda parsed_pages: calls.append(parsed_pages),
    )

    title_level_postprocess.finalize_client_side_pages(pages, "pipeline")

    assert calls == [pages]


def test_finalize_client_side_pages_uses_round_tripped_merge_prev_then_cleans_it() -> None:
    previous_block = Block(
        index=0,
        type="text",
        bbox=(0.0, 0.0, 100.0, 20.0),
        lines=[
            Line(
                bbox=(0.0, 0.0, 100.0, 10.0),
                spans=[Span(type=ContentType.TEXT, bbox=(0.0, 0.0, 100.0, 10.0), content="hello")],
            )
        ],
    )
    current_block = Block(
        index=1,
        type="text",
        bbox=(0.0, 15.0, 100.0, 30.0),
        merge_prev=True,
        lines=[
            Line(
                bbox=(0.0, 15.0, 100.0, 25.0),
                spans=[Span(type=ContentType.TEXT, bbox=(0.0, 15.0, 100.0, 25.0), content="world")],
            )
        ],
    )
    staged_payload = ParseResult(
        pages=[PageInfo(page_idx=0, preproc_blocks=[previous_block, current_block])]
    ).to_dict()
    round_tripped_pages = ParseResult.from_dict(staged_payload).pages

    title_level_postprocess.finalize_client_side_pages(round_tripped_pages, "vlm")

    para_blocks = round_tripped_pages[0].para_blocks
    merged_text = [span.content for line in para_blocks[0].lines for span in line.spans]
    finalized_payload = ParseResult(pages=round_tripped_pages).to_dict()
    assert merged_text == ["hello", "world"]
    assert para_blocks[1].lines == []
    assert "merge_prev" not in finalized_payload["pages"][0]["preproc_blocks"][1]
    assert "merge_prev" not in finalized_payload["pages"][0]["para_blocks"][1]
