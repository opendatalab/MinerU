import json

import pytest

from mineru.parser.base import ParseResult
from mineru.types import Block, ContentType, Line, PageInfo, Span
from mineru.utils import title_level_postprocess


def test_finalize_client_side_pages_passes_hybrid_effort(monkeypatch) -> None:
    """校验客户端 Hybrid finalize 会沿用请求侧传入的 effort。"""
    calls: list[str] = []
    pages = [PageInfo(page_idx=0, _backend="hybrid")]
    monkeypatch.setattr(
        "mineru.backend.hybrid.model_output_to_middle_json.finalize_middle_json_from_preproc",
        lambda parsed_pages, effort="medium": calls.append(effort),
    )

    title_level_postprocess.finalize_client_side_pages(pages, "hybrid", effort="high")

    assert calls == ["high"]


def test_finalize_client_side_pages_rejects_low_effort(monkeypatch) -> None:
    """校验客户端 Hybrid finalize 不再接受 low effort。"""
    calls: list[str] = []
    pages = [PageInfo(page_idx=0, _backend="hybrid")]
    monkeypatch.setattr(
        "mineru.backend.hybrid.model_output_to_middle_json.finalize_middle_json_from_preproc",
        lambda parsed_pages, effort="medium": calls.append(effort),
    )

    with pytest.raises(ValueError, match="Unsupported effort 'low'"):
        title_level_postprocess.finalize_client_side_pages(pages, "hybrid", effort="low")

    assert calls == []


def test_regenerate_client_side_outputs_forwards_hybrid_effort(monkeypatch, tmp_path) -> None:
    """校验客户端重建输出时继续把 Hybrid effort 传到 finalize 阶段。"""
    from mineru.cli_old import client_side_output

    middle_json_path = tmp_path / "demo_middle.json"
    middle_json_path.write_text(
        json.dumps(ParseResult(pages=[PageInfo(page_idx=0)]).to_dict()),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client_side_output,
        "finalize_client_side_pages",
        lambda pages, backend, effort="medium": calls.append((backend, effort)),
    )

    client_side_output.regenerate_client_side_outputs(tmp_path, "demo", "hybrid-engine", effort="high")

    assert calls == [("hybrid", "high")]


def test_finalize_client_side_pages_uses_explicit_backend(monkeypatch) -> None:
    legacy_backend = "pipeline"
    pages = [PageInfo(page_idx=0, _backend=legacy_backend)]

    with pytest.raises(ValueError, match="Unsupported client-side finalize backend 'pipeline'"):
        title_level_postprocess.finalize_client_side_pages(pages, legacy_backend)


def test_finalize_client_side_hybrid_pages_uses_round_tripped_merge_prev_then_cleans_it() -> None:
    """校验客户端 Hybrid finalize 会消费并清理 round-trip 后的 merge_prev 元数据。"""
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

    title_level_postprocess.finalize_client_side_pages(round_tripped_pages, "hybrid", effort="high")

    para_blocks = round_tripped_pages[0].para_blocks
    merged_text = [span.content for line in para_blocks[0].lines for span in line.spans]
    finalized_payload = ParseResult(pages=round_tripped_pages).to_dict()
    assert merged_text == ["hello", "world"]
    assert para_blocks[1].lines == []
    assert "merge_prev" not in finalized_payload["pages"][0]["preproc_blocks"][1]
    assert "merge_prev" not in finalized_payload["pages"][0]["para_blocks"][1]
