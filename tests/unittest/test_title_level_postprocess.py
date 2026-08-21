import json

import pytest

from mineru.parser.base import ParseResult
from mineru.types import PageInfo
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
