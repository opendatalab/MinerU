from mineru.types import PageInfo
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
