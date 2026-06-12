from types import SimpleNamespace

from mineru.types import PageInfo
from mineru.utils import title_level_postprocess


def test_finalize_client_side_middle_json_uses_page_backend(monkeypatch) -> None:
    calls: list[list[PageInfo]] = []
    pages = [PageInfo(page_idx=0, _backend="pipeline")]
    fake_result = SimpleNamespace(pages=pages)

    monkeypatch.setattr(
        "mineru.parser.base.ParseResult.from_dict",
        lambda middle_json: fake_result,
    )
    monkeypatch.setattr(
        "mineru.backend.pipeline.model_output_to_middle_json.finalize_middle_json_from_preproc",
        lambda parsed_pages: calls.append(parsed_pages),
    )

    middle_json = {"pages": []}
    result = title_level_postprocess.finalize_client_side_middle_json(middle_json)

    assert result is middle_json
    assert calls == [pages]
