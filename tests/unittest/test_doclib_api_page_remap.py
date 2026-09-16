"""验证 API 分页结果的正文、布局几何与落盘缓存使用相同源页号。"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
from docvortex.document.pdf.layout import LAYOUT_EXTENSION
from docvortex.schema import MiddleJson, PageInfo, TextBlock, TextSpan
from pypdf import PdfReader

from mineru.doclib.services.parse_svc import ParseFailure, _remap_api_result_pages_to_page_range, parse_batch_json_path
from mineru.integrations.docvortex import build_metadata
from mineru.parser.base import ParseResult
from mineru.render import PdfLayout, render_pdf


def _api_result() -> ParseResult:
    """构造含不同尺寸和图片旋转的三页 API 局部结果，不依赖推理模型。"""
    return ParseResult(
        MiddleJson(
            pages=[
                PageInfo(
                    page_idx=idx,
                    blocks=[
                        TextBlock(
                            type="text",
                            index=0,
                            bbox=(0.1, 0.1, 0.9, 0.3),
                            content=[TextSpan(type="text", content=f"Source page {idx}")],
                        )
                    ],
                )
                for idx in range(3)
            ],
            is_full_document=False,
            metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
            extensions={
                **build_metadata(effort="high", parse_mode="txt"),
                "application": {"keep": ["unchanged"]},
                LAYOUT_EXTENSION: {
                    "version": 1,
                    "pages": [
                        {"page_idx": idx, "width_pt": width, "height_pt": height, "image_rotations": {"1": rotation}}
                        for idx, (width, height, rotation) in enumerate([(400, 600, 90), (480, 320, 270), (600, 400, 0)])
                    ],
                },
            },
        )
    )


@pytest.mark.parametrize("page_range,expected", [("2-4", [1, 2, 3]), ("11,13-14", [10, 12, 13])])
def test_api_page_remap_survives_cache_and_original_render(tmp_path: Path, page_range: str, expected: list[int]) -> None:
    """连续及非连续请求落盘重读后，正文、尺寸与旋转仍按源页号对应。"""
    result = _api_result()
    original = deepcopy(result.middle_json.extensions)
    original_geometry = result.middle_json.extensions[LAYOUT_EXTENSION]
    _remap_api_result_pages_to_page_range(result, page_range)

    geometry = result.middle_json.extensions[LAYOUT_EXTENSION]
    assert [page.page_idx for page in result.pages] == expected
    assert geometry == {
        "version": 1,
        "pages": [{**page, "page_idx": idx} for page, idx in zip(original_geometry["pages"], expected, strict=True)],
    }
    assert original_geometry == original[LAYOUT_EXTENSION]
    assert {key: value for key, value in result.middle_json.extensions.items() if key != LAYOUT_EXTENSION} == {
        key: value for key, value in original.items() if key != LAYOUT_EXTENSION
    }

    path = Path(parse_batch_json_path(str(tmp_path), "a" * 64, "standard", page_range, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
    replayed = ParseResult.from_json(path.read_text(encoding="utf-8"))
    assert replayed.to_dict() == result.to_dict()
    before_render = deepcopy(replayed.to_dict())
    pdf = PdfReader(BytesIO(render_pdf(replayed.middle_json, layout=PdfLayout.ORIGINAL)))
    assert [tuple(page.mediabox)[2:] for page in pdf.pages] == [(400, 600), (480, 320), (600, 400)]
    assert [f"Source page {idx}" in page.extract_text() for idx, page in enumerate(pdf.pages)] == [True, True, True]
    assert replayed.to_dict() == before_render


def test_api_page_remap_is_idempotent() -> None:
    """已使用源页号的结果重复进入兼容边界时，不再映射几何。"""
    result = _api_result()
    _remap_api_result_pages_to_page_range(result, "11,13-14")
    before = deepcopy(result.to_dict())
    _remap_api_result_pages_to_page_range(result, "11,13-14")
    assert result.to_dict() == before


def test_api_page_remap_without_geometry() -> None:
    """旧服务未提供布局扩展时，继续恢复正文页号且不伪造几何。"""
    result = _api_result()
    del result.middle_json.extensions[LAYOUT_EXTENSION]
    before = deepcopy(result.middle_json.extensions)
    _remap_api_result_pages_to_page_range(result, "11,13-14")
    assert [page.page_idx for page in result.pages] == [10, 12, 13]
    assert result.middle_json.extensions == before


@pytest.mark.parametrize("failure", ["count", "out_of_range", "missing_pages", "null_layout", "invalid_index"])
def test_api_page_remap_failure_preserves_original_result(failure: str) -> None:
    """页数或几何映射无效时报告统一错误，不留下部分更新的正文或扩展。"""
    result = _api_result()
    geometry = result.middle_json.extensions[LAYOUT_EXTENSION]
    if failure == "out_of_range":
        geometry["pages"][-1]["page_idx"] = 3
    elif failure == "missing_pages":
        del geometry["pages"]
    elif failure == "null_layout":
        result.middle_json.extensions[LAYOUT_EXTENSION] = None
    elif failure == "invalid_index":
        geometry["pages"][-1]["page_idx"] = "2"
    before = deepcopy(result.to_dict())
    with pytest.raises(ParseFailure) as exc:
        _remap_api_result_pages_to_page_range(result, "11-12" if failure == "count" else "11,13-14")
    assert exc.value.code == "parse_page_remap_failed"
    assert result.to_dict() == before


def test_api_page_remap_preserves_missing_geometry() -> None:
    """个别页几何缺失时只映射现有记录，不按列表位置错配或补造尺寸。"""
    result = _api_result()
    del result.middle_json.extensions[LAYOUT_EXTENSION]["pages"][1]
    _remap_api_result_pages_to_page_range(result, "11,13-14")
    assert [page.page_idx for page in result.pages] == [10, 12, 13]
    assert [page["page_idx"] for page in result.middle_json.extensions[LAYOUT_EXTENSION]["pages"]] == [10, 13]


def test_empty_api_page_remap_is_unchanged() -> None:
    """空解析结果由既有上层处理，页号兼容函数不修改附带扩展。"""
    result = _api_result()
    result.pages.clear()
    before = deepcopy(result.to_dict())
    _remap_api_result_pages_to_page_range(result, "11,13-14")
    assert result.to_dict() == before
