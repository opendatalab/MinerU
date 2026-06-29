import base64
from dataclasses import fields
import json

import pytest

from mineru.parser.base import ParseResult
from mineru.schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import Block, ContentType, Line, PageInfo, Span
from mineru.utils.image_payload import ImagePayloadCache


def _data_uri(payload: bytes, image_type: str = "png") -> str:
    return f"data:image/{image_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _table_page_with_cached_inline_image(img_bytes: bytes) -> tuple[PageInfo, ImagePayloadCache, str]:
    """构造已完成图片外置化的表格页，验证 ParseResult 只承载顶层图片缓存。"""
    image_cache = ImagePayloadCache()
    inline_image = _data_uri(img_bytes)
    html = image_cache.replace_html_data_uri_sources(
        f'<table><tr><td><img src="{inline_image}"/></td></tr></table>'
    )
    page = PageInfo(
        page_idx=0,
        page_size=(100, 100),
        para_blocks=[
            Block(
                index=0,
                type="table",
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[Span(type=ContentType.TABLE, bbox=(0.0, 0.0, 10.0, 10.0), content=html)],
                    )
                ],
            )
        ],
        _backend="pipeline",
    )
    return page, image_cache, inline_image


def test_parse_result_does_not_expose_backend_version_or_file_name() -> None:
    field_names = {field.name for field in fields(ParseResult)}

    assert "_backend" not in field_names
    assert "_version_name" not in field_names
    assert "_file_name" not in field_names


def test_parse_result_from_dict_restores_pages() -> None:
    result = ParseResult(
        pages=[
            PageInfo(
                page_idx=3,
                page_size=(640, 480),
                para_blocks=[
                    Block(
                        index=0,
                        type="text",
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        lines=[
                            Line(
                                bbox=(1.0, 2.0, 3.0, 4.0),
                                spans=[Span(type="text", bbox=(1.0, 2.0, 3.0, 4.0), content="hello")],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    restored = ParseResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.pages[0].page_size == (640, 480)
    assert restored.pages[0].para_blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert restored.pages[0].para_blocks[0].lines[0].spans[0].content == "hello"


def test_parse_result_to_dict_includes_schema_version_without_meta() -> None:
    result = ParseResult(pages=[PageInfo(page_idx=0)])

    payload = result.to_dict()

    assert payload["schema_version"] == MIDDLE_JSON_SCHEMA_VERSION
    assert "pages" in payload
    assert "_meta" not in payload


def test_parse_result_from_dict_rejects_missing_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        ParseResult.from_dict({"pdf_info": []})


def test_parse_result_from_dict_accepts_pages_and_preserves_page_backend() -> None:
    restored = ParseResult.from_dict(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "page_size": [100, 200],
                    "_backend": "pipeline",
                    "preproc_blocks": [
                        {
                            "index": 0,
                            "type": "title",
                            "bbox": [1.0, 2.0, 3.0, 4.0],
                            "level": 1,
                            "lines": [
                                {
                                    "bbox": [1.0, 2.0, 3.0, 4.0],
                                    "spans": [
                                        {
                                            "type": "text",
                                            "bbox": [1.0, 2.0, 3.0, 4.0],
                                            "content": "Heading",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    assert len(restored.pages) == 1
    assert restored.pages[0]._backend == "pipeline"
    assert restored.pages[0].page_size == (100, 200)
    assert restored.pages[0].preproc_blocks[0].level == 1


def test_parse_result_from_dict_uses_root_backend_when_page_backend_missing() -> None:
    restored = ParseResult.from_dict(
        {
            "_backend": "office",
            "pages": [
                {
                    "page_idx": 0,
                }
            ],
        }
    )

    assert restored.pages[0]._backend == "office"


def test_parse_result_from_dict_keeps_page_backend_over_root_backend() -> None:
    restored = ParseResult.from_dict(
        {
            "_backend": "office",
            "pages": [
                {
                    "page_idx": 0,
                    "_backend": "pipeline",
                }
            ],
        }
    )

    assert restored.pages[0]._backend == "pipeline"


def test_parse_result_preserves_true_merge_prev_in_staged_middle_json() -> None:
    block = Block(index=0, type="text", bbox=(0.0, 0.0, 10.0, 10.0), merge_prev=True)
    result = ParseResult(pages=[PageInfo(page_idx=0, preproc_blocks=[block])])

    payload = result.to_dict()
    restored = ParseResult.from_dict(payload)

    serialized_block = payload["pages"][0]["preproc_blocks"][0]
    assert serialized_block["merge_prev"] is True
    assert "angle" not in serialized_block
    assert "score" not in serialized_block
    assert restored.pages[0].preproc_blocks[0].merge_prev is True


def test_parse_result_from_dict_restores_merge_prev_from_payload() -> None:
    restored = ParseResult.from_dict(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "preproc_blocks": [
                        {
                            "index": 0,
                            "type": "text",
                            "bbox": [0.0, 0.0, 10.0, 10.0],
                            "merge_prev": True,
                        }
                    ],
                }
            ]
        }
    )

    assert restored.pages[0].preproc_blocks[0].merge_prev is True


def test_parse_result_from_json_restores_pages() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "text",
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "lines": [
                            {
                                "bbox": [0.0, 0.0, 0.0, 0.0],
                                "spans": [{"type": "text", "bbox": [0.0, 0.0, 0.0, 0.0], "content": "round trip"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    restored = ParseResult.from_json(json.dumps(data))

    assert restored.to_dict() == ParseResult.from_dict(data).to_dict()


def test_parse_result_structured_content_method_and_save_name() -> None:
    page = PageInfo(
        page_idx=0,
        page_size=(100, 100),
        para_blocks=[
            Block(
                index=0,
                type="text",
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[
                            Span(
                                type=ContentType.TEXT,
                                bbox=(0.0, 0.0, 10.0, 10.0),
                                content="hello",
                            )
                        ],
                    )
                ],
            )
        ],
        _backend="pipeline",
    )
    result = ParseResult(pages=[page])
    writes: dict[str, str] = {}

    class MemoryWriter:
        def write_string(self, path: str, content: str) -> None:
            """记录 ParseResult.save 写出的文本产物，避免测试依赖真实文件系统。"""
            writes[path] = content

        def write(self, path: str, content: bytes) -> None:
            """记录图片产物写出接口，当前用例不会实际产生图片。"""
            writes[path] = content.decode("utf-8")

    structured_content = result.structured_content()
    result.save(MemoryWriter())

    assert structured_content[0][0]["content"]["paragraph_content"][0]["content"] == "hello"
    assert "structured_content.json" in writes
    assert "content_list" + "_v2.json" not in writes


def test_parse_result_render_methods_use_export_pages_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img_bytes = b"render-inline-image"
    inline_image = _data_uri(img_bytes)
    image_cache = ImagePayloadCache()
    html = image_cache.replace_html_data_uri_sources(f'<table><tr><td><img src="{inline_image}"/></td></tr></table>')
    page = PageInfo(
        page_idx=0,
        page_size=(100, 100),
        para_blocks=[
            Block(
                index=0,
                type="table",
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[Span(type=ContentType.TABLE, bbox=(0.0, 0.0, 10.0, 10.0), content=html)],
                    )
                ],
            )
        ],
        _backend="pipeline",
    )
    result = ParseResult(pages=[page], _image_cache=image_cache)
    captured: list[list[PageInfo]] = []

    def _assert_clean_export_pages(pages: list[PageInfo]) -> None:
        """确认 public 渲染入口直接收到生成阶段清理后的 page tree。"""
        captured.append(pages)
        content = pages[0].para_blocks[0].lines[0].spans[0].content
        assert inline_image not in content
        assert next(iter(result.images())) in content

    def fake_render_markdown(pages: list[PageInfo], *args, **kwargs) -> str:
        """记录 markdown 渲染输入，避免测试依赖具体 markdown 输出细节。"""
        _assert_clean_export_pages(pages)
        return "markdown"

    def fake_render_content_list(pages: list[PageInfo], *args, **kwargs) -> list[dict[str, str]]:
        """记录 content_list 渲染输入，验证导出视图在所有 public 渲染方法中一致。"""
        _assert_clean_export_pages(pages)
        return [{"type": "table"}]

    def fake_render_structured_content(pages: list[PageInfo], *args, **kwargs) -> list[list[dict[str, str]]]:
        """记录 structured_content 渲染输入，覆盖最新评论指出的 data URI 残留路径。"""
        _assert_clean_export_pages(pages)
        return [[{"type": "table"}]]

    monkeypatch.setattr("mineru.parser.base.render_markdown", fake_render_markdown)
    monkeypatch.setattr("mineru.parser.base.render_content_list", fake_render_content_list)
    monkeypatch.setattr("mineru.parser.base.render_structured_content", fake_render_structured_content)

    assert result.markdown() == "markdown"
    assert result.content_list() == [{"type": "table"}]
    assert result.structured_content() == [[{"type": "table"}]]
    assert len(captured) == 3
    assert all(pages is result.pages for pages in captured)
    assert inline_image not in page.para_blocks[0].lines[0].spans[0].content


def test_parse_result_save_writes_cached_images_and_exports_clean_middle_json() -> None:
    img_bytes = b"image-bytes"
    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(img_bytes, "jpeg", image_path="figure.png")
    page = PageInfo(
        page_idx=0,
        page_size=(100, 100),
        para_blocks=[
            Block(
                index=0,
                type="image",
                bbox=(0.0, 0.0, 10.0, 10.0),
                lines=[
                    Line(
                        bbox=(0.0, 0.0, 10.0, 10.0),
                        spans=[
                            Span(
                                type=ContentType.IMAGE,
                                bbox=(0.0, 0.0, 10.0, 10.0),
                                image_path=image_path,
                            )
                        ],
                    )
                ],
            )
        ],
        _backend="pipeline",
    )
    result = ParseResult(pages=[page], _image_cache=image_cache)
    writes: dict[str, bytes | str] = {}

    class MemoryWriter:
        def write_string(self, path: str, content: str) -> None:
            """记录导出文本，确认 public middle_json 不再携带 base64。"""
            writes[path] = content

        def write(self, path: str, content: bytes) -> None:
            """记录导出图片，确认图片统一由 ParseResult 落盘。"""
            writes[path] = content

    result.save(MemoryWriter())
    exported = json.loads(writes["middle_json.json"])  # type: ignore[arg-type]
    exported_span = exported["pages"][0]["para_blocks"][0]["lines"][0]["spans"][0]

    assert exported_span["image_path"] == "figure.png"
    assert "image_base64" not in exported_span
    assert writes["figure.png"] == img_bytes


def test_parse_result_save_renders_structured_content_from_clean_export_pages() -> None:
    img_bytes = b"table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)
    result = ParseResult(pages=[page], _image_cache=image_cache)
    writes: dict[str, bytes | str] = {}

    class MemoryWriter:
        def write_string(self, path: str, content: str) -> None:
            """记录 save 写出的文本产物，验证 public 文件不再携带内联 base64 图片。"""
            writes[path] = content

        def write(self, path: str, content: bytes) -> None:
            """记录 save 写出的图片产物，验证导出视图引用的图片确实落盘。"""
            writes[path] = content

    result.save(MemoryWriter())
    image_path = next(iter(result.images()))
    structured_content = writes["structured_content.json"]

    assert isinstance(structured_content, str)
    assert inline_image not in structured_content
    assert image_path in structured_content
    assert writes[image_path] == img_bytes
    assert inline_image not in page.para_blocks[0].lines[0].spans[0].content


def test_parse_result_public_outputs_use_top_level_image_cache() -> None:
    img_bytes = b"cached-table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)
    result = ParseResult(pages=[page], _image_cache=image_cache)

    output_text = json.dumps(
        {
            "markdown": result.markdown(),
            "content_list": result.content_list(),
            "structured_content": result.structured_content(),
            "middle_json": result.to_dict(),
            "images": sorted(result.images()),
        },
        ensure_ascii=False,
    )

    assert inline_image not in output_text
    assert next(iter(result.images())) in output_text
    assert inline_image not in page.para_blocks[0].lines[0].spans[0].content


def test_parse_result_export_pages_returns_defensive_copy_from_cache() -> None:
    img_bytes = b"defensive-table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)
    result = ParseResult(pages=[page], _image_cache=image_cache)
    first_export = result.export_pages()
    first_export[0].para_blocks[0].lines[0].spans[0].content = "mutated by caller"

    second_export = result.export_pages()
    second_content = second_export[0].para_blocks[0].lines[0].spans[0].content
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert "mutated by caller" not in second_content
    assert "mutated by caller" not in exported_json
    assert inline_image not in second_content
    assert inline_image not in exported_json
    assert next(iter(result.images())) in exported_json


def test_parse_result_export_rewrites_inline_table_base64_images() -> None:
    img_bytes = b"table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)

    result = ParseResult(pages=[page], _image_cache=image_cache)
    images = result.images()
    exported_page = result.export_pages()[0]
    exported_span = exported_page.para_blocks[0].lines[0].spans[0]
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert list(images.values()) == [img_bytes]
    assert inline_image not in exported_json
    assert 'src="' in exported_span.content
    assert next(iter(images)) in exported_json
