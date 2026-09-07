from __future__ import annotations

from docvortex.schema import Producer

from mineru.integrations.docvortex import build_metadata
from mineru.render import (
    render_content_list,
    render_content_list_v2,
    render_structured_content,
)
from mineru.types import IndexBlock, MiddleJson, PageInfo, TextBlock


def _inline(text: str) -> list[dict[str, str]]:
    """构造最小结构化文本 span。"""
    return [{"type": "text", "content": text}]


def _middle_with_text_anchor() -> MiddleJson:
    """构造目录前向引用顶层 TextBlock 的严格文档。"""
    return MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    IndexBlock(
                        type="index",
                        index=0,
                        content=[TextBlock(type="text", anchor="body target", content=_inline("Body target\t3"))],
                    )
                ],
            ),
            PageInfo(
                page_idx=1,
                blocks=[TextBlock(type="text", index=0, anchor="body target", content=_inline("Body paragraph"))],
            ),
        ],
        is_full_document=True,
        file_suffix="docx",
        producer=Producer(name="mineru", version="test"),
        extensions=build_metadata(effort="flash", parse_mode="txt", mineru_version="test"),
    )


def test_text_anchor_structured_and_content_list_metadata() -> None:
    """验证结构化输出只保留 anchor 元数据，不向正文内容注入目标标签。"""
    middle = _middle_with_text_anchor()

    structured = render_structured_content(middle)
    structured_target = structured["pages"][1]["blocks"][0]
    assert structured_target["anchor"] == "body target"
    assert structured_target["content"] == "Body paragraph"
    assert "<a " not in structured_target["content"]

    content_list = render_content_list(middle)
    target_v1 = next(item for item in content_list if item.get("text") == "Body paragraph")
    assert target_v1["anchor"] == "body target"
    assert content_list[0]["list_items"] == ["- [Body target](#body%20target)"]

    content_list_v2 = render_content_list_v2(middle)
    assert content_list_v2[1][0]["anchor"] == "body target"
    assert content_list_v2[0][0]["content"]["list_items"][0]["anchor"] == "body target"
