from __future__ import annotations
from _span_test_utils import inline as _inline

from copy import deepcopy
from typing import get_args

import pytest
from pydantic import ValidationError

import mineru.types as types_module
from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.backend.postprocess.inline import inline_plain_text, normalize_inline_spans
from mineru.types import (
    BLOCK_ADAPTER,
    BLOCK_TYPES,
    PAGE_BLOCK_TYPES,
    RAW_FORMULA_NUMBER,
    RAW_ONLY_BLOCK_TYPES,
    BlockType,
    BlockTypes,
    ChartAnnotationBlock,
    CodeAnnotationBlock,
    CodeBlock,
    ContinuableTextBlockBase,
    EquationBlock,
    ImageAnnotationBlock,
    ListBlock,
    MiddleJson,
    ModelJson,
    PageAuxTextBlock,
    PageBlockTypes,
    PageFootnoteBlock,
    PageInfo,
    RefTextBlock,
    TableAnnotationBlock,
    TextBlock,
    TextSpan,
    parse_block,
    parse_inline_spans,
)


def _public_block_payloads() -> dict[str, dict[str, object]]:
    """构造 29 个公开 discriminator 的最小合法载荷。"""
    text_leaf = {"type": "text", "content": _inline("item")}
    payloads: dict[str, dict[str, object]] = {
        "aside_text": {"type": "aside_text", "content": _inline("aside")},
        "doc_title": {"type": "doc_title", "content": _inline("title"), "level": 1},
        "footer": {"type": "footer", "content": _inline("footer")},
        "header": {"type": "header", "content": _inline("header")},
        "index": {"type": "index", "content": [text_leaf]},
        "equation": {"type": "equation", "content": "x=1"},
        "list": {"type": "list", "content": [text_leaf]},
        "page_footnote": {"type": "page_footnote", "content": _inline("note")},
        "page_number": {"type": "page_number", "content": _inline("1")},
        "paragraph_title": {"type": "paragraph_title", "content": _inline("section"), "level": 2},
        "ref_text": {"type": "ref_text", "content": _inline("reference")},
        "text": {"type": "text", "content": _inline("text")},
        "image_body": {"type": "image_body", "content": ""},
        "image_caption": {"type": "image_caption", "content": _inline("caption")},
        "image_footnote": {"type": "image_footnote", "content": _inline("note")},
        "table_body": {"type": "table_body", "content": "<table></table>"},
        "table_caption": {"type": "table_caption", "content": _inline("caption")},
        "table_footnote": {"type": "table_footnote", "content": _inline("note")},
        "chart_body": {"type": "chart_body", "content": ""},
        "chart_caption": {"type": "chart_caption", "content": _inline("caption")},
        "chart_footnote": {"type": "chart_footnote", "content": _inline("note")},
        "code_body": {"type": "code_body", "content": "print(1)"},
        "algorithm_body": {"type": "algorithm_body", "content": _inline("algorithm")},
        "code_caption": {"type": "code_caption", "content": _inline("caption")},
        "code_footnote": {"type": "code_footnote", "content": _inline("note")},
    }
    payloads.update(
        {
            "image": {"type": "image", "content": [payloads["image_body"]]},
            "table": {"type": "table", "content": [payloads["table_body"]]},
            "chart": {"type": "chart", "content": [payloads["chart_body"]]},
            "code": {
                "type": "code",
                "content": [payloads["code_body"]],
                "sub_type": "code",
                "guess_lang": "python",
            },
        }
    )
    return payloads


def test_all_29_public_discriminators_parse_to_concrete_models() -> None:
    """验证公开类型集合与 discriminated union 完整一致。"""
    payloads = _public_block_payloads()

    assert set(payloads) == BLOCK_TYPES
    assert len(payloads) == 29
    assert "equation" not in RAW_ONLY_BLOCK_TYPES
    assert all(parse_block(payload).type == block_type for block_type, payload in payloads.items())
    assert isinstance(parse_block(payloads["equation"]), EquationBlock)


def test_public_block_type_declarations_match_block_union() -> None:
    """验证 BlockType、BlockTypes 和 BLOCK_TYPES 只公开 Block 联合的 discriminator。"""
    class_values = {value for name, value in vars(BlockType).items() if name.isupper()}

    assert class_values == BLOCK_TYPES
    assert set(get_args(BlockTypes)) == BLOCK_TYPES
    assert set(get_args(PageBlockTypes)) == PAGE_BLOCK_TYPES


def test_shared_models_preserve_all_discriminator_values() -> None:
    """验证同结构 discriminator 共用模型，但仍保留原始 type 语义。"""
    assert isinstance(parse_block({"type": "header", "content": _inline("h")}), PageAuxTextBlock)
    assert isinstance(parse_block({"type": "image_caption", "content": _inline("c")}), ImageAnnotationBlock)
    assert isinstance(parse_block({"type": "table_footnote", "content": _inline("f")}), TableAnnotationBlock)
    assert isinstance(parse_block({"type": "chart_caption", "content": _inline("c")}), ChartAnnotationBlock)
    assert isinstance(parse_block({"type": "code_footnote", "content": _inline("f")}), CodeAnnotationBlock)


def test_inline_span_models_normalize_styles_and_merge_adjacent_text() -> None:
    """验证 TextSpan 样式顺序去重、空白保真和相邻同样式合并。"""
    spans = normalize_inline_spans(
        [
            {"type": "text", "content": "A", "styles": ["underline", "bold", "bold"]},
            {"type": "text", "content": " \n", "styles": ["bold", "underline"]},
            {"type": "text", "content": "B"},
        ]
    )

    assert spans == [
        TextSpan(type="text", content="A \n", styles=["bold", "underline"]),
        TextSpan(type="text", content="B"),
    ]
    assert inline_plain_text(spans) == "A \nB"
    with pytest.raises(ValidationError, match="string_too_short"):
        parse_inline_spans([{"type": "text", "content": ""}])
    with pytest.raises(ValidationError, match="both superscript and subscript"):
        parse_inline_spans([{"type": "text", "content": "x", "styles": ["superscript", "subscript"]}])


def test_inline_equation_code_and_hyperlink_are_strict_and_safe() -> None:
    """验证非文字 Span 的空值、嵌套链接、危险 URL 和额外字段均被拒绝。"""
    valid = parse_inline_spans(
        [
            {"type": "equation_inline", "content": "x<y"},
            {"type": "code_inline", "content": "print(x)"},
            {"type": "hyperlink", "url": "#target", "content": _inline("link")},
        ]
    )
    assert inline_plain_text(valid) == "x<yprint(x)link"

    invalid_payloads = [
        [{"type": "equation_inline", "content": "   "}],
        [{"type": "code_inline", "content": ""}],
        [{"type": "hyperlink", "url": "", "content": _inline("link")}],
        [{"type": "hyperlink", "url": "javascript:alert(1)", "content": _inline("link")}],
        [{"type": "hyperlink", "url": "https://example.com", "content": []}],
        [
            {
                "type": "hyperlink",
                "url": "https://example.com",
                "content": [{"type": "hyperlink", "url": "#nested", "content": _inline("nested")}],
            }
        ],
        [{"type": "text", "content": "x", "unknown": True}],
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            parse_inline_spans(payload)


def test_inline_text_preserves_markup_entities_and_unicode_verbatim() -> None:
    """验证符号、实体外观、引号、Unicode 和完整标签字面量不被解释。"""
    content = 'A&B / 1<2 / 3>2 / &amp; / "quote" / 中文🙂 / <eq>x</eq> / <script>alert(1)</script>'
    spans = parse_inline_spans(_inline(content))

    assert inline_plain_text(spans) == content
    assert len(spans) == 1 and isinstance(spans[0], TextSpan)


def test_model_json_rejects_legacy_inline_string_with_page_and_block_location() -> None:
    """验证 ModelJson 对旧字符串 content 给出页号和块号定位。"""
    with pytest.raises(ValidationError, match=r"pages\[0\]\[0\].*type=text"):
        ModelJson(
            pages=[[{"type": BlockType.TEXT, "content": "legacy string"}]],
            page_index_map=[],
            file_suffix="docx",
            effort="flash",
            parse_mode="txt",
            mineru_version="test",
        )


def test_page_footnote_uses_independent_model_and_exclusive_anchor() -> None:
    """验证页面脚注使用独立模型，页面辅助块不再接受 anchor。"""
    footnote = parse_block(
        {
            "type": "page_footnote",
            "content": _inline("note"),
            "anchor": "note-one",
        }
    )

    assert isinstance(footnote, PageFootnoteBlock)
    assert footnote.anchor == "note-one"
    assert footnote.to_dict()["anchor"] == "note-one"
    for block_type in ("header", "footer", "page_number", "aside_text"):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            parse_block({"type": block_type, "content": _inline("aux"), "anchor": "invalid"})


def test_title_levels_follow_global_hierarchy() -> None:
    """验证文档标题固定为一级，段落标题严格限制在二至六级。"""
    doc_title = parse_block({"type": "doc_title", "content": _inline("doc"), "level": 1})
    paragraph_title = parse_block({"type": "paragraph_title", "content": _inline("section"), "level": 2})
    deepest_title = parse_block({"type": "paragraph_title", "content": _inline("deep"), "level": 6})

    assert doc_title.to_dict()["level"] == 1
    assert paragraph_title.to_dict()["level"] == 2
    assert deepest_title.to_dict()["level"] == 6
    for payload in (
        {"type": "doc_title", "content": _inline("doc")},
        {"type": "doc_title", "content": _inline("doc"), "level": 2},
        {"type": "paragraph_title", "content": _inline("section")},
        {"type": "paragraph_title", "content": _inline("section"), "level": 1},
        {"type": "paragraph_title", "content": _inline("section"), "level": 7},
    ):
        with pytest.raises(ValidationError):
            parse_block(payload)


@pytest.mark.parametrize("block_type", ["equation", "image_body", "table_body", "chart_body"])
def test_image_payload_content_must_be_string(block_type: str) -> None:
    """验证所有图片载荷块在严格 Middle JSON 中都拒绝 null content。"""
    with pytest.raises(ValidationError):
        parse_block({"type": block_type, "content": None})


def test_cell_merge_belongs_only_to_table_root() -> None:
    """验证 cell_merge 只允许位于 table 根块。"""
    table_payload = deepcopy(_public_block_payloads()["table"])
    table_payload["cell_merge"] = [1, 0]

    table = parse_block(table_payload)

    assert table.cell_merge == [1, 0]
    with pytest.raises(ValidationError):
        parse_block({"type": "table_body", "content": "<table></table>", "cell_merge": [1, 0]})


@pytest.mark.parametrize("raw_type", sorted(RAW_ONLY_BLOCK_TYPES))
def test_raw_only_types_are_rejected(raw_type: str) -> None:
    """验证 Analyze 私有 raw type 不能越过公开对象边界。"""
    assert raw_type in RAW_ONLY_BLOCK_TYPES
    with pytest.raises(ValidationError):
        parse_block({"type": raw_type, "content": "raw"})


def test_legacy_interline_equation_discriminator_is_rejected() -> None:
    """验证旧 interline_equation 不提供兼容入口，严格对象只接受 equation。"""
    with pytest.raises(ValidationError):
        parse_block({"type": "interline_equation", "content": "x=1"})


def test_equation_schema_uses_only_canonical_discriminator() -> None:
    """验证生成的 JSON Schema 只公开 equation 与 EquationBlock。"""
    schema = BLOCK_ADAPTER.json_schema()
    mapping = schema["discriminator"]["mapping"]
    legacy_model_name = "Interline" + "EquationBlock"

    assert mapping["equation"] == "#/$defs/EquationBlock"
    assert "interline_equation" not in mapping
    assert "EquationBlock" in schema["$defs"]
    assert legacy_model_name not in schema["$defs"]


def test_formula_number_is_rejected_by_middle_json_boundary_and_schema() -> None:
    """验证 formula_number 仅属于 Analyze raw 阶段，不能进入公开 Block 或 Middle JSON。"""
    schema = BLOCK_ADAPTER.json_schema()
    mapping = schema["discriminator"]["mapping"]

    assert RAW_FORMULA_NUMBER in RAW_ONLY_BLOCK_TYPES
    assert RAW_FORMULA_NUMBER not in mapping
    assert "FormulaNumberBlock" not in schema["$defs"]
    with pytest.raises(ValidationError):
        parse_block({"type": RAW_FORMULA_NUMBER, "content": "(1)"})
    with pytest.raises(ValidationError):
        model_json_to_pages(
            ModelJson(
                pages=[[{"type": RAW_FORMULA_NUMBER, "bbox": [0.7, 0.3, 0.8, 0.4], "content": "(1)"}]],
                page_index_map=[],
                file_suffix="pdf",
                effort="flash",
                parse_mode="txt",
                mineru_version="test",
            )
        )


def test_continuable_text_models_share_marker_and_reject_unknown_fields() -> None:
    """验证 Text/RefText 共用续接基类，同时继续执行严格字段校验。"""
    text = parse_block({"type": "text", "content": _inline("x"), "continues_prev": True})
    ref_text = parse_block({"type": "ref_text", "content": _inline("r"), "continues_prev": True})

    assert isinstance(text, ContinuableTextBlockBase)
    assert isinstance(ref_text, RefTextBlock)
    assert isinstance(ref_text, ContinuableTextBlockBase)
    assert set(TextBlock.model_fields) == set(RefTextBlock.model_fields)
    assert text.continues_prev is True
    assert ref_text.continues_prev is True
    with pytest.raises(ValidationError):
        parse_block({"type": "text", "content": _inline("x"), "anchor": "a"})
    with pytest.raises(ValidationError):
        parse_block({"type": "text", "content": _inline("x"), "unknown": 1})


def test_every_public_block_rejects_removed_merge_field() -> None:
    """验证已废弃合并字段不会被任一公开 block 静默接收。"""
    for payload in _public_block_payloads().values():
        invalid_payload = deepcopy(payload)
        invalid_payload["merge_prev"] = False
        with pytest.raises(ValidationError):
            parse_block(invalid_payload)


def test_removed_block_classes_are_not_exposed() -> None:
    """验证未采用的旧 block 类没有重新进入公开对象体系。"""
    legacy_equation_model_name = "Interline" + "EquationBlock"
    removed_names = (
        "TitleBlock",
        "AsideTextBlock",
        "HeaderBlock",
        "FooterBlock",
        "PageNumberBlock",
        "ImageCaptionBlock",
        "ImageFootnoteBlock",
        "TableCaptionBlock",
        "TableFootnoteBlock",
        "ChartCaptionBlock",
        "ChartFootnoteBlock",
        "CodeCaptionBlock",
        "CodeFootnoteBlock",
        "HeaderImageBlock",
        "FooterImageBlock",
        legacy_equation_model_name,
        "_DocElement",
        "Span",
        "Line",
        "ContentItem",
        "EMPTY_BBOX",
        "_is_default_value",
        "_origin_is_list",
        "_list_arg",
    )
    for removed_name in removed_names:
        assert not hasattr(types_module, removed_name)
    assert not hasattr(types_module.BlockType, "INTERLINE_" + "EQUATION")


@pytest.mark.parametrize(
    "bbox",
    [(-0.1, 0.1, 0.8, 0.8), (0.1, 0.1, 1.1, 0.8), (0.5, 0.1, 0.5, 0.8), (0.1, 0.8, 0.5, 0.2)],
)
def test_bbox_must_be_normalized_and_positive(bbox: tuple[float, ...]) -> None:
    """验证 bbox 必须有限、归一化且具有正面积。"""
    with pytest.raises(ValidationError):
        parse_block({"type": "text", "content": _inline("x"), "bbox": bbox})


def test_recursive_list_and_index_round_trip() -> None:
    """验证四层 List 与递归 Index 的顺序和具体类型可无损恢复。"""
    list_payload: dict[str, object] = {"type": "text", "content": _inline("leaf")}
    for _ in range(4):
        list_payload = {"type": "list", "content": [{"type": "text", "content": _inline("item")}, list_payload]}
    index_payload = {
        "type": "index",
        "content": [
            {"type": "text", "content": _inline("one")},
            {"type": "index", "content": [{"type": "text", "content": _inline("two")}]},
        ],
    }

    list_block = parse_block(list_payload)
    index_block = parse_block(index_payload)

    assert isinstance(list_block, ListBlock)
    assert isinstance(list_block.content[1], ListBlock)
    assert [child.content[0].content for child in index_block.content if isinstance(child, TextBlock)] == ["one"]
    assert parse_block(list_block.to_dict(skip_defaults=False)) == list_block
    assert parse_block(index_block.to_dict(skip_defaults=False)) == index_block


def test_list_subtype_allows_mixed_direct_text_children() -> None:
    """验证 List subtype 是子项类型的统计结果，不约束每个直接子项。"""
    block = parse_block(
        {
            "type": "list",
            "sub_type": "ref_text",
            "content": [
                {"type": "text", "content": _inline("x")},
                {"type": "ref_text", "content": _inline("r1")},
                {"type": "ref_text", "content": _inline("r2")},
                {"type": "list", "sub_type": "text", "content": [{"type": "ref_text", "content": _inline("nested")}]},
            ],
        }
    )
    assert isinstance(block, ListBlock)
    assert block.sub_type == BlockType.REF_TEXT
    assert [child.type for child in block.content] == [
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.REF_TEXT,
        BlockType.LIST,
    ]
    assert isinstance(block.content[3], ListBlock)
    assert block.content[3].content[0].type == BlockType.REF_TEXT


@pytest.mark.parametrize("visual_type", ["image", "table", "chart", "code"])
def test_visual_parent_requires_exactly_one_body(visual_type: str) -> None:
    """验证视觉父块必须且只能包含一个对应 body。"""
    payload = deepcopy(_public_block_payloads()[visual_type])
    payload["content"] = []
    with pytest.raises(ValidationError, match="exactly one"):
        parse_block(payload)


def test_visual_parent_body_location_contract() -> None:
    """验证视觉 parent/body 的 index 和同时存在的 bbox 必须一致。"""
    with pytest.raises(ValidationError, match="index"):
        parse_block(
            {
                "type": "image",
                "index": 1,
                "content": [{"type": "image_body", "index": 2, "content": ""}],
            }
        )
    with pytest.raises(ValidationError, match="bbox"):
        parse_block(
            {
                "type": "image",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "content": [{"type": "image_body", "bbox": [0.2, 0.2, 0.8, 0.8], "content": ""}],
            }
        )


def test_visual_parent_rejects_other_family_annotation() -> None:
    """验证合并 annotation 模型后仍禁止视觉父块接收其他家族子块。"""
    with pytest.raises(ValidationError):
        parse_block(
            {
                "type": "image",
                "content": [
                    {"type": "image_body", "content": ""},
                    {"type": "table_caption", "content": _inline("wrong")},
                ],
            }
        )


def test_code_subtype_controls_guess_language() -> None:
    """验证 code 必须有语言，而 algorithm 禁止语言字段。"""
    with pytest.raises(ValidationError):
        parse_block({"type": "code", "sub_type": "code", "content": [{"type": "code_body", "content": "x"}]})
    with pytest.raises(ValidationError):
        parse_block(
            {
                "type": "code",
                "sub_type": "algorithm",
                "guess_lang": "python",
                "content": [{"type": "algorithm_body", "content": _inline("x")}],
            }
        )
    assert isinstance(
        parse_block(
            {"type": "code", "sub_type": "algorithm", "content": [{"type": "algorithm_body", "content": _inline("x")}]}
        ),
        CodeBlock,
    )


def test_page_info_requires_unique_strictly_increasing_top_indices() -> None:
    """验证顶层 index 必填、唯一、严格递增，但允许缺号。"""
    page = PageInfo(
        page_idx=0,
        blocks=[
            TextBlock(type="text", index=2, content=_inline("a")),
            TextBlock(type="text", index=8, content=_inline("b")),
        ],
    )
    assert [block.index for block in page.blocks] == [2, 8]
    with pytest.raises(ValidationError):
        PageInfo(page_idx=0, blocks=[TextBlock(type="text", content=_inline("missing"))])
    with pytest.raises(ValidationError):
        PageInfo(
            page_idx=0,
            blocks=[
                TextBlock(type="text", index=1, content=_inline("a")),
                TextBlock(type="text", index=1, content=_inline("b")),
            ],
        )


@pytest.mark.parametrize("block_type", ["image_body", "table_caption", "code_footnote"])
def test_page_info_rejects_visual_child_as_top_level(block_type: str) -> None:
    """验证视觉 body/caption/footnote 只能出现在对应父块内部。"""
    with pytest.raises(ValidationError):
        PageInfo.model_validate(
            {
                "page_idx": 0,
                "blocks": [{"type": block_type, "index": 0, "content": ""}],
            }
        )


def test_page_info_accepts_all_page_root_discriminators() -> None:
    """验证 PAGE_BLOCK_TYPES 中的全部页面根类型都能进入 PageInfo。"""
    payloads = _public_block_payloads()
    for block_type in PAGE_BLOCK_TYPES:
        payload = deepcopy(payloads[block_type])
        payload["index"] = 0
        content = payload.get("content")
        if isinstance(content, list):
            for child in content:
                if isinstance(child, dict) and str(child.get("type", "")).endswith("_body"):
                    child["index"] = 0
        page = PageInfo.model_validate({"page_idx": 0, "blocks": [payload]})
        assert page.blocks[0].type == block_type


@pytest.mark.parametrize("child_type", [BlockType.TEXT, BlockType.REF_TEXT])
@pytest.mark.parametrize("continues_prev", [True, None])
def test_nested_continues_prev_is_rejected_by_page_tree(child_type: str, continues_prev: bool | None) -> None:
    """验证 continues_prev 只能出现在页面顶层 text/ref_text/list/table。"""
    with pytest.raises(ValidationError, match="nested"):
        PageInfo.model_validate(
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "list",
                        "index": 0,
                        "content": [{"type": child_type, "content": _inline("x"), "continues_prev": continues_prev}],
                    }
                ],
            }
        )


def test_middle_json_pdf_requires_top_level_bbox_and_round_trips() -> None:
    """验证 PDF 顶层 bbox 约束及 MiddleJson JSON 往返。"""
    with pytest.raises(ValidationError, match="requires bbox"):
        MiddleJson(
            pages=[PageInfo(page_idx=0, blocks=[TextBlock(type="text", index=0, content=_inline("x"))])],
            is_full_document=True,
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="test",
        )
    middle_json = MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[TextBlock(type="text", index=0, bbox=(0.1, 0.1, 0.9, 0.9), content=_inline("x"))],
            )
        ],
        is_full_document=True,
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )
    restored = MiddleJson.model_validate_json(middle_json.to_json())
    assert restored == middle_json
