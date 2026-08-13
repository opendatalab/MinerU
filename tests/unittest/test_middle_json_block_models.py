from __future__ import annotations

from copy import deepcopy
from typing import get_args

import pytest
from pydantic import ValidationError

import mineru.types as types_module
from mineru.backend.utils.raw_block_types import RAW_ONLY_BLOCK_TYPES
from mineru.types import (
    BLOCK_ADAPTER,
    BLOCK_TYPES,
    BlockType,
    BlockTypes,
    CodeBlock,
    EquationBlock,
    ListBlock,
    MiddleJson,
    PageInfo,
    TextBlock,
    parse_block,
)


def _public_block_payloads() -> dict[str, dict[str, object]]:
    """构造 29 个公开 discriminator 的最小合法载荷。"""
    text_leaf = {"type": "text", "content": "item"}
    payloads: dict[str, dict[str, object]] = {
        "aside_text": {"type": "aside_text", "content": "aside"},
        "doc_title": {"type": "doc_title", "content": "title"},
        "footer": {"type": "footer", "content": "footer"},
        "formula_number": {"type": "formula_number", "content": "(1)"},
        "header": {"type": "header", "content": "header"},
        "index": {"type": "index", "content": [text_leaf]},
        "equation": {"type": "equation", "content": "x=1"},
        "list": {"type": "list", "content": [text_leaf]},
        "page_footnote": {"type": "page_footnote", "content": "note"},
        "page_number": {"type": "page_number", "content": "1"},
        "paragraph_title": {"type": "paragraph_title", "content": "section"},
        "ref_text": {"type": "ref_text", "content": "reference"},
        "text": {"type": "text", "content": "text"},
        "image_body": {"type": "image_body", "content": None},
        "image_caption": {"type": "image_caption", "content": "caption"},
        "image_footnote": {"type": "image_footnote", "content": "note"},
        "table_body": {"type": "table_body", "content": "<table></table>"},
        "table_caption": {"type": "table_caption", "content": "caption"},
        "table_footnote": {"type": "table_footnote", "content": "note"},
        "chart_body": {"type": "chart_body", "content": None},
        "chart_caption": {"type": "chart_caption", "content": "caption"},
        "chart_footnote": {"type": "chart_footnote", "content": "note"},
        "code_body": {"type": "code_body", "content": "print(1)"},
        "code_caption": {"type": "code_caption", "content": "caption"},
        "code_footnote": {"type": "code_footnote", "content": "note"},
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
    class_values = {
        value
        for name, value in vars(BlockType).items()
        if name.isupper()
    }

    assert class_values == BLOCK_TYPES
    assert set(get_args(BlockTypes)) == BLOCK_TYPES


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


def test_cross_type_fields_and_unknown_fields_are_rejected() -> None:
    """验证 Text/RefText 的专属字段边界与 extra forbid。"""
    with pytest.raises(ValidationError):
        parse_block({"type": "text", "content": "x", "anchor": "a"})
    with pytest.raises(ValidationError):
        parse_block({"type": "ref_text", "content": "x", "continues_prev": True})
    with pytest.raises(ValidationError):
        parse_block({"type": "text", "content": "x", "unknown": 1})


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
        parse_block({"type": "text", "content": "x", "bbox": bbox})


def test_recursive_list_and_index_round_trip() -> None:
    """验证四层 List 与递归 Index 的顺序和具体类型可无损恢复。"""
    list_payload: dict[str, object] = {"type": "text", "content": "leaf"}
    for _ in range(4):
        list_payload = {"type": "list", "content": [{"type": "text", "content": "item"}, list_payload]}
    index_payload = {
        "type": "index",
        "content": [{"type": "text", "content": "one"}, {"type": "index", "content": [{"type": "text", "content": "two"}]}],
    }

    list_block = parse_block(list_payload)
    index_block = parse_block(index_payload)

    assert isinstance(list_block, ListBlock)
    assert isinstance(list_block.content[1], ListBlock)
    assert [child.content for child in index_block.content if isinstance(child, TextBlock)] == ["one"]
    assert parse_block(list_block.to_dict(skip_defaults=False)) == list_block
    assert parse_block(index_block.to_dict(skip_defaults=False)) == index_block


def test_list_subtype_allows_mixed_direct_text_children() -> None:
    """验证 List subtype 是子项类型的统计结果，不约束每个直接子项。"""
    block = parse_block(
        {
            "type": "list", "sub_type": "ref_text",
            "content": [
                {"type": "text", "content": "x"},
                {"type": "ref_text", "content": "r1"},
                {"type": "ref_text", "content": "r2"},
                {"type": "list", "sub_type": "text", "content": [{"type": "ref_text", "content": "nested"}]},
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
                "content": [{"type": "image_body", "index": 2, "content": None}],
            }
        )
    with pytest.raises(ValidationError, match="bbox"):
        parse_block(
            {
                "type": "image",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "content": [{"type": "image_body", "bbox": [0.2, 0.2, 0.8, 0.8], "content": None}],
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
                "content": [{"type": "code_body", "content": "x"}],
            }
        )
    assert isinstance(
        parse_block({"type": "code", "sub_type": "algorithm", "content": [{"type": "code_body", "content": "x"}]}),
        CodeBlock,
    )


def test_page_info_requires_unique_strictly_increasing_top_indices() -> None:
    """验证顶层 index 必填、唯一、严格递增，但允许缺号。"""
    page = PageInfo(
        page_idx=0,
        blocks=[
            TextBlock(type="text", index=2, content="a"),
            TextBlock(type="text", index=8, content="b"),
        ],
    )
    assert [block.index for block in page.blocks] == [2, 8]
    with pytest.raises(ValidationError):
        PageInfo(page_idx=0, blocks=[TextBlock(type="text", content="missing")])
    with pytest.raises(ValidationError):
        PageInfo(
            page_idx=0,
            blocks=[TextBlock(type="text", index=1, content="a"), TextBlock(type="text", index=1, content="b")],
        )


def test_nested_continues_prev_is_rejected_by_page_tree() -> None:
    """验证 continues_prev 只能出现在页面顶层 text/list/table。"""
    with pytest.raises(ValidationError, match="nested"):
        PageInfo.model_validate(
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "list",
                        "index": 0,
                        "content": [{"type": "text", "content": "x", "continues_prev": True}],
                    }
                ],
            }
        )

    with pytest.raises(ValidationError, match="nested"):
        PageInfo.model_validate(
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "list",
                        "index": 0,
                        "content": [{"type": "text", "content": "x", "continues_prev": None}],
                    }
                ],
            }
        )


def test_middle_json_pdf_requires_top_level_bbox_and_round_trips() -> None:
    """验证 PDF 顶层 bbox 约束及 MiddleJson JSON 往返。"""
    with pytest.raises(ValidationError, match="requires bbox"):
        MiddleJson(
            pages=[PageInfo(page_idx=0, blocks=[TextBlock(type="text", index=0, content="x")])],
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="test",
        )
    middle_json = MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[TextBlock(type="text", index=0, bbox=(0.1, 0.1, 0.9, 0.9), content="x")],
            )
        ],
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )
    restored = MiddleJson.model_validate_json(middle_json.to_json())
    assert restored == middle_json
