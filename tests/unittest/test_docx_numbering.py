from io import BytesIO
from typing import Any

import pytest
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from mineru.backend.office.office_magic_model import parse_list_block
from mineru.model.docx.docx_converter import DocxConverter
from mineru.render.office.output import _flatten_list_items
from mineru.types import BlockType


def _append_numbering_level(
    abstract_num: Any,
    *,
    ilvl: int,
    start: int,
    num_fmt: str,
    lvl_text: str,
) -> None:
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), str(ilvl))

    start_element = OxmlElement("w:start")
    start_element.set(qn("w:val"), str(start))
    level.append(start_element)

    format_element = OxmlElement("w:numFmt")
    format_element.set(qn("w:val"), num_fmt)
    level.append(format_element)

    text_element = OxmlElement("w:lvlText")
    text_element.set(qn("w:val"), lvl_text)
    level.append(text_element)
    abstract_num.append(level)


def _build_numbered_docx(
    levels: list[dict[str, int | str]], paragraphs: list[tuple[int, str]]
) -> BytesIO:
    document = Document()
    numbering = document.part.numbering_part.element

    abstract_num = OxmlElement("w:abstractNum")
    abstract_num.set(qn("w:abstractNumId"), "42")
    for level in levels:
        _append_numbering_level(abstract_num, **level)
    numbering.append(abstract_num)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), "42")
    abstract_num_id = OxmlElement("w:abstractNumId")
    abstract_num_id.set(qn("w:val"), "42")
    num.append(abstract_num_id)
    numbering.append(num)

    for ilvl, text in paragraphs:
        paragraph = document.add_paragraph(text)
        num_pr = OxmlElement("w:numPr")
        ilvl_element = OxmlElement("w:ilvl")
        ilvl_element.set(qn("w:val"), str(ilvl))
        num_id_element = OxmlElement("w:numId")
        num_id_element.set(qn("w:val"), "42")
        num_pr.extend((ilvl_element, num_id_element))
        paragraph._p.get_or_add_pPr().append(num_pr)

    stream = BytesIO()
    document.save(stream)
    stream.seek(0)
    return stream


def _collect_list_text(blocks: list[dict[str, Any]]) -> list[str]:
    result = []
    for block in blocks:
        if block.get("type") == BlockType.TEXT:
            result.append(block["content"])
            continue
        result.extend(_collect_list_text(block.get("content", [])))
    return result


@pytest.mark.parametrize(
    ("value", "num_fmt", "expected"),
    [
        (36, "chineseCounting", "三十六"),
        (101, "chineseCountingThousand", "一百零一"),
        (27, "upperLetter", "AA"),
        (28, "lowerLetter", "ab"),
        (36, "upperRoman", "XXXVI"),
        (9, "decimalZero", "09"),
    ],
)
def test_format_numbering_value(value: int, num_fmt: str, expected: str) -> None:
    assert DocxConverter._format_numbering_value(value, num_fmt) == expected


def test_custom_chinese_multilevel_labels_are_preserved() -> None:
    stream = _build_numbered_docx(
        levels=[
            {
                "ilvl": 0,
                "start": 36,
                "num_fmt": "chineseCountingThousand",
                "lvl_text": "第%1条",
            },
            {
                "ilvl": 1,
                "start": 1,
                "num_fmt": "chineseCounting",
                "lvl_text": "（%2）",
            },
        ],
        paragraphs=[
            (0, "本制度由行政部负责解释。"),
            (1, "例行检查"),
            (1, "专项检查"),
            (0, "本制度自发布之日起施行。"),
        ],
    )

    converter = DocxConverter()
    converter.convert(stream)

    assert _collect_list_text(converter.pages[0]) == [
        "第三十六条 本制度由行政部负责解释。",
        "（一） 例行检查",
        "（二） 专项检查",
        "第三十七条 本制度自发布之日起施行。",
    ]

    list_block = parse_list_block(converter.pages[0][0])
    assert list_block is not None
    assert _flatten_list_items(list_block) == [
        "- 第三十六条 本制度由行政部负责解释。",
        "    - （一） 例行检查",
        "    - （二） 专项检查",
        "- 第三十七条 本制度自发布之日起施行。",
    ]


def test_plain_decimal_list_keeps_markdown_ordering() -> None:
    stream = _build_numbered_docx(
        levels=[
            {
                "ilvl": 0,
                "start": 3,
                "num_fmt": "decimal",
                "lvl_text": "%1.",
            }
        ],
        paragraphs=[(0, "First"), (0, "Second")],
    )

    converter = DocxConverter()
    converter.convert(stream)

    list_block = converter.pages[0][0]
    assert list_block["attribute"] == "ordered"
    assert list_block["start"] == 3
    assert [item["content"] for item in list_block["content"]] == ["First", "Second"]
