# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.oxml.xmlchemy import BaseOxmlElement

from mineru.backend.office.docx_analyze import office_docx_analyze
from mineru.backend.office.model_output_to_middle_json import result_to_middle_json
from mineru.backend.office.office_middle_json_mkcontent import union_make
from mineru.utils.enum_class import BlockType, MakeMode


def _append_value(
    parent: BaseOxmlElement, tag: str, value: object
) -> BaseOxmlElement:
    element = OxmlElement(tag)
    element.set(qn("w:val"), str(value))
    parent.append(element)
    return element


def _add_numbering_definition(
    document: DocxDocument,
    *,
    num_id: int,
    abstract_num_id: int,
) -> None:
    # 在内存文档中创建一套独立的十进制编号规则，不依赖任何外部 DOCX 文件。
    numbering = document.part.numbering_part.element

    # 抽象编号定义：第 0 层从 1 开始，显示模板为“%1.”。
    abstract_num = OxmlElement("w:abstractNum")
    abstract_num.set(qn("w:abstractNumId"), str(abstract_num_id))
    _append_value(abstract_num, "w:multiLevelType", "multilevel")
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    _append_value(level, "w:start", 1)
    _append_value(level, "w:numFmt", "decimal")
    _append_value(level, "w:lvlText", "%1.")
    abstract_num.append(level)
    numbering.append(abstract_num)

    # 创建编号实例，并将其关联到上面的抽象编号定义。
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    _append_value(num, "w:abstractNumId", abstract_num_id)
    numbering.append(num)


def _add_numbered_heading(
    document: DocxDocument,
    text: str,
    *,
    num_id: int,
) -> None:
    paragraph = document.add_paragraph(text)
    properties = paragraph._p.get_or_add_pPr()

    # 编号层级设为 0，表示使用“%1.”这一第一级编号规则。
    num_properties = OxmlElement("w:numPr")
    _append_value(num_properties, "w:ilvl", 0)
    _append_value(num_properties, "w:numId", num_id)
    properties.append(num_properties)
    # 大纲层级设为 1，对应内部二级标题，刻意与编号层级构造不同的维度。
    _append_value(properties, "w:outlineLvl", 1)


def _analyze_document(document: DocxDocument) -> tuple[dict, list]:
    # 文档只在内存中保存和解析，测试过程不读取磁盘上的 Word 文件。
    stream = BytesIO()
    document.save(stream)
    return office_docx_analyze(stream.getvalue())


def _model_titles(model_output: list[list[dict]]) -> list[dict]:
    return [
        block
        for page in model_output
        for block in page
        if block.get("type") == BlockType.TITLE
    ]


def _middle_titles(middle_json: dict) -> list[dict]:
    return [
        block
        for page in middle_json["pdf_info"]
        for block in page["para_blocks"]
        if block.get("type") == BlockType.TITLE
    ]


def test_heading_number_uses_word_numbering_level_instead_of_outline_level() -> None:
    # 使用无业务含义的虚拟标题构造最小可复现文档。
    document = Document()
    num_id = 42
    abstract_num_id = 7
    virtual_titles = ["虚拟标题甲", "虚拟标题乙"]
    # 使用不同的实例 ID 和抽象定义 ID，验证完整的编号定义映射链路。
    _add_numbering_definition(
        document,
        num_id=num_id,
        abstract_num_id=abstract_num_id,
    )
    _add_numbered_heading(
        document,
        virtual_titles[0],
        num_id=num_id,
    )
    _add_numbered_heading(
        document,
        virtual_titles[1],
        num_id=num_id,
    )

    middle_json, model_output = _analyze_document(document)

    # 大纲层级仍应解析为二级标题，不能为了修正编号而改成一级标题。
    model_titles = _model_titles(model_output)
    assert [block["level"] for block in model_titles] == [2, 2]
    # 编号应来自 Word 的第 0 层编号模板，而不是根据二级标题层级重新推算。
    assert [block["section_number"] for block in model_titles] == ["1.", "2."]
    assert [block["section_number"] for block in _middle_titles(middle_json)] == [
        "1.",
        "2.",
    ]

    # 验证真实输出链路保留“二级标题 + 第一级编号”的组合。
    markdown = union_make(middle_json["pdf_info"], MakeMode.MM_MD, "")
    assert f"## 1. {virtual_titles[0]}" in markdown
    assert f"## 2. {virtual_titles[1]}" in markdown
    assert f"1.1 {virtual_titles[0]}" not in markdown


def test_legacy_office_title_without_explicit_number_keeps_fallback() -> None:
    # 构造没有显式 section_number 的旧格式输入，确认原有回退逻辑仍然可用。
    middle_json = result_to_middle_json(
        [[
            {
                "type": BlockType.TITLE,
                "level": 2,
                "is_numbered_style": True,
                "content": "虚拟兼容标题",
            }
        ]],
        None,
    )

    titles = _middle_titles(middle_json)
    assert len(titles) == 1
    assert titles[0]["section_number"] == "1.1"
