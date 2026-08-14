# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from mineru.model.flash.docx.docx_converter import DocxConverter


NO_BREAK_HYPHEN = "‑"
SOFT_HYPHEN = "­"


def _save_docx_bytes(doc: Document) -> bytes:
    """将内存中的 DOCX 文档序列化为字节。"""
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _convert_docx_bytes(file_bytes: bytes) -> list[dict]:
    """调用当前 Flash DOCX Converter 并拉平分页块。"""
    converter = DocxConverter()
    converter.convert(BytesIO(file_bytes))
    return [block for page in converter.pages for block in page]


def _table_blocks(blocks: list[dict]) -> list[dict]:
    """从拉平块列表中筛选表格块。"""
    return [block for block in blocks if block["type"] == "table"]


def _build_docx_with_no_break_hyphen_table() -> bytes:
    """构造同时含不间断连字符和编号列表的整表丢失用例。"""
    doc = Document()
    doc.add_heading("1 项目概述", level=1)

    table = doc.add_table(rows=2, cols=2)
    paragraph = table.cell(0, 0).paragraphs[0]
    run = paragraph.add_run("NCI")
    etree.SubElement(run._r, qn("w:noBreakHyphen"))
    paragraph.add_run("CTCAE 标准")
    numbered = table.cell(0, 1).paragraphs[0]
    numbered.style = doc.styles["List Number"]
    numbered.add_run("第一条编号内容")

    doc.add_heading("2 进度安排", level=1)
    plain = doc.add_table(rows=1, cols=2)
    plain.cell(0, 0).paragraphs[0].add_run("里程碑")
    plain.cell(0, 1).paragraphs[0].add_run("2026-08-01")
    return _save_docx_bytes(doc)


def _build_docx_with_sym_table(*, char: str) -> bytes:
    """构造同时含 Symbol 字符和编号列表的表格。"""
    doc = Document()
    doc.add_heading("符号表格", level=1)
    table = doc.add_table(rows=1, cols=2)
    paragraph = table.cell(0, 0).paragraphs[0]
    run = paragraph.add_run("符号")
    etree.SubElement(
        run._r,
        qn("w:sym"),
        {qn("w:char"): char, qn("w:font"): "Symbol"},
    )
    paragraph.add_run("结束")
    numbered = table.cell(0, 1).paragraphs[0]
    numbered.style = doc.styles["List Number"]
    numbered.add_run("编号项")
    return _save_docx_bytes(doc)


def _build_plain_docx_table() -> bytes:
    """构造不含特殊字符的普通表格对照组。"""
    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).paragraphs[0].add_run("APTT")
    table.cell(0, 1).paragraphs[0].add_run("部分凝血酶时间")
    return _save_docx_bytes(doc)


def test_no_break_hyphen_table_not_lost() -> None:
    """验证不间断连字符与编号列表并存时整表不会丢失。"""
    tables = _table_blocks(_convert_docx_bytes(_build_docx_with_no_break_hyphen_table()))

    assert len(tables) == 2
    assert f"NCI{NO_BREAK_HYPHEN}CTCAE" in tables[0]["content"]
    assert "<li>" in tables[0]["content"]
    assert "第一条编号内容" in tables[0]["content"]
    assert "里程碑" in tables[1]["content"]


def test_mapped_sym_table_takes_mammoth_path() -> None:
    """验证普通 Symbol 映射字符与 Mammoth 表格签名保持一致。"""
    tables = _table_blocks(_convert_docx_bytes(_build_docx_with_sym_table(char="0022")))

    assert len(tables) == 1
    assert "∀" in tables[0]["content"]
    assert "<li>" in tables[0]["content"]


def test_f0_sym_table_takes_low_byte_fallback() -> None:
    """验证 F0 前缀 Symbol 字符按 Mammoth 规则使用低字节回退映射。"""
    tables = _table_blocks(_convert_docx_bytes(_build_docx_with_sym_table(char="F0B7")))

    assert len(tables) == 1
    assert "•" in tables[0]["content"]
    assert "<li>" in tables[0]["content"]


def test_plain_table_still_emitted() -> None:
    """验证特殊字符签名修复不影响普通表格。"""
    tables = _table_blocks(_convert_docx_bytes(_build_plain_docx_table()))

    assert len(tables) == 1
    assert "APTT" in tables[0]["content"]


def test_xml_table_signature_renders_special_chars() -> None:
    """验证 XML 签名保留不间断连字符和软连字符。"""
    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    paragraph = table.cell(0, 0).paragraphs[0]
    run = paragraph.add_run("NCI")
    etree.SubElement(run._r, qn("w:noBreakHyphen"))
    paragraph.add_run("CTCAE")
    second_paragraph = table.cell(0, 1).paragraphs[0]
    second_run = second_paragraph.add_run("软连")
    etree.SubElement(second_run._r, qn("w:softHyphen"))
    second_paragraph.add_run("字符")

    signature = DocxConverter._xml_table_signature(table._tbl)

    assert f"NCI{NO_BREAK_HYPHEN}CTCAE" in signature["text"]
    assert f"软连{SOFT_HYPHEN}字符" in signature["text"]


def test_xml_table_signature_renders_sym_like_mammoth() -> None:
    """验证已映射 Symbol 输出字符，真正未映射 Symbol 输出空字符串。"""
    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    paragraph = table.cell(0, 0).paragraphs[0]
    run = paragraph.add_run("映射")
    etree.SubElement(
        run._r,
        qn("w:sym"),
        {qn("w:char"): "0022", qn("w:font"): "Symbol"},
    )
    paragraph.add_run("结束")
    second_paragraph = table.cell(0, 1).paragraphs[0]
    second_run = second_paragraph.add_run("未映射")
    etree.SubElement(
        second_run._r,
        qn("w:sym"),
        {qn("w:char"): "F0B7", qn("w:font"): "UnknownFont"},
    )
    second_paragraph.add_run("结束")

    signature = DocxConverter._xml_table_signature(table._tbl)

    assert signature["text"] == "映射∀结束未映射结束"


def test_xml_table_signature_excludes_omml_equation_text() -> None:
    """验证 Mammoth 不渲染的 OMML 公式文本不参与表格签名。"""
    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    paragraph = table.cell(0, 0).paragraphs[0]
    run = paragraph.add_run("正文")
    equation = etree.SubElement(run._r, qn("m:oMath"))
    equation_run = etree.SubElement(equation, qn("m:r"))
    etree.SubElement(equation_run, qn("m:t")).text = "E=mc2"
    table.cell(0, 1).paragraphs[0].add_run("单元格")

    signature = DocxConverter._xml_table_signature(table._tbl)

    assert "正文" in signature["text"]
    assert "E=mc2" not in signature["text"]
