"""构造带 VML ``equationxml`` 公式的确定性 DOCX 测试包。"""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

from lxml import etree  # type: ignore[reportAttributeAccessIssue]

from _mtef_test_utils import _TINY_PNG
from _ooxml_mtef_test_utils import (
    M_NS,
    O_NS,
    V_NS,
    W_NS,
    _rewrite_zip,
    build_equation_docx,
)

WORD_2003_NS = "http://schemas.microsoft.com/office/word/2003/wordml"


def _wrap_word_2003_equation(equation: etree._Element) -> str:
    """把一个 OMML ``oMath`` 包装为规范要求的完整 Word 2003 XML 文档。"""

    root = etree.Element(
        f"{{{WORD_2003_NS}}}wordDocument",
        nsmap={"w": WORD_2003_NS, "m": M_NS},
    )
    body = etree.SubElement(root, f"{{{WORD_2003_NS}}}body")
    paragraph = etree.SubElement(body, f"{{{WORD_2003_NS}}}p")
    math_paragraph = etree.SubElement(paragraph, f"{{{M_NS}}}oMathPara")
    math_paragraph.append(equation)
    return etree.tostring(root, encoding="unicode")


def build_word_2003_equation_xml(text: str = "x+y") -> str:
    """构造只含一个普通数学 run 的规范 Equation XML 属性值。"""

    equation = etree.Element(f"{{{M_NS}}}oMath")
    run = etree.SubElement(equation, f"{{{M_NS}}}r")
    etree.SubElement(run, f"{{{M_NS}}}t").text = text
    return _wrap_word_2003_equation(equation)


def build_word_2003_equation_xml_from_omml(
    equation: etree._Element,
) -> str:
    """把测试生成的 OMML ``oMath`` 节点包装为 Equation XML 属性值。"""

    if equation.tag != f"{{{M_NS}}}oMath":
        raise ValueError("Equation XML fixture requires an m:oMath element")
    return _wrap_word_2003_equation(equation)


def build_word_2003_fraction_equation_xml(
    numerator: str = "a",
    denominator: str = "b",
) -> str:
    """构造包含分式对象的规范 Equation XML 属性值。"""

    equation = etree.Element(f"{{{M_NS}}}oMath")
    fraction = etree.SubElement(equation, f"{{{M_NS}}}f")
    for tag_name, text in (("num", numerator), ("den", denominator)):
        argument = etree.SubElement(fraction, f"{{{M_NS}}}{tag_name}")
        run = etree.SubElement(argument, f"{{{M_NS}}}r")
        etree.SubElement(run, f"{{{M_NS}}}t").text = text
    return _wrap_word_2003_equation(equation)


def _set_paragraph_style(shape: etree._Element, style_id: str) -> None:
    """给承载 Equation XML shape 的最近段落设置 Word 段落样式。"""

    paragraphs = shape.xpath("ancestor::w:p[1]", namespaces={"w": W_NS})
    if not paragraphs:
        return
    paragraph = paragraphs[0]
    paragraph_properties = paragraph.find(f"{{{W_NS}}}pPr")
    if paragraph_properties is None:
        paragraph_properties = etree.Element(f"{{{W_NS}}}pPr")
        paragraph.insert(0, paragraph_properties)
    style = paragraph_properties.find(f"{{{W_NS}}}pStyle")
    if style is None:
        style = etree.SubElement(paragraph_properties, f"{{{W_NS}}}pStyle")
    style.set(f"{{{W_NS}}}val", style_id)


def _patch_equationxml_part(
    payload: bytes,
    equation_xml_values: list[str],
    *,
    keep_mtef: bool,
    paragraph_style: str | None,
    share_preview: bool,
) -> tuple[bytes, int]:
    """向一个 Word XML part 的公式 shape 写入属性并可移除 OLE 语义对象。"""

    root = etree.fromstring(payload)
    shapes = [
        shape
        for shape in root.findall(f".//{{{V_NS}}}shape")
        if shape.find(f"{{{V_NS}}}imagedata") is not None
    ]
    for shape, equation_xml in zip(shapes, equation_xml_values, strict=False):
        shape.set("equationxml", equation_xml)
        if paragraph_style:
            _set_paragraph_style(shape, paragraph_style)
        if keep_mtef:
            continue
        container = shape.getparent()
        if container is None or container.tag != f"{{{W_NS}}}object":
            continue
        for ole_object in container.findall(f"{{{O_NS}}}OLEObject"):
            container.remove(ole_object)
        container.tag = f"{{{W_NS}}}pict"
    if share_preview and len(shapes) > 1:
        first_image = shapes[0].find(f"{{{V_NS}}}imagedata")
        first_rel_id = (
            first_image.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if first_image is not None
            else None
        )
        if first_rel_id:
            for shape in shapes[1:]:
                image = shape.find(f"{{{V_NS}}}imagedata")
                if image is not None:
                    image.set(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
                        first_rel_id,
                    )
    return (
        etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        ),
        len(shapes),
    )


def build_equationxml_docx(
    equation_xml_values: list[str],
    *,
    inline: bool = False,
    table: bool = False,
    header_footer: bool = False,
    alternate_omml: bool = False,
    textbox: bool = False,
    paragraph_style: str | None = None,
    keep_mtef: bool = False,
    mtef_payloads: list[bytes] | None = None,
    share_preview: bool = False,
    prog_id: str = "Equation.3",
    preview_image: bytes | None = None,
) -> bytes:
    """基于现有 OOXML fixture 构造 Equation XML 正文及兼容分支。"""

    if not equation_xml_values:
        raise ValueError("Equation XML fixture requires at least one formula")
    payloads = mtef_payloads or [b"invalid MTEF"] * len(equation_xml_values)
    if len(payloads) != len(equation_xml_values):
        raise ValueError("Equation XML and MTEF fixture counts must match")

    package = build_equation_docx(
        payloads,
        inline=inline,
        table=table,
        header_footer=header_footer,
        alternate_omml=alternate_omml,
        textbox=textbox,
        prog_id=prog_id,
        preview_image=preview_image or _TINY_PNG,
    )
    replacements: dict[str, bytes] = {}
    consumed = 0
    with ZipFile(BytesIO(package)) as source:
        part_names = [
            name
            for name in source.namelist()
            if name.startswith("word/")
            and name.endswith(".xml")
            and "/_rels/" not in name
        ]
        if header_footer:
            part_names.sort(
                key=lambda name: (
                    0 if "/header" in name else 1 if "/footer" in name else 2,
                    name,
                )
            )
        for part_name in part_names:
            remaining = equation_xml_values[consumed:]
            if not remaining:
                break
            patched, shape_count = _patch_equationxml_part(
                source.read(part_name),
                remaining,
                keep_mtef=keep_mtef,
                paragraph_style=paragraph_style,
                share_preview=share_preview,
            )
            if shape_count:
                replacements[part_name] = patched
                consumed += min(shape_count, len(remaining))
    if consumed != len(equation_xml_values):
        raise ValueError("Equation XML fixture could not locate every VML shape")
    return _rewrite_zip(package, replacements, {})
