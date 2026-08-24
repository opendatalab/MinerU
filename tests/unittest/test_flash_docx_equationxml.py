from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from lxml import etree  # type: ignore[reportAttributeAccessIssue]
import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import DocxModel
from mineru.model.flash.docx.docx_converter import DocxConverter
from mineru.model.flash.docx.equationxml import DocxEquationXmlDecoder
from mineru.model.flash.legacy_office.errors import (
    LegacyOfficeResourceLimitError,
)
from mineru.render._internal.docx.math import latex_to_omml
from mineru.types import BlockType, MiddleJson, ModelJson

import mineru.model.flash.docx.equationxml as equationxml_module
from _docx_equationxml_test_utils import (
    M_NS,
    WORD_2003_NS,
    build_equationxml_docx,
    build_word_2003_equation_xml,
    build_word_2003_equation_xml_from_omml,
    build_word_2003_fraction_equation_xml,
)
from _mtef_test_utils import formula_corpus


def _equation_contents(pages: list[list[dict]]) -> list[str]:
    """按分页顺序提取独立公式 block 的 LaTeX 内容。"""

    return [
        block["content"]
        for page in pages
        for block in page
        if block["type"] == BlockType.EQUATION
    ]


def _invalid_equationxml_with_multiple_math() -> str:
    """构造包含两个 ``m:oMath`` 的非规范 Equation XML。"""

    root = etree.fromstring(build_word_2003_equation_xml("x").encode())
    math_paragraph = root.find(
        f".//{{{M_NS}}}oMathPara"
    )
    assert math_paragraph is not None
    second = etree.SubElement(math_paragraph, f"{{{M_NS}}}oMath")
    run = etree.SubElement(second, f"{{{M_NS}}}r")
    etree.SubElement(run, f"{{{M_NS}}}t").text = "y"
    return etree.tostring(root, encoding="unicode")


def _doctype_equationxml() -> str:
    """构造带外部实体声明的 Equation XML，验证解析器不会读取实体。"""

    return (
        '<!DOCTYPE w:wordDocument [<!ENTITY probe SYSTEM "file:///etc/passwd">]>'
        f'<w:wordDocument xmlns:w="{WORD_2003_NS}" xmlns:m="{M_NS}">'
        "<w:body><w:p><m:oMathPara><m:oMath><m:r><m:t>&probe;</m:t>"
        "</m:r></m:oMath></m:oMathPara></w:p></w:body></w:wordDocument>"
    )


def _equationxml_with_forbidden_pict() -> str:
    """构造包含 Word 2003 ``pict`` 的不可恢复 Equation XML。"""

    root = etree.fromstring(build_word_2003_equation_xml("x").encode())
    run = root.find(f".//{{{M_NS}}}r")
    assert run is not None
    etree.SubElement(run, f"{{{WORD_2003_NS}}}pict")
    return etree.tostring(root, encoding="unicode")


def test_equationxml_decoder_converts_spec_document_and_fraction() -> None:
    """验证规范 Word 2003 XML 包装可复用现有 OMML 转换器。"""

    decoder = DocxEquationXmlDecoder()

    assert decoder.decode(build_word_2003_equation_xml()) == "x+y"
    assert (
        decoder.decode(build_word_2003_fraction_equation_xml())
        == r"\frac{a}{b}"
    )


@pytest.mark.parametrize(
    ("source_latex", "expected_latex"),
    [
        (r"\sqrt{x}", r"\sqrt{x}"),
        (r"x^2", r"x^{2}"),
        (r"\int_0^1 x", r"\int_{0}^{1}x"),
        (r"\left(x\right)", r"\left(x\right)"),
        (
            r"\begin{matrix}a&b\\c&d\end{matrix}",
            r"\begin{matrix}a&b\\c&d\end{matrix}",
        ),
        ("α+β", r"\alpha +\beta"),
    ],
)
def test_equationxml_decoder_reuses_omml_formula_coverage(
    source_latex: str,
    expected_latex: str,
) -> None:
    """验证根式、脚本、积分、定界符、矩阵和 Unicode 沿用 OMML 能力。"""

    equation = latex_to_omml(source_latex, display=False)
    equation_xml = build_word_2003_equation_xml_from_omml(equation)

    assert DocxEquationXmlDecoder().decode(equation_xml) == expected_latex


@pytest.mark.parametrize(
    "payload",
    [
        "<broken",
        "<m:oMath xmlns:m='http://schemas.openxmlformats.org/officeDocument/2006/math'/>",
        _invalid_equationxml_with_multiple_math(),
        _doctype_equationxml(),
        _equationxml_with_forbidden_pict(),
        "<!--comment-->" + build_word_2003_equation_xml("x"),
    ],
)
def test_equationxml_decoder_rejects_malformed_or_unsafe_documents(
    payload: str,
) -> None:
    """验证损坏、裸 OMML、多公式和实体文档整体回退。"""

    assert DocxEquationXmlDecoder().decode(payload) is None


def test_equationxml_decoder_enforces_entry_total_and_cache_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单属性、累计资源限制及相同 payload 缓存不重复计费。"""

    first = build_word_2003_equation_xml("x")
    second = build_word_2003_equation_xml("y")
    monkeypatch.setattr(equationxml_module, "MAX_ENTRY_BYTES", len(first.encode()) + 10)
    monkeypatch.setattr(equationxml_module, "MAX_ASSET_TOTAL_BYTES", len(first.encode()) + 1)
    decoder = DocxEquationXmlDecoder()

    assert decoder.decode(first) == "x"
    assert decoder.decode(first) == "x"
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_asset_total_bytes"):
        decoder.decode(second)

    monkeypatch.setattr(equationxml_module, "MAX_ENTRY_BYTES", 8)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_entry_bytes"):
        DocxEquationXmlDecoder().decode(first)


def test_docx_equationxml_standalone_inline_table_and_textbox_flows() -> None:
    """验证正文独立、行内、表格和文本框 Equation XML 输出语义。"""

    equation_xml = build_word_2003_equation_xml()
    standalone = DocxModel().predict(
        BytesIO(build_equationxml_docx([equation_xml]))
    )
    inline = DocxModel().predict(
        BytesIO(build_equationxml_docx([equation_xml], inline=True))
    )
    table = DocxModel().predict(
        BytesIO(build_equationxml_docx([equation_xml], table=True))
    )
    textbox = DocxModel().predict(
        BytesIO(build_equationxml_docx([equation_xml], textbox=True))
    )

    assert standalone == [
        [{"type": BlockType.EQUATION, "content": "x+y"}]
    ]
    assert inline == [
        [
            {
                "type": BlockType.TEXT,
                "content": "before <eq>x+y</eq> after",
            }
        ]
    ]
    assert table[0][0]["type"] == BlockType.TABLE
    assert "<eq>x+y</eq>" in table[0][0]["content"]
    assert "<img" not in table[0][0]["content"]
    assert textbox == [
        [{"type": BlockType.EQUATION, "content": "x+y"}]
    ]


def test_docx_equationxml_title_list_header_and_footer_flows() -> None:
    """验证标题、列表及页眉页脚沿用现有公式内容重建路径。"""

    first = build_word_2003_equation_xml("x")
    second = build_word_2003_fraction_equation_xml()
    title = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [first],
                inline=True,
                paragraph_style="Title",
            )
        )
    )
    bullet = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [first],
                inline=True,
                paragraph_style="ListBullet",
            )
        )
    )
    header_footer = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [first, second],
                header_footer=True,
            )
        )
    )

    assert title[0][0] == {
        "type": BlockType.DOC_TITLE,
        "level": 1,
        "content": "before <eq>x</eq> after",
    }
    assert bullet == [
        [
            {
                "type": BlockType.LIST,
                "attribute": "unordered",
                "content": [
                    {
                        "type": BlockType.TEXT,
                        "content": "before <eq>x</eq> after",
                    }
                ],
                "ilevel": 0,
            }
        ]
    ]
    assert header_footer == [
        [
            {"type": BlockType.HEADER, "content": "<eq>x</eq>"},
            {
                "type": BlockType.FOOTER,
                "content": r"<eq>\frac{a}{b}</eq>",
            },
        ]
    ]


def test_docx_equationxml_precedence_and_preview_fallback() -> None:
    """验证 OMML、Equation XML、MTEF 和预览图片的确定优先级。"""

    equation_xml = build_word_2003_equation_xml("x")
    _name, mtef, mtef_latex = formula_corpus()[1]
    native_omml = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [equation_xml],
                alternate_omml=True,
                keep_mtef=True,
                mtef_payloads=[mtef],
            )
        )
    )
    equationxml_over_mtef = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [equation_xml],
                keep_mtef=True,
                mtef_payloads=[mtef],
            )
        )
    )
    mtef_over_bad_equationxml = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                ["<broken"],
                keep_mtef=True,
                mtef_payloads=[mtef],
            )
        )
    )
    preview = DocxModel().predict(
        BytesIO(build_equationxml_docx(["<broken"]))
    )

    assert native_omml == [
        [{"type": BlockType.EQUATION, "content": "z"}]
    ]
    assert equationxml_over_mtef == [
        [{"type": BlockType.EQUATION, "content": "x"}]
    ]
    assert mtef_over_bad_equationxml == [
        [{"type": BlockType.EQUATION, "content": mtef_latex}]
    ]
    assert len(preview[0]) == 1
    assert preview[0][0]["type"] == BlockType.IMAGE


def test_docx_equationxml_same_payload_is_not_semantically_deduplicated() -> None:
    """验证缓存只减少解码成本，不合并两个独立公式 shape。"""

    equation_xml = build_word_2003_equation_xml("x")
    pages = DocxModel().predict(
        BytesIO(build_equationxml_docx([equation_xml, equation_xml]))
    )

    assert _equation_contents(pages) == ["x", "x"]


def test_docx_equationxml_shared_preview_is_suppressed_per_shape() -> None:
    """验证共享图片关系只抑制有效公式所属 shape，不全局删除图片。"""

    pages = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [build_word_2003_equation_xml("x"), "<broken"],
                share_preview=True,
            )
        )
    )

    assert [block["type"] for block in pages[0]] == [
        BlockType.EQUATION,
        BlockType.IMAGE,
    ]
    assert pages[0][0]["content"] == "x"


def test_docx_equationxml_attribute_is_unescaped_exactly_once() -> None:
    """验证外层属性和内层 XML 实体各解码一次，不发生二次展开。"""

    pages = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [build_word_2003_equation_xml("x&y")]
            )
        )
    )

    assert _equation_contents(pages) == [r"x\&y"]


def test_docx_equationxml_analyze_lifecycle_and_strict_contracts() -> None:
    """验证 Equation XML 贯穿同步异步 Analyze 且调用方流保持打开。"""

    file_bytes = build_equationxml_docx(
        [build_word_2003_fraction_equation_xml()]
    )
    stream = BytesIO(file_bytes)
    pages = DocxModel().predict(stream)
    middle, model = doc_analyze(file_bytes, file_suffix="docx")
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(file_bytes, file_suffix="docx")
    )

    assert not stream.closed
    assert _equation_contents(pages) == [r"\frac{a}{b}"]
    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == "docx"
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == r"\frac{a}{b}"
    assert async_model == model
    assert async_middle == middle


def test_docx_equationxml_converter_reuse_resets_decoder_state() -> None:
    """验证 converter 复用时页、缓存、资源预算和告警状态均重置。"""

    converter = DocxConverter()
    converter.convert(
        BytesIO(
            build_equationxml_docx([build_word_2003_equation_xml("x")])
        )
    )
    assert _equation_contents(converter.pages) == ["x"]

    converter.convert(
        BytesIO(
            build_equationxml_docx([build_word_2003_equation_xml("y")])
        )
    )
    assert _equation_contents(converter.pages) == ["y"]


def test_invalid_docx_equationxml_preview_exports_to_sidecar(
    tmp_path: Path,
) -> None:
    """验证坏 Equation XML 的图片回退导出后无 base64 JSON 残留。"""

    middle, _model = doc_analyze(
        build_equationxml_docx(["<broken"]),
        file_suffix="docx",
    )

    result = middle.export(tmp_path / "docx-equationxml")

    assert len(result.image_paths) == 1
    assert result.image_paths[0].stat().st_size > 0
    assert "base64," not in result.json_path.read_text(encoding="utf-8")
