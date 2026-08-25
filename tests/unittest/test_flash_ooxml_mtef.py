from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import DocxModel, PptxModel, XlsxModel
from mineru.model.flash.office.docx.docx_converter import DocxConverter
from mineru.model.flash.office.legacy import LegacyOfficeResourceLimitError
from mineru.model.flash.office.legacy.limits import MAX_ASSET_TOTAL_BYTES
from mineru.model.flash.office.ooxml_equation import (
    OoxmlEquationDecoder,
    is_mathtype_equation_prog_id,
)
from mineru.model.flash.office.pptx.pptx_converter import PptxConverter
from mineru.model.flash.office.xlsx.xlsx_converter import XlsxConverter
from mineru.types import BlockType, MiddleJson, ModelJson

from _mtef_test_utils import build_equation_object, formula_corpus
from _ooxml_mtef_test_utils import (
    build_equation_docx,
    build_equation_pptx,
    build_equation_xlsx,
)


def _equation_contents(pages: list[list[dict]]) -> list[str]:
    """按分页顺序返回所有独立 equation block 内容。"""

    return [
        block["content"]
        for page in pages
        for block in page
        if block.get("type") == BlockType.EQUATION
    ]


def test_ooxml_equation_decoder_enforces_scope_icon_and_total_budget() -> None:
    """验证共享入口只接受公式 ProgID、非图标及预算内的 CFB。"""

    _name, mtef, expected = formula_corpus()[0]
    blob = build_equation_object(mtef)
    decoder = OoxmlEquationDecoder()

    assert decoder.decode(blob, prog_id="Equation.3") == expected
    assert decoder.decode(blob, prog_id="equation.3") == expected
    assert decoder.decode(blob, prog_id="Equation.DSMT4") == expected
    assert decoder.decode(blob, prog_id="Equation") == expected
    assert decoder.decode(blob, prog_id="Equation.") is None
    assert decoder.decode(blob, prog_id="Package") is None
    assert decoder.decode(blob, prog_id="Equation.3", show_as_icon=True) is None
    assert decoder.decode(b"not CFB", prog_id="Equation.3") is None

    exhausted = OoxmlEquationDecoder(total_bytes=MAX_ASSET_TOTAL_BYTES)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_asset_total_bytes"):
        exhausted.decode(blob, prog_id="Equation.3")


@pytest.mark.parametrize(
    ("prog_id", "expected"),
    [
        ("Equation", True),
        ("equation.3", True),
        ("EQUATION.DSMT4", True),
        (" Equation.Custom ", True),
        ("Equation.", False),
        ("EquationXML", False),
        ("Package", False),
        (None, False),
    ],
)
def test_mathtype_equation_prog_id_matching(
    prog_id: object | None,
    expected: bool,
) -> None:
    """验证 OLE1 Equation 和 OLE2 Equation.* 的大小写无关匹配。"""

    assert is_mathtype_equation_prog_id(prog_id) is expected


def test_docx_pptx_xlsx_decode_the_full_mtef_corpus_exactly() -> None:
    """验证三种 OOXML Model 对11类MTEF公式给出完全一致的LaTeX。"""

    corpus = formula_corpus()
    formulas = [mtef for _name, mtef, _expected in corpus]
    expected = [value for _name, _mtef, value in corpus]

    docx_pages = DocxModel().predict(BytesIO(build_equation_docx(formulas)))
    pptx_pages = PptxModel().predict(BytesIO(build_equation_pptx(formulas)))
    xlsx_pages = XlsxModel().predict(BytesIO(build_equation_xlsx(formulas)))

    assert _equation_contents(docx_pages) == expected
    assert _equation_contents(pptx_pages) == expected
    assert _equation_contents(xlsx_pages) == expected
    assert len(pptx_pages) == len(corpus)


def test_docx_mtef_flows_inline_table_header_footer_and_omml_precedence() -> None:
    """验证 DOCX 行内、表格、页眉页脚和 OMML 优先级语义。"""

    corpus = formula_corpus()
    first = corpus[0]
    second = corpus[1]

    inline = DocxModel().predict(
        BytesIO(build_equation_docx([first[1]], inline=True))
    )
    table = DocxModel().predict(
        BytesIO(build_equation_docx([first[1]], table=True))
    )
    header_footer = DocxModel().predict(
        BytesIO(build_equation_docx([first[1], second[1]], header_footer=True))
    )
    alternate = DocxModel().predict(
        BytesIO(build_equation_docx([first[1]], alternate_omml=True))
    )
    textbox = DocxModel().predict(
        BytesIO(build_equation_docx([first[1]], textbox=True))
    )

    assert inline == [[{"type": BlockType.TEXT, "content": "before <eq>x+y</eq> after"}]]
    assert table[0][0]["type"] == BlockType.TABLE
    assert "<eq>x+y</eq>" in table[0][0]["content"]
    assert header_footer == [
        [
            {"type": BlockType.HEADER, "content": "<eq>x+y</eq>"},
            {
                "type": BlockType.FOOTER,
                "content": r"<eq>\frac{a+b}{c}</eq>",
            },
        ]
    ]
    assert alternate == [[{"type": BlockType.EQUATION, "content": "z"}]]
    assert textbox == [[{"type": BlockType.EQUATION, "content": "x+y"}]]


def test_omml_precedes_mtef_and_preview_for_the_same_ooxml_object() -> None:
    """验证三种 OOXML 兼容对象只输出 OMML 分支且不重复预览。"""

    _name, mtef, _expected = formula_corpus()[0]
    cases = [
        (
            DocxModel(),
            build_equation_docx([b"invalid MTEF"], alternate_omml=True),
        ),
        (
            PptxModel(),
            build_equation_pptx([mtef], alternate_omml=True),
        ),
        (
            XlsxModel(),
            build_equation_xlsx(
                [mtef],
                anchor_mode="drawing",
                alternate_omml=True,
            ),
        ),
    ]

    for model, file_bytes in cases:
        assert model.predict(BytesIO(file_bytes)) == [
            [{"type": BlockType.EQUATION, "content": "z"}]
        ]


@pytest.mark.parametrize(
    ("model", "valid_file", "invalid_file"),
    [
        (
            DocxModel(),
            lambda mtef: build_equation_docx([mtef]),
            lambda: build_equation_docx([b"invalid MTEF"]),
        ),
        (
            PptxModel(),
            lambda mtef: build_equation_pptx([mtef]),
            lambda: build_equation_pptx([b"invalid MTEF"]),
        ),
        (
            XlsxModel(),
            lambda mtef: build_equation_xlsx([mtef]),
            lambda: build_equation_xlsx([b"invalid MTEF"]),
        ),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_valid_ooxml_mtef_suppresses_preview_and_invalid_keeps_it(
    model: Any,
    valid_file: Callable[[bytes], bytes],
    invalid_file: Callable[[], bytes],
) -> None:
    """验证有效原生公式胜出，坏 MTEF 在三种格式均保留缓存图。"""

    _name, mtef, expected = formula_corpus()[1]
    valid_pages = model.predict(BytesIO(valid_file(mtef)))
    invalid_pages = model.predict(BytesIO(invalid_file()))

    assert valid_pages == [[{"type": BlockType.EQUATION, "content": expected}]]
    assert invalid_pages[0][0]["type"] == BlockType.IMAGE
    assert invalid_pages[0][0]["image_base64"].startswith("data:image/")


@pytest.mark.parametrize("file_suffix", ["docx", "pptx", "xlsx"])
def test_ooxml_icon_mode_preserves_preview_instead_of_expanding_formula(
    file_suffix: str,
) -> None:
    """验证 DrawAspect/showAsIcon/dvAspect 图标模式不展开公式。"""

    _name, mtef, _expected = formula_corpus()[0]
    builders = {
        "docx": build_equation_docx,
        "pptx": build_equation_pptx,
        "xlsx": build_equation_xlsx,
    }
    models = {
        "docx": DocxModel(),
        "pptx": PptxModel(),
        "xlsx": XlsxModel(),
    }

    pages = models[file_suffix].predict(
        BytesIO(builders[file_suffix]([mtef], show_as_icon=True))
    )

    assert pages[0][0]["type"] == BlockType.IMAGE
    assert all(block["type"] != BlockType.EQUATION for block in pages[0])


def test_pptx_notes_equation_is_appended_as_page_footnote() -> None:
    """验证 notesSlide 中的 Equation.3 公式精确归属当前幻灯片。"""

    _name, mtef, expected = formula_corpus()[0]

    pages = PptxModel().predict(
        BytesIO(build_equation_pptx([mtef], notes=True))
    )

    assert pages == [
        [
            {
                "type": BlockType.PAGE_FOOTNOTE,
                "content": f"<eq>{expected}</eq>",
            }
        ]
    ]


@pytest.mark.parametrize("anchor_mode", ["objectPr", "drawing", "vml", "none"])
def test_xlsx_equation_anchor_variants_and_tail_fallback(anchor_mode: str) -> None:
    """验证 XLSX objectPr、DrawingML、VML 与无 anchor 的稳定输出。"""

    _name, mtef, expected = formula_corpus()[0]

    pages = XlsxModel().predict(
        BytesIO(build_equation_xlsx([mtef], anchor_mode=anchor_mode))
    )

    assert pages == [[{"type": BlockType.EQUATION, "content": expected}]]


def test_xlsx_mtef_inside_table_and_hidden_sheet_behavior() -> None:
    """验证表内公式不重复输出，隐藏公式工作表沿用现有跳过策略。"""

    _name, mtef, expected = formula_corpus()[0]
    table_pages = XlsxModel().predict(
        BytesIO(build_equation_xlsx([mtef], cell_value="value"))
    )
    hidden_pages = XlsxModel().predict(
        BytesIO(build_equation_xlsx([mtef], hidden=True))
    )

    assert len(table_pages[0]) == 1
    assert table_pages[0][0]["type"] == BlockType.TABLE
    assert f"<eq>{expected}</eq>" in table_pages[0][0]["content"]
    assert hidden_pages == [[]]


def test_xlsx_linked_equation_never_loads_external_target() -> None:
    """验证外链 Equation.3 只保留包内预览，不访问外部对象。"""

    _name, mtef, _expected = formula_corpus()[0]

    pages = XlsxModel().predict(
        BytesIO(build_equation_xlsx([mtef], linked=True))
    )

    assert pages[0][0]["type"] == BlockType.IMAGE
    assert all(block["type"] != BlockType.EQUATION for block in pages[0])


@pytest.mark.parametrize("file_suffix", ["docx", "pptx", "xlsx"])
def test_ooxml_mtef_backend_analyze_sync_async_contract(file_suffix: str) -> None:
    """验证 MTEF 公式贯穿三种现代 Office 同步与异步严格契约。"""

    _name, mtef, expected = formula_corpus()[2]
    builders = {
        "docx": build_equation_docx,
        "pptx": build_equation_pptx,
        "xlsx": build_equation_xlsx,
    }
    file_bytes = builders[file_suffix]([mtef])
    middle, model = doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(file_bytes, file_suffix=file_suffix)  # type: ignore[arg-type]
    )

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == file_suffix
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle


def test_ooxml_mtef_model_input_streams_remain_open() -> None:
    """验证三种 Model 均不关闭调用方持有的输入流。"""

    _name, mtef, _expected = formula_corpus()[0]
    cases = [
        (DocxModel(), build_equation_docx([mtef])),
        (PptxModel(), build_equation_pptx([mtef])),
        (XlsxModel(), build_equation_xlsx([mtef])),
    ]
    for model, file_bytes in cases:
        stream = BytesIO(file_bytes)
        model.predict(stream)
        assert not stream.closed


def test_ooxml_converter_reuse_does_not_leak_equations_between_documents() -> None:
    """验证三个 converter 重用时公式缓存和分页状态都会重置。"""

    first = formula_corpus()[0]
    second = formula_corpus()[1]
    cases = [
        (DocxConverter(), build_equation_docx),
        (PptxConverter(), build_equation_pptx),
        (XlsxConverter(), build_equation_xlsx),
    ]
    for converter, builder in cases:
        converter.convert(BytesIO(builder([first[1]])))
        assert _equation_contents(converter.pages) == [first[2]]
        converter.convert(BytesIO(builder([second[1]])))
        assert _equation_contents(converter.pages) == [second[2]]


@pytest.mark.parametrize("file_suffix", ["docx", "pptx", "xlsx"])
def test_invalid_ooxml_mtef_preview_exports_to_sidecar(
    file_suffix: str,
    tmp_path: Path,
) -> None:
    """验证三种格式的图片回退导出后没有 base64 残留。"""

    builders = {
        "docx": build_equation_docx,
        "pptx": build_equation_pptx,
        "xlsx": build_equation_xlsx,
    }
    middle, _model = doc_analyze(
        builders[file_suffix]([b"invalid MTEF"]),
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )

    result = middle.export(tmp_path / file_suffix)

    assert len(result.image_paths) == 1
    assert result.image_paths[0].stat().st_size > 0
    assert "base64," not in result.json_path.read_text(encoding="utf-8")
