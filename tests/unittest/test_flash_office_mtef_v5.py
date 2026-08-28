from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import BytesIO

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import (
    DocModel,
    DocxModel,
    PptModel,
    PptxModel,
    XlsModel,
    XlsxModel,
)
from mineru.model.flash.office.docx.docx_converter import DocxConverter
from mineru.model.flash.office.pptx.pptx_converter import PptxConverter
from mineru.model.flash.office.xlsx.xlsx_converter import XlsxConverter
from mineru.types import BlockType, MiddleJson, ModelJson

from _docx_equationxml_test_utils import (
    build_equationxml_docx,
    build_word_2003_equation_xml,
)
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls, label_cell
from _mtef_test_utils import build_equation_doc, formula_corpus
from _mtef_v5_test_utils import v5_formula_corpus
from _ooxml_mtef_test_utils import (
    build_equation_docx,
    build_equation_pptx,
    build_equation_xlsx,
)
from _span_test_utils import equation


def _equation_contents(pages: list[list[dict]]) -> list[str]:
    """按分页和 block 顺序收集独立公式内容。"""

    return [block["content"] for page in pages for block in page if block.get("type") == BlockType.EQUATION]


def _has_preview_image(pages: list[list[dict]]) -> bool:
    """判断 model-list 是否至少保留一个图片回退。"""

    return any(block.get("type") == BlockType.IMAGE for page in pages for block in page)


def test_six_office_models_decode_the_full_mtef_v5_corpus() -> None:
    """验证 DOC/DOCX/PPT/PPTX/XLS/XLSX 对完整 v5 语料输出一致。"""

    corpus = v5_formula_corpus()
    formulas = [mtef for _name, mtef, _expected in corpus]
    expected = [latex for _name, _mtef, latex in corpus]
    prog_id = "Equation.DSMT4"
    cases = [
        (
            DocModel(),
            build_equation_doc(
                [(1000 + index, mtef) for index, mtef in enumerate(formulas)],
                prog_id=prog_id,
            ),
        ),
        (
            DocxModel(),
            build_equation_docx(formulas, prog_id=prog_id),
        ),
        (
            PptModel(),
            build_equation_ppt(formulas, preview=False, prog_id=prog_id),
        ),
        (
            PptxModel(),
            build_equation_pptx(formulas, prog_id=prog_id),
        ),
        (
            XlsModel(),
            build_equation_xls(
                [(2000 + index, mtef) for index, mtef in enumerate(formulas)],
                preview=False,
                prog_id=prog_id,
            ),
        ),
        (
            XlsxModel(),
            build_equation_xlsx(formulas, prog_id=prog_id),
        ),
    ]

    for model, file_bytes in cases:
        assert _equation_contents(model.predict(BytesIO(file_bytes))) == expected


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_equation_docx),
        (PptxModel(), build_equation_pptx),
        (XlsxModel(), build_equation_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_ooxml_prog_id_and_mtef_version_are_orthogonal(
    model: DocxModel | PptxModel | XlsxModel,
    builder: Callable[..., bytes],
) -> None:
    """验证 Equation.3 可承载 v5，DSMT4/Equation 也可承载 v3。"""

    _v5_name, v5, v5_expected = v5_formula_corpus()[1]
    _v3_name, v3, v3_expected = formula_corpus()[1]

    assert _equation_contents(model.predict(BytesIO(builder([v5], prog_id="Equation.3")))) == [v5_expected]
    assert _equation_contents(model.predict(BytesIO(builder([v3], prog_id="Equation.DSMT4")))) == [v3_expected]
    assert _equation_contents(model.predict(BytesIO(builder([v5], prog_id="Equation")))) == [v5_expected]


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_equation_docx),
        (PptxModel(), build_equation_pptx),
        (XlsxModel(), build_equation_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_non_equation_ooxml_prog_id_is_not_probed(
    model: DocxModel | PptxModel | XlsxModel,
    builder: Callable[..., bytes],
) -> None:
    """验证非公式或空后缀 ProgID 不探测有效 v5 OLE 内容。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    for prog_id in ("Package", "Equation."):
        pages = model.predict(BytesIO(builder([mtef], prog_id=prog_id)))
        assert _equation_contents(pages) == []


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_equation_docx),
        (PptxModel(), build_equation_pptx),
        (XlsxModel(), build_equation_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_mtef_v5_icon_mode_keeps_icon_preview(
    model: DocxModel | PptxModel | XlsxModel,
    builder: Callable[..., bytes],
) -> None:
    """验证图标模式不展开有效 v5，只保留原图标预览。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    pages = model.predict(
        BytesIO(
            builder(
                [mtef],
                prog_id="Equation.DSMT4",
                show_as_icon=True,
            )
        )
    )

    assert _equation_contents(pages) == []
    assert _has_preview_image(pages)


def test_xlsx_linked_mtef_v5_object_never_loads_external_target() -> None:
    """验证 XLSX 外链 MathType 对象只保留包内预览，不访问目标。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    pages = XlsxModel().predict(
        BytesIO(
            build_equation_xlsx(
                [mtef],
                prog_id="Equation.DSMT4",
                linked=True,
            )
        )
    )

    assert _equation_contents(pages) == []
    assert _has_preview_image(pages)


@pytest.mark.parametrize(
    ("model", "file_bytes"),
    [
        (
            DocModel(),
            build_equation_doc(
                [(1, bytes([4, 1, 0, 3, 5, 0]))],
                preview_storage_ids={1},
                prog_id="Equation.DSMT4",
            ),
        ),
        (
            DocxModel(),
            build_equation_docx(
                [bytes([4, 1, 0, 3, 5, 0])],
                prog_id="Equation.DSMT4",
            ),
        ),
        (
            PptModel(),
            build_equation_ppt(
                [bytes([4, 1, 0, 3, 5, 0])],
                prog_id="Equation.DSMT4",
            ),
        ),
        (
            PptxModel(),
            build_equation_pptx(
                [bytes([4, 1, 0, 3, 5, 0])],
                prog_id="Equation.DSMT4",
            ),
        ),
        (
            XlsModel(),
            build_equation_xls(
                [(1, bytes([4, 1, 0, 3, 5, 0]))],
                prog_id="Equation.DSMT4",
            ),
        ),
        (
            XlsxModel(),
            build_equation_xlsx(
                [bytes([4, 1, 0, 3, 5, 0])],
                prog_id="Equation.DSMT4",
            ),
        ),
    ],
    ids=["doc", "docx", "ppt", "pptx", "xls", "xlsx"],
)
def test_mtef_v4_keeps_preview_in_all_six_formats(
    model: DocModel | DocxModel | PptModel | PptxModel | XlsModel | XlsxModel,
    file_bytes: bytes,
) -> None:
    """验证六格式遇到 v4 均不猜测解析并保留缓存预览。"""

    pages = model.predict(BytesIO(file_bytes))

    assert _equation_contents(pages) == []
    assert _has_preview_image(pages)


@pytest.mark.parametrize(
    "file_suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx"],
)
def test_mtef_v5_runs_through_sync_async_analyze(
    file_suffix: str,
) -> None:
    """验证 v5 贯穿六格式同步/异步严格 Analyze 契约。"""

    _name, mtef, expected = v5_formula_corpus()[3]
    prog_id = "Equation.DSMT4"
    builders: dict[str, Callable[[], bytes]] = {
        "doc": lambda: build_equation_doc([(10, mtef)], prog_id=prog_id),
        "docx": lambda: build_equation_docx([mtef], prog_id=prog_id),
        "ppt": lambda: build_equation_ppt([mtef], preview=False, prog_id=prog_id),
        "pptx": lambda: build_equation_pptx([mtef], prog_id=prog_id),
        "xls": lambda: build_equation_xls([(10, mtef)], preview=False, prog_id=prog_id),
        "xlsx": lambda: build_equation_xlsx([mtef], prog_id=prog_id),
    }
    file_bytes = builders[file_suffix]()
    middle, model = doc_analyze(
        file_bytes,
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(
            file_bytes,
            file_suffix=file_suffix,  # type: ignore[arg-type]
        )
    )

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == file_suffix
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle


def test_mtef_v5_enters_docx_xls_xlsx_table_cells() -> None:
    """验证三种文档/表格格式的 v5 公式进入 cell HTML 且不重复输出。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    docx = DocxModel().predict(
        BytesIO(
            build_equation_docx(
                [mtef],
                table=True,
                prog_id="Equation.DSMT4",
            )
        )
    )
    xls = XlsModel().predict(
        BytesIO(
            build_equation_xls(
                [(10, mtef)],
                cell_records=label_cell(0, 0, "value"),
                prog_id="Equation.DSMT4",
            )
        )
    )
    xlsx = XlsxModel().predict(
        BytesIO(
            build_equation_xlsx(
                [mtef],
                cell_value="value",
                prog_id="Equation.DSMT4",
            )
        )
    )

    for pages in (docx, xls, xlsx):
        assert _equation_contents(pages) == []
        assert pages[0][0]["type"] == BlockType.TABLE
        assert f"<eq>{expected}</eq>" in pages[0][0]["content"]


def test_docx_equationxml_and_ooxml_omml_precede_mtef_v5() -> None:
    """验证 DOCX equationxml 以及 PPTX/XLSX OMML 都优先于有效 v5。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    docx = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [build_word_2003_equation_xml("x")],
                keep_mtef=True,
                mtef_payloads=[mtef],
                prog_id="Equation.DSMT4",
            )
        )
    )
    pptx = PptxModel().predict(
        BytesIO(
            build_equation_pptx(
                [mtef],
                alternate_omml=True,
                prog_id="Equation.DSMT4",
            )
        )
    )
    xlsx = XlsxModel().predict(
        BytesIO(
            build_equation_xlsx(
                [mtef],
                anchor_mode="drawing",
                alternate_omml=True,
                prog_id="Equation.DSMT4",
            )
        )
    )

    assert _equation_contents(docx) == ["x"]
    assert _equation_contents(pptx) == ["z"]
    assert _equation_contents(xlsx) == ["z"]


def test_docx_header_footer_and_pptx_notes_accept_mtef_v5() -> None:
    """验证 v5 在 DOCX 页眉页脚和 PPTX notes 中保持归属。"""

    corpus = v5_formula_corpus()
    first = corpus[0]
    second = corpus[1]
    docx = DocxModel().predict(
        BytesIO(
            build_equation_docx(
                [first[1], second[1]],
                header_footer=True,
                prog_id="Equation.DSMT4",
            )
        )
    )
    pptx = PptxModel().predict(
        BytesIO(
            build_equation_pptx(
                [first[1]],
                notes=True,
                prog_id="Equation.DSMT4",
            )
        )
    )

    assert docx == [
        [
            {"type": BlockType.HEADER, "content": [equation(first[2])]},
            {"type": BlockType.FOOTER, "content": [equation(second[2])]},
        ]
    ]
    assert pptx == [
        [
            {
                "type": BlockType.PAGE_FOOTNOTE,
                "content": [equation(first[2])],
            }
        ]
    ]


def test_mtef_v5_model_streams_remain_open_in_all_six_formats() -> None:
    """验证六种 Model 的 v5 路径均不关闭调用方输入流。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    prog_id = "Equation.DSMT4"
    cases = [
        (DocModel(), build_equation_doc([(1, mtef)], prog_id=prog_id)),
        (DocxModel(), build_equation_docx([mtef], prog_id=prog_id)),
        (PptModel(), build_equation_ppt([mtef], prog_id=prog_id)),
        (PptxModel(), build_equation_pptx([mtef], prog_id=prog_id)),
        (XlsModel(), build_equation_xls([(1, mtef)], prog_id=prog_id)),
        (XlsxModel(), build_equation_xlsx([mtef], prog_id=prog_id)),
    ]

    for model, file_bytes in cases:
        stream = BytesIO(file_bytes)
        model.predict(stream)
        assert not stream.closed


def test_modern_office_converter_reuse_resets_v5_state() -> None:
    """验证现代 Office converter 复用时 v5 缓存和分页状态不会串文档。"""

    first = v5_formula_corpus()[0]
    second = v5_formula_corpus()[1]
    prog_id = "Equation.DSMT4"
    cases = [
        (DocxConverter(), build_equation_docx),
        (PptxConverter(), build_equation_pptx),
        (XlsxConverter(), build_equation_xlsx),
    ]

    for converter, builder in cases:
        converter.convert(BytesIO(builder([first[1]], prog_id=prog_id)))
        assert _equation_contents(converter.pages) == [first[2]]
        converter.convert(BytesIO(builder([second[1]], prog_id=prog_id)))
        assert _equation_contents(converter.pages) == [second[2]]
