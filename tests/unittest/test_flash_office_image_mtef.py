from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

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
from mineru.model.flash.office.doc.doc_converter import DocConverter
from mineru.model.flash.office.docx.docx_converter import DocxConverter
from mineru.model.flash.office.doc.models import (
    DocImage,
    DocImagePayload,
    DocParagraph,
    DocTable,
    DocTableCell,
    DocTableRow,
)
from mineru.model.flash.office.xlsx.xlsx_converter import XlsxConverter
from mineru.model.flash.office.pptx.pptx_converter import PptxConverter
from mineru.types import BlockType, MiddleJson, ModelJson

from _docx_equationxml_test_utils import (
    build_equationxml_docx,
    build_word_2003_equation_xml,
)
from _image_mtef_test_utils import (
    apps_mfcc_comments,
    baseline_wmf_comment,
    build_baseline_only_gif,
    build_gif_with_mtef,
    build_wmf,
)
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls, label_cell
from _mtef_test_utils import build_equation_doc, formula_corpus
from _mtef_v5_test_utils import v5_formula_corpus
from _office_image_mtef_test_utils import (
    build_image_docx,
    build_image_pptx,
    build_image_xlsx,
)
from _ooxml_mtef_test_utils import (
    build_equation_docx,
    build_equation_pptx,
    build_equation_xlsx,
)


def _equation_contents(pages: list[list[dict]]) -> list[str]:
    """按分页顺序收集独立 equation block 内容。"""

    return [
        block["content"]
        for page in pages
        for block in page
        if block.get("type") == BlockType.EQUATION
    ]


def _has_image(pages: list[list[dict]]) -> bool:
    """判断 model-list 是否至少保留一个图片 block。"""

    return any(
        block.get("type") == BlockType.IMAGE
        for page in pages
        for block in page
    )


def _wmf_formula(mtef: bytes) -> bytes:
    """把 MTEF 包装为跨 chunk AppsMFCC WMF。"""

    return build_wmf(
        apps_mfcc_comments(
            mtef,
            chunk_size=7,
        ),
        placeable=True,
    )


def test_legacy_doc_ppt_xls_recover_wmf_comment_after_bad_native() -> None:
    """验证旧三格式 Native 失败后用同一预览 WMF comment 恢复公式。"""

    _name, mtef, expected = v5_formula_corpus()[1]
    preview = _wmf_formula(mtef)
    invalid = b"invalid native"
    cases = [
        (
            DocModel(),
            build_equation_doc(
                [(1, invalid)],
                preview_storage_ids={1},
                preview_payloads={1: preview},
            ),
        ),
        (
            PptModel(),
            build_equation_ppt(
                [invalid],
                preview_payload=preview,
            ),
        ),
        (
            XlsModel(),
            build_equation_xls(
                [(1, invalid)],
                preview_payload=preview,
            ),
        ),
    ]

    for model, file_bytes in cases:
        pages = model.predict(BytesIO(file_bytes))
        assert _equation_contents(pages) == [expected]
        assert not _has_image(pages)


def test_legacy_doc_direct_gif_comment_recovers_after_bad_native() -> None:
    """验证 DOC PICF magic fallback 保留并解析原始 GIF/001。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    image = build_gif_with_mtef(mtef, chunk_size=3)
    pages = DocModel().predict(
        BytesIO(
            build_equation_doc(
                [(1, b"invalid native")],
                preview_storage_ids={1},
                preview_payloads={1: image},
            )
        )
    )

    assert _equation_contents(pages) == [expected]
    assert not _has_image(pages)


def test_legacy_native_mtef_precedes_conflicting_wmf_comment() -> None:
    """验证旧三格式有效 Native 优先于内容不同的 WMF comment。"""

    _native_name, native, expected = formula_corpus()[0]
    _image_name, image_mtef, _image_expected = v5_formula_corpus()[1]
    preview = _wmf_formula(image_mtef)
    cases = [
        (
            DocModel(),
            build_equation_doc(
                [(1, native)],
                preview_storage_ids={1},
                preview_payloads={1: preview},
            ),
        ),
        (
            PptModel(),
            build_equation_ppt([native], preview_payload=preview),
        ),
        (
            XlsModel(),
            build_equation_xls([(1, native)], preview_payload=preview),
        ),
    ]

    for model, file_bytes in cases:
        assert _equation_contents(model.predict(BytesIO(file_bytes))) == [expected]


def test_xls_wmf_comment_equation_enters_table_cell() -> None:
    """验证 XLS 图片 comment 公式进入 cell HTML 且不重复输出。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    pages = XlsModel().predict(
        BytesIO(
            build_equation_xls(
                [(1, b"invalid native")],
                preview_payload=_wmf_formula(mtef),
                cell_records=label_cell(0, 0, "value"),
            )
        )
    )

    assert _equation_contents(pages) == []
    assert len(pages[0]) == 1
    assert pages[0][0]["type"] == BlockType.TABLE
    assert f"<eq>{expected}</eq>" in pages[0][0]["content"]


def test_doc_image_comment_equation_enters_nested_table_html() -> None:
    """验证 DOC 段落图片与独立图片在嵌套表格中都写入 eq。"""

    payload = DocImagePayload(
        data=b"",
        extension="wmf",
        content_type="image/wmf",
        equation_latex="x+y",
    )
    paragraph = DocParagraph(
        cp_start=0,
        cp_end=1,
        images=[payload],
    )
    nested = DocTable(
        cp_start=1,
        cp_end=2,
        rows=[
            DocTableRow(
                cells=[
                    DocTableCell(
                        blocks=[DocImage(cp=1, payload=payload)]
                    )
                ]
            )
        ],
    )
    table = DocTable(
        cp_start=0,
        cp_end=2,
        rows=[
            DocTableRow(
                cells=[DocTableCell(blocks=[paragraph, nested])]
            )
        ],
    )

    html = DocConverter._table_html(table)

    assert html.count("<eq>x+y</eq>") == 2
    assert "<img" not in html


def test_docx_picture_comment_flows_inline_table_header_and_standalone() -> None:
    """验证 DOCX 普通 WMF/GIF 图片进入统一公式 token 重建链路。"""

    first = v5_formula_corpus()[0]
    second = v5_formula_corpus()[1]
    wmf = _wmf_formula(first[1])
    gif = build_gif_with_mtef(second[1], chunk_size=5)
    standalone = DocxModel().predict(BytesIO(build_image_docx(wmf)))
    inline = DocxModel().predict(
        BytesIO(build_image_docx(gif, inline=True))
    )
    table = DocxModel().predict(
        BytesIO(build_image_docx(gif, table=True))
    )
    header = DocxModel().predict(
        BytesIO(build_image_docx(gif, header=True))
    )

    assert standalone == [
        [{"type": BlockType.EQUATION, "content": first[2]}]
    ]
    assert inline == [
        [
            {
                "type": BlockType.TEXT,
                "content": f"before <eq>{second[2]}</eq> after",
            }
        ]
    ]
    assert table[0][0]["type"] == BlockType.TABLE
    assert f"<eq>{second[2]}</eq>" in table[0][0]["content"]
    assert "<img" not in table[0][0]["content"]
    assert header == [
        [
            {
                "type": BlockType.HEADER,
                "content": f"<eq>{second[2]}</eq>",
            }
        ]
    ]


def test_docx_picture_comment_flows_title_and_list() -> None:
    """验证 DOCX 图片公式在标题和列表中保持 eq 行内语义。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    image = build_gif_with_mtef(mtef)
    title = DocxModel().predict(
        BytesIO(
            build_image_docx(
                image,
                inline=True,
                paragraph_style="Title",
            )
        )
    )
    bullet = DocxModel().predict(
        BytesIO(
            build_image_docx(
                image,
                inline=True,
                paragraph_style="List Bullet",
            )
        )
    )

    assert title[0][0] == {
        "type": BlockType.DOC_TITLE,
        "level": 1,
        "content": f"before <eq>{expected}</eq> after",
    }
    assert bullet[0][0]["type"] == BlockType.LIST
    assert bullet[0][0]["content"][0]["content"] == (
        f"before <eq>{expected}</eq> after"
    )


@pytest.mark.parametrize("carrier", ["wmf", "gif"])
def test_pptx_picture_comment_equation_keeps_shape_order(carrier: str) -> None:
    """验证 PPTX 普通图片 comment 以原 shape 顺序输出 equation。"""

    _name, mtef, expected = v5_formula_corpus()[2]
    image = (
        _wmf_formula(mtef)
        if carrier == "wmf"
        else build_gif_with_mtef(mtef, chunk_size=7)
    )

    pages = PptxModel().predict(BytesIO(build_image_pptx(image)))

    assert _equation_contents(pages) == [expected]
    assert not _has_image(pages)


@pytest.mark.parametrize("carrier", ["wmf", "gif"])
def test_pptx_notes_picture_comment_becomes_page_footnote(
    carrier: str,
) -> None:
    """验证 PPTX notes 中的 WMF/GIF 图片公式输出 page_footnote。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    image = (
        _wmf_formula(mtef)
        if carrier == "wmf"
        else build_gif_with_mtef(mtef)
    )
    pages = PptxModel().predict(
        BytesIO(
            build_image_pptx(
                image,
                notes=True,
            )
        )
    )

    assert pages == [
        [
            {
                "type": BlockType.PAGE_FOOTNOTE,
                "content": f"<eq>{expected}</eq>",
            }
        ]
    ]


def test_pptx_notes_bad_ole_native_uses_wmf_preview_comment() -> None:
    """验证 notes 中 OLE Native 失败时继续解析同对象 WMF preview。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    pages = PptxModel().predict(
        BytesIO(
            build_equation_pptx(
                [b"invalid native"],
                notes=True,
                preview_image=_wmf_formula(mtef),
            )
        )
    )

    assert pages == [
        [
            {
                "type": BlockType.PAGE_FOOTNOTE,
                "content": f"<eq>{expected}</eq>",
            }
        ]
    ]


@pytest.mark.parametrize("carrier", ["wmf", "gif"])
def test_xlsx_image_comment_equation_enters_visual_and_table_paths(
    carrier: str,
) -> None:
    """验证 XLSX 普通 WMF/GIF comment 按 anchor 输出或进入表格 cell。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    image = (
        _wmf_formula(mtef)
        if carrier == "wmf"
        else build_gif_with_mtef(mtef, chunk_size=3)
    )
    standalone = XlsxModel().predict(
        BytesIO(build_image_xlsx(image))
    )
    table = XlsxModel().predict(
        BytesIO(build_image_xlsx(image, cell_value="value"))
    )

    assert _equation_contents(standalone) == [expected]
    assert _equation_contents(table) == []
    assert table[0][0]["type"] == BlockType.TABLE
    assert f"<eq>{expected}</eq>" in table[0][0]["content"]


def test_xlsx_cell_image_comment_returns_eq_html() -> None:
    """验证 XLSX cellimages media 在返回 img 前尝试 comment 公式。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    image = build_gif_with_mtef(mtef)
    package = BytesIO()
    with ZipFile(package, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/media/image1.gif", image)
    converter = XlsxConverter()
    converter.zf = ZipFile(BytesIO(package.getvalue()))
    converter.cell_image_map = {"image-id": "media/image1.gif"}
    try:
        html = converter._get_cell_image('DISPIMG("image-id")')
    finally:
        converter.zf.close()

    assert html == f"<eq>{expected}</eq>"


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_equation_docx),
        (PptxModel(), build_equation_pptx),
        (XlsxModel(), build_equation_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_ooxml_bad_native_recovers_from_wmf_preview_comment(
    model: DocxModel | PptxModel | XlsxModel,
    builder: Callable[..., bytes],
) -> None:
    """验证现代三格式 OLE Native 失败后升级有效 WMF preview。"""

    _name, mtef, expected = v5_formula_corpus()[1]
    pages = model.predict(
        BytesIO(
            builder(
                [b"invalid native"],
                prog_id="Equation.DSMT4",
                preview_image=_wmf_formula(mtef),
            )
        )
    )

    assert _equation_contents(pages) == [expected]
    assert not _has_image(pages)


@pytest.mark.parametrize(
    ("model", "builder"),
    [
        (DocxModel(), build_equation_docx),
        (PptxModel(), build_equation_pptx),
        (XlsxModel(), build_equation_xlsx),
    ],
    ids=["docx", "pptx", "xlsx"],
)
def test_ooxml_native_precedes_conflicting_wmf_preview_comment(
    model: DocxModel | PptxModel | XlsxModel,
    builder: Callable[..., bytes],
) -> None:
    """验证现代三格式有效 OLE Native 优先于图片 comment。"""

    _native_name, native, expected = formula_corpus()[0]
    _image_name, image_mtef, _image_expected = v5_formula_corpus()[1]
    pages = model.predict(
        BytesIO(
            builder(
                [native],
                prog_id="Equation.DSMT4",
                preview_image=_wmf_formula(image_mtef),
            )
        )
    )

    assert _equation_contents(pages) == [expected]


def test_docx_equationxml_and_ooxml_omml_precede_image_comment() -> None:
    """验证 equationxml/OMML 继续高于同对象图片 comment。"""

    _name, image_mtef, _expected = v5_formula_corpus()[0]
    preview = _wmf_formula(image_mtef)
    docx = DocxModel().predict(
        BytesIO(
            build_equationxml_docx(
                [build_word_2003_equation_xml("x")],
                keep_mtef=True,
                mtef_payloads=[b"invalid native"],
                preview_image=preview,
            )
        )
    )
    pptx = PptxModel().predict(
        BytesIO(
            build_equation_pptx(
                [b"invalid native"],
                alternate_omml=True,
                preview_image=preview,
            )
        )
    )
    xlsx = XlsxModel().predict(
        BytesIO(
            build_equation_xlsx(
                [b"invalid native"],
                anchor_mode="drawing",
                alternate_omml=True,
                preview_image=preview,
            )
        )
    )

    assert _equation_contents(docx) == ["x"]
    assert _equation_contents(pptx) == ["z"]
    assert _equation_contents(xlsx) == ["z"]


def test_bad_image_comment_keeps_original_picture_or_placeholder() -> None:
    """验证 baseline-only WMF 不升级公式并继续输出图片。"""

    image = build_wmf([baseline_wmf_comment(0)], placeable=True)
    cases = [
        (DocxModel(), build_image_docx(image)),
        (PptxModel(), build_image_pptx(image)),
    ]

    for model, file_bytes in cases:
        pages = model.predict(BytesIO(file_bytes))
        assert _equation_contents(pages) == []
        assert _has_image(pages)


def test_bad_image_comment_preview_exports_to_sidecar(tmp_path: Path) -> None:
    """验证未升级的图片导出后 sidecar 完整且 JSON 不残留 base64。"""

    middle, _model = doc_analyze(
        build_image_docx(build_baseline_only_gif()),
        file_suffix="docx",
    )

    result = middle.export(tmp_path / "image-mtef-fallback")

    assert len(result.image_paths) == 1
    assert result.image_paths[0].stat().st_size > 0
    assert "base64," not in result.json_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "file_suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx"],
)
def test_image_comment_mtef_runs_through_sync_async_analyze(
    file_suffix: str,
) -> None:
    """验证图片 comment 公式贯穿六格式同步/异步严格 Analyze。"""

    _name, mtef, expected = v5_formula_corpus()[0]
    wmf = _wmf_formula(mtef)
    gif = build_gif_with_mtef(mtef)
    builders = {
        "doc": lambda: build_equation_doc(
            [(1, b"invalid")],
            preview_storage_ids={1},
            preview_payloads={1: wmf},
        ),
        "docx": lambda: build_image_docx(gif),
        "ppt": lambda: build_equation_ppt(
            [b"invalid"],
            preview_payload=wmf,
        ),
        "pptx": lambda: build_image_pptx(gif),
        "xls": lambda: build_equation_xls(
            [(1, b"invalid")],
            preview_payload=wmf,
        ),
        "xlsx": lambda: build_image_xlsx(gif),
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
    assert middle.pages[0].blocks[0].type == BlockType.EQUATION
    assert middle.pages[0].blocks[0].content == expected
    assert async_model == model
    assert async_middle == middle


def test_modern_converter_reuse_resets_image_comment_decoder() -> None:
    """验证现代 converter 复用时图片公式缓存和资源预算不会串文档。"""

    first = v5_formula_corpus()[0]
    second = v5_formula_corpus()[1]
    cases = [
        (DocxConverter(), build_image_docx),
        (PptxConverter(), build_image_pptx),
        (XlsxConverter(), build_image_xlsx),
    ]

    for converter, builder in cases:
        converter.convert(
            BytesIO(builder(build_gif_with_mtef(first[1])))
        )
        assert _equation_contents(converter.pages) == [first[2]]
        converter.convert(
            BytesIO(builder(build_gif_with_mtef(second[1])))
        )
        assert _equation_contents(converter.pages) == [second[2]]
