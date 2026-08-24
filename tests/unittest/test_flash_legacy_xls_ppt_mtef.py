from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import BytesIO

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import PptModel, XlsModel
from mineru.model.flash.legacy_office import LegacyOfficeResourceLimitError
from mineru.model.flash.legacy_office.limits import MAX_ENTRY_BYTES
from mineru.types import BlockType, MiddleJson, ModelJson

from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls, label_cell
from _mtef_test_utils import formula_corpus


def test_xls_equation_editor_corpus_decodes_to_exact_equation_blocks() -> None:
    """验证 XLS 的 MBD/OBJ 绑定把完整公式语料恢复为精确 LaTeX。"""

    corpus = formula_corpus()
    file_bytes = build_equation_xls(
        [
            (100 + index, mtef)
            for index, (_name, mtef, _expected) in enumerate(corpus)
        ],
        preview=False,
    )

    pages = XlsModel().predict(BytesIO(file_bytes))

    assert pages == [
        [
            {"type": BlockType.EQUATION, "content": expected}
            for _name, _mtef, expected in corpus
        ]
    ]


def test_ppt_equation_editor_corpus_stays_bound_to_its_slides() -> None:
    """验证 PPT 的 ExObjRef/persist 绑定逐 slide 恢复完整公式语料。"""

    corpus = formula_corpus()
    file_bytes = build_equation_ppt(
        [mtef for _name, mtef, _expected in corpus],
        preview=False,
    )

    pages = PptModel().predict(BytesIO(file_bytes))

    assert pages == [
        [{"type": BlockType.EQUATION, "content": expected}]
        for _name, _mtef, expected in corpus
    ]


def test_xls_equation_inside_table_is_not_duplicated_as_top_level_block() -> None:
    """验证落在表格坐标内的 XLS 公式进入 cell HTML 且不重复输出。"""

    _name, mtef, expected = formula_corpus()[0]
    pages = XlsModel().predict(
        BytesIO(
            build_equation_xls(
                [(42, mtef)],
                cell_records=label_cell(0, 0, "value"),
            )
        )
    )

    assert len(pages[0]) == 1
    assert pages[0][0]["type"] == BlockType.TABLE
    assert f"<eq>{expected}</eq>" in pages[0][0]["content"]


@pytest.mark.parametrize(
    ("model", "valid_file", "invalid_file"),
    [
        (
            XlsModel(),
            lambda mtef: build_equation_xls([(42, mtef)]),
            lambda: build_equation_xls([(43, b"invalid MTEF")]),
        ),
        (
            PptModel(),
            lambda mtef: build_equation_ppt([mtef]),
            lambda: build_equation_ppt([b"invalid MTEF"]),
        ),
    ],
    ids=["xls", "ppt"],
)
def test_native_equation_wins_and_invalid_native_keeps_preview(
    model: XlsModel | PptModel,
    valid_file: Callable[[bytes], bytes],
    invalid_file: Callable[[], bytes],
) -> None:
    """验证 XLS/PPT 都优先原生公式，坏 MTEF 则保留缓存预览。"""

    _name, mtef, expected = formula_corpus()[1]
    native_pages = model.predict(BytesIO(valid_file(mtef)))
    fallback_pages = model.predict(BytesIO(invalid_file()))

    assert native_pages == [[{"type": BlockType.EQUATION, "content": expected}]]
    assert fallback_pages[0][0]["type"] == BlockType.IMAGE
    assert fallback_pages[0][0]["image_base64"].startswith("data:image/")


def test_ppt_uncompressed_equation_storage_is_supported() -> None:
    """验证 recInstance=0 的未压缩 ExOleObjStg 同样可以恢复公式。"""

    _name, mtef, expected = formula_corpus()[2]

    pages = PptModel().predict(
        BytesIO(build_equation_ppt([mtef], compressed=False, preview=False))
    )

    assert pages == [[{"type": BlockType.EQUATION, "content": expected}]]


@pytest.mark.parametrize("file_suffix", ["xls", "ppt"])
def test_backend_analyze_preserves_native_equations_sync_and_async(
    file_suffix: str,
) -> None:
    """验证原生公式贯穿 XLS/PPT 同步与异步严格 Analyze 契约。"""

    _name, mtef, expected = formula_corpus()[3]
    file_bytes = (
        build_equation_xls([(42, mtef)], preview=False)
        if file_suffix == "xls"
        else build_equation_ppt([mtef], preview=False)
    )
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


def test_ppt_equation_decompression_honors_shared_entry_limit() -> None:
    """验证恶意 ExOleObjStg 声明长度触发稳定资源限制。"""

    _name, mtef, _expected = formula_corpus()[0]
    file_bytes = build_equation_ppt(
        [mtef],
        declared_size=MAX_ENTRY_BYTES + 1,
        preview=False,
    )

    with pytest.raises(LegacyOfficeResourceLimitError, match="max_entry_bytes"):
        PptModel().predict(BytesIO(file_bytes))
