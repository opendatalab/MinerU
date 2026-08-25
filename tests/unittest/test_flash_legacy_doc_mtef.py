from __future__ import annotations

import asyncio
from io import BytesIO

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.model.flash import DocModel
from mineru.model.flash.office.legacy.mtef import (
    decode_equation_native,
    decode_equation_object,
    decode_mtef_v3,
)
from mineru.render.contracts import RenderMode
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.types import BlockType, MiddleJson, ModelJson

from _mtef_test_utils import (
    build_equation_doc,
    build_equation_object,
    equation_native,
    formula_corpus,
    mtef_equation,
    mtef_line,
    mtef_template,
    mtef_text,
)

_CAPTURED_EQUATION_NATIVE = bytes.fromhex(
    """
    1c0000000200d4c0410000000000000050a414003cbc14000000000003010103
    010a0112836300030f01000b01128376000011000a030f00000b110102883200
    00000a028612220288340012836d0012836b0002863c00028830000000
    """
)


@pytest.mark.parametrize(
    ("name", "mtef", "expected"),
    formula_corpus(),
    ids=[name for name, _mtef, _expected in formula_corpus()],
)
def test_mtef_v3_corpus_decodes_to_exact_latex(
    name: str,
    mtef: bytes,
    expected: str,
) -> None:
    """验证各类手工 MTEF v3 结构精确转换为预期 LaTeX。"""

    assert name
    assert decode_mtef_v3(mtef) == expected


def test_equation_native_header_and_standalone_ole_object_decode() -> None:
    """验证 28 字节 EQNOLEFILEHDR 和独立 Equation.3 OLE 对象均可读取。"""

    _name, mtef, expected = formula_corpus()[1]
    native = equation_native(mtef)
    equation_object = build_equation_object(mtef)

    assert decode_equation_native(native) == expected
    assert decode_equation_object(equation_object) == expected


def test_captured_equation_editor_native_stream_decodes_independently() -> None:
    """验证独立采集的 Equation.3 stream 恢复上下标、减号和关系式。"""

    assert decode_equation_native(_CAPTURED_EQUATION_NATIVE) == "c_{v}^{2}-4mk<0"


def test_equation_editor_object_doc_integrates_as_native_equation_blocks() -> None:
    """验证 ObjectPool storage id、字段分隔符和 MTEF 形成完整 DOC 原生公式链。"""

    corpus = formula_corpus()
    file_bytes = build_equation_doc(
        [(1000 + index, mtef) for index, (_name, mtef, _expected) in enumerate(corpus)]
    )
    stream = BytesIO(file_bytes)

    pages = DocModel().predict(stream)

    assert not stream.closed
    assert pages == [
        [
            {"type": BlockType.EQUATION, "content": expected}
            for _name, _mtef, expected in corpus
        ]
    ]
    assert all(block.get("type") != BlockType.IMAGE for block in pages[0])


def test_equation_editor_doc_sync_async_middle_json_and_renderers() -> None:
    """验证原生公式贯穿同步/异步 Analyze、严格 MiddleJson、Markdown 和 HTML。"""

    corpus = formula_corpus()
    file_bytes = build_equation_doc(
        [(2000 + index, mtef) for index, (_name, mtef, _expected) in enumerate(corpus)]
    )
    middle, model = doc_analyze(file_bytes, file_suffix="doc")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="doc"))

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert async_model == model
    assert async_middle == middle
    assert [block.type for block in middle.pages[0].blocks] == [BlockType.EQUATION] * len(corpus)
    markdown = render_markdown(middle, mode=RenderMode.FULL)
    html = render_html(middle, mode=RenderMode.FULL)
    for _name, _mtef, expected in corpus:
        assert expected in markdown
        assert expected.replace("&", "&amp;") in html or expected in html


def test_native_equation_wins_over_preview_and_invalid_native_keeps_preview() -> None:
    """验证有效 MTEF 抑制缓存图，坏 MTEF 则安全回退到原图片。"""

    _name, mtef, expected = formula_corpus()[0]
    native_pages = DocModel().predict(
        BytesIO(
            build_equation_doc(
                [(42, mtef)],
                preview_storage_ids={42},
            )
        )
    )
    fallback_pages = DocModel().predict(
        BytesIO(
            build_equation_doc(
                [(43, b"invalid MTEF")],
                preview_storage_ids={43},
            )
        )
    )

    assert native_pages == [[{"type": BlockType.EQUATION, "content": expected}]]
    assert fallback_pages[0][0]["type"] == BlockType.IMAGE
    assert fallback_pages[0][0]["image_base64"].startswith("data:image/")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x02\x01\x01\x03\x00\x00",
        mtef_equation(mtef_template(49, mtef_line(mtef_text("x")))),
        bytes([3, 1, 1, 3, 0, 5, 0, 0, 0, 255, 255]),
        bytes([3, 1, 1, 3, 0, 8, 131, 0x41]),
    ],
)
def test_invalid_or_unsupported_mtef_fails_closed(payload: bytes) -> None:
    """验证截断、未知 template、异常矩阵和坏 FONT 不制造错误 LaTeX。"""

    assert decode_mtef_v3(payload) is None


def test_invalid_equation_native_header_fails_closed() -> None:
    """验证错误 header size、版本和对象长度均被拒绝。"""

    _name, mtef, _expected = formula_corpus()[0]
    valid = bytearray(equation_native(mtef))
    cases: list[bytes] = []
    broken = bytearray(valid)
    broken[0:2] = (27).to_bytes(2, "little")
    cases.append(bytes(broken))
    broken = bytearray(valid)
    broken[2:6] = (0x0001_0000).to_bytes(4, "little")
    cases.append(bytes(broken))
    broken = bytearray(valid)
    broken[8:12] = (len(mtef) + 100).to_bytes(4, "little")
    cases.append(bytes(broken))

    assert all(decode_equation_native(case) is None for case in cases)
