from __future__ import annotations

import asyncio

import pytest
from _image_mtef_test_utils import (
    apps_mfcc_comments,
    build_gif_with_mtef,
    build_wmf,
)
from _legacy_ppt_test_utils import build_equation_ppt
from _legacy_xls_test_utils import build_equation_xls
from _mtef_test_utils import build_equation_doc
from _mtef_v5_test_utils import v5_formula_corpus
from _office_image_mtef_test_utils import (
    build_image_docx,
    build_image_pptx,
    build_image_xlsx,
)

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.types import BlockType, MiddleJson, ModelJson


def _wmf_formula(mtef: bytes) -> bytes:
    """把 MTEF 包装为跨 chunk AppsMFCC WMF。"""

    return build_wmf(
        apps_mfcc_comments(
            mtef,
            chunk_size=7,
        ),
        placeable=True,
    )


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
