from __future__ import annotations

import ast
import inspect

import pytest

from mineru.backend.flash import pdf_extractor
from mineru.backend.flash.native_pdf import (
    auxiliary_text,
    formulas,
    geometry,
    graphics,
    index_blocks,
    line_layout,
    line_merging,
    models,
    native_text,
    pipeline,
    tables,
    text_blocks,
    titles,
)
from mineru.utils.pdf_document import PDFImageInfo


def _image_info(
    fingerprint: str | None,
    bbox: tuple[float, float, float, float],
) -> PDFImageInfo:
    """构造跨页图片水印规则使用的轻量图片信息。"""

    return PDFImageInfo(bbox=bbox, fingerprint=fingerprint)


def test_repeated_large_image_requires_three_distinct_pages() -> None:
    """验证同页重复只计一次，面积恰好 8% 且跨三页时才命中水印。"""

    page_sizes = [(100.0, 100.0)] * 4
    page_image_infos = [
        [
            _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
            _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
            _image_info("two-pages", (0.0, 0.0, 50.0, 20.0)),
        ],
        [_image_info("watermark", (10.0, 10.0, 50.0, 30.0))],
        [_image_info("watermark", (20.0, 20.0, 60.0, 40.0))],
        [_image_info("two-pages", (0.0, 0.0, 50.0, 20.0))],
    ]

    fingerprints = pipeline._detect_repeated_raster_watermark_fingerprints(
        page_image_infos,
        page_sizes,
    )

    assert fingerprints == {"watermark"}


def test_repeated_watermark_filter_keeps_small_or_unfingerprinted_images() -> None:
    """验证已命中指纹也只删除大图，小图和指纹读取失败的图片继续进入现有流程。"""

    image_infos = [
        _image_info("watermark", (0.0, 0.0, 40.0, 20.0)),
        _image_info("watermark", (0.0, 0.0, 39.9, 20.0)),
        _image_info(None, (0.0, 0.0, 80.0, 80.0)),
        _image_info("ordinary", (0.0, 0.0, 80.0, 80.0)),
    ]

    filtered = pipeline._filter_repeated_raster_watermark_bboxes(
        image_infos,
        (100.0, 100.0),
        {"watermark"},
    )

    assert filtered == [
        (0.0, 0.0, 39.9, 20.0),
        (0.0, 0.0, 80.0, 80.0),
        (0.0, 0.0, 80.0, 80.0),
    ]


def test_flash_extractor_has_no_local_ocr_runtime_logic() -> None:
    """守卫 Flash extractor 不再引入或实现本地 OCR 运行时逻辑。"""

    source = "\n".join(
        inspect.getsource(module)
        for module in (
            pdf_extractor,
            auxiliary_text,
            formulas,
            geometry,
            graphics,
            index_blocks,
            line_layout,
            line_merging,
            models,
            native_text,
            pipeline,
            tables,
            text_blocks,
            titles,
        )
    )
    forbidden_tokens = (
        "import numpy",
        "import cv2",
        "project_ocr_table_text",
        "_load_ocr_runtime",
        "_run_full_page_ocr",
        "_project_ocr_candidate",
        "_recognize_rotated_ocr_table",
        "AtomModelSingleton",
        "run_ocr_inference",
        "get_processing_window_size",
        "bgr_image",
        "pixel_quad",
        "ocr_model",
    )

    assert not [token for token in forbidden_tokens if token in source]


def test_flash_extractor_only_exports_supported_entrypoints() -> None:
    """验证旧模块只声明两个稳定公共入口，不再承载内部实现。"""

    assert pdf_extractor.__all__ == ["doc_analyze", "extract_pages_text"]
    assert not hasattr(pdf_extractor, "_analyze_native_document")


def test_native_pdf_domain_modules_do_not_import_each_other() -> None:
    """守卫领域处理器只依赖共享层，跨领域组合统一留在 pipeline。"""

    domain_modules = (
        auxiliary_text,
        formulas,
        graphics,
        index_blocks,
        tables,
        text_blocks,
        titles,
    )
    domain_names = {module.__name__.rsplit(".", 1)[-1] for module in domain_modules}
    for module in domain_modules:
        tree = ast.parse(inspect.getsource(module))
        relative_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
        }
        assert not relative_imports & domain_names, module.__name__


@pytest.mark.parametrize(
    "block_type",
    [
        "doc_title",
        "paragraph_title",
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "aside_text",
        "index",
        "equation",
    ],
)
def test_output_normalization_preserves_new_flash_types(block_type: str) -> None:
    """验证 Flash 文本语义类型和公式类型不会在归一化阶段退回 text。"""

    block = pipeline._normalize_output_block(
        {"type": block_type, "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": "value"},
        (100.0, 100.0),
    )

    assert block is not None
    assert block["type"] == block_type


def test_output_normalization_keeps_empty_equation_but_drops_empty_text() -> None:
    """验证纯矢量公式可保留空 content，普通空文本仍不会进入 model_list。"""

    equation = pipeline._normalize_output_block(
        {"type": "equation", "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": ""},
        (100.0, 100.0),
    )
    text = pipeline._normalize_output_block(
        {"type": "text", "bbox": (10.0, 20.0, 40.0, 50.0), "angle": 0, "content": ""},
        (100.0, 100.0),
    )

    assert equation == {
        "type": "equation",
        "bbox": [0.1, 0.2, 0.4, 0.5],
        "angle": 0,
        "content": "",
    }
    assert text is None
