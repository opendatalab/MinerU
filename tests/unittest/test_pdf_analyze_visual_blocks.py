from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mineru.backend import analyze
from mineru.backend.utils.raw_block_types import RAW_CAPTION, RAW_FOOTNOTE, RAW_PHONETIC
from mineru.types import BlockType, MiddleJson


JPEG_DATA_URI_PREFIX = "data:image/jpeg;base64,"
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICE_SAMPLE_DIR = _PROJECT_ROOT / "demo" / "office_docs"


def _build_quadrant_image(width: int = 80, height: int = 40) -> Image.Image:
    """构造四角颜色不同的非对称测试图，便于验证视觉块回正方向。"""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    middle_x = width // 2
    middle_y = height // 2
    draw.rectangle((0, 0, middle_x - 1, middle_y - 1), fill=RED)
    draw.rectangle((middle_x, 0, width - 1, middle_y - 1), fill=GREEN)
    draw.rectangle((0, middle_y, middle_x - 1, height - 1), fill=BLUE)
    draw.rectangle((middle_x, middle_y, width - 1, height - 1), fill=YELLOW)
    return image


def _decode_jpeg_data_uri(data_uri: str) -> Image.Image:
    """解码 JPEG data URI 并返回独立的 RGB 图片对象。"""
    assert data_uri.startswith(JPEG_DATA_URI_PREFIX)
    image_bytes = base64.b64decode(data_uri[len(JPEG_DATA_URI_PREFIX) :])
    assert image_bytes.startswith(b"\xff\xd8")
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.format == "JPEG"
        return image.convert("RGB")


def _sample_quadrant_colors(image: Image.Image) -> tuple[tuple[int, int, int], ...]:
    """读取图片四个象限中心的颜色，避开 JPEG 分区边缘压缩噪声。"""
    width, height = image.size
    sample_points = (
        (width // 4, height // 4),
        (3 * width // 4, height // 4),
        (width // 4, 3 * height // 4),
        (3 * width // 4, 3 * height // 4),
    )
    return tuple(image.getpixel(point) for point in sample_points)


def _assert_colors_close(
    actual_colors: tuple[tuple[int, int, int], ...],
    expected_colors: tuple[tuple[int, int, int], ...],
) -> None:
    """允许少量 JPEG 压缩误差地比较四个象限颜色。"""
    for actual, expected in zip(actual_colors, expected_colors):
        assert max(abs(actual_channel - expected_channel) for actual_channel, expected_channel in zip(actual, expected)) <= 12


@pytest.mark.parametrize(
    ("angle", "expected_size", "expected_colors"),
    [
        (0, (80, 40), (RED, GREEN, BLUE, YELLOW)),
        (90, (40, 80), (GREEN, YELLOW, RED, BLUE)),
        (180, (80, 40), (YELLOW, BLUE, GREEN, RED)),
        (270, (40, 80), (BLUE, RED, YELLOW, GREEN)),
    ],
)
def test_visual_block_crop_rotates_to_upright(
    angle: int,
    expected_size: tuple[int, int],
    expected_colors: tuple[tuple[int, int, int], ...],
) -> None:
    """验证四种合法 angle 会按现有表格语义把视觉块裁图旋转回正。"""
    page_image = _build_quadrant_image()
    block = {
        "type": BlockType.IMAGE,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "angle": angle,
        "content": "keep-content",
    }

    analyze._attach_visual_block_images([[block]], [{"img_pil": page_image}])

    crop_image = _decode_jpeg_data_uri(block["image_base64"])
    try:
        assert crop_image.size == expected_size
        _assert_colors_close(_sample_quadrant_colors(crop_image), expected_colors)
    finally:
        crop_image.close()
        page_image.close()
    assert block["angle"] == angle
    assert block["content"] == "keep-content"


def test_medium_table_task_reuses_visual_block_upright_rotation() -> None:
    """验证 Medium 表格任务继续复用视觉块的 270 度回正语义。"""
    page_image = _build_quadrant_image()
    np_image = np.asarray(page_image).copy()
    table_block = {
        "type": BlockType.TABLE,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "angle": 270,
    }

    table_tasks = analyze._collect_medium_table_tasks(
        [[table_block]],
        [[]],
        [np_image],
    )

    rotated_image = Image.fromarray(table_tasks[0]["table_img"])
    try:
        assert rotated_image.size == (40, 80)
        _assert_colors_close(
            _sample_quadrant_colors(rotated_image),
            (BLUE, RED, YELLOW, GREEN),
        )
    finally:
        rotated_image.close()
        page_image.close()


def test_visual_block_types_receive_jpeg_data_uri_only() -> None:
    """验证 model.json 的四类视觉块写入裁图，非视觉块及其字段保持不变。"""
    page_image = Image.new("RGB", (100, 60), "white")
    visual_blocks = [
        {
            "type": block_type,
            "bbox": [0.1, 0.2, 0.6, 0.7],
            "angle": 0,
            "content": f"content-{block_type}",
            "image_base64": "stale",
        }
        for block_type in (
            BlockType.IMAGE,
            BlockType.CHART,
            BlockType.TABLE,
            BlockType.EQUATION,
        )
    ]
    text_block = {
        "type": BlockType.TEXT,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "content": "text",
        "image_base64": "keep-text-field",
    }

    analyze._attach_visual_block_images(
        [[*visual_blocks, text_block]],
        [{"img_pil": page_image}],
        page_start_index=7,
    )

    try:
        for block in visual_blocks:
            assert block["image_base64"] != "stale"
            crop_image = _decode_jpeg_data_uri(block["image_base64"])
            try:
                assert crop_image.size == (50, 30)
            finally:
                crop_image.close()
            assert block["content"] == f"content-{block['type']}"
            assert block["bbox"] == [0.1, 0.2, 0.6, 0.7]
        assert text_block["image_base64"] == "keep-text-field"
    finally:
        page_image.close()


def test_visual_block_crop_clips_page_boundary_and_skips_invalid_bbox() -> None:
    """验证像素框会裁到页面范围，无效框会跳过载荷写入且不影响其他块。"""
    page_image = Image.new("L", (100, 80), 128)
    clipped_block = {
        "type": BlockType.CHART,
        "bbox": [-10, -5, 30, 20],
        "angle": 0,
    }
    invalid_block = {
        "type": BlockType.TABLE,
        "bbox": [0.5, 0.5, 0.5, 0.8],
        "angle": 0,
    }

    analyze._attach_visual_block_images(
        [[clipped_block, invalid_block]],
        [{"img_pil": page_image}],
    )

    crop_image = _decode_jpeg_data_uri(clipped_block["image_base64"])
    try:
        assert crop_image.size == (30, 20)
    finally:
        crop_image.close()
        page_image.close()
    assert "image_base64" not in invalid_block


def test_image_block_collapses_contained_blocks_before_visual_crop() -> None:
    """验证 image_block 截图前吸收内部小块，并按子块面积应用 0.8 包含阈值。"""
    page_image = Image.new("RGB", (100, 60), "white")
    leading_text = {
        "type": BlockType.TEXT,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "content": "keep-leading-text",
    }
    image_block = {
        "type": "image_block",
        "bbox": [0.0, 0.0, 0.8, 0.8],
        "angle": 0,
        "content": None,
    }
    threshold_image = {
        "type": BlockType.IMAGE,
        "bbox": [0.0, 0.0, 1.0, 0.8],
        "angle": 0,
    }
    contained_caption = {
        "type": BlockType.IMAGE_CAPTION,
        "bbox": [0.1, 0.1, 0.3, 0.2],
        "content": "remove-contained-caption",
    }
    contained_text = {
        "type": BlockType.TEXT,
        "bbox": [0.3, 0.3, 0.4, 0.4],
        "content": "remove-contained-text",
    }
    contained_table = {
        "type": BlockType.TABLE,
        "bbox": [0.4, 0.4, 0.5, 0.5],
    }
    contained_equation = {
        "type": BlockType.EQUATION,
        "bbox": [0.5, 0.5, 0.6, 0.6],
    }
    below_threshold_caption = {
        "type": BlockType.IMAGE_CAPTION,
        "bbox": [0.0, 0.0, 1.0, 0.81],
        "content": "keep-below-threshold-caption",
    }
    invalid_caption = {
        "type": BlockType.IMAGE_CAPTION,
        "bbox": [0.2, 0.2, 0.2, 0.3],
        "content": "keep-invalid-caption",
    }
    external_caption = {
        "type": BlockType.IMAGE_CAPTION,
        "bbox": [0.0, 0.82, 0.8, 0.9],
        "content": "keep-external-caption",
    }
    page_model_list = [
        leading_text,
        image_block,
        threshold_image,
        contained_caption,
        contained_text,
        contained_table,
        contained_equation,
        below_threshold_caption,
        invalid_caption,
        external_caption,
    ]

    analyze._attach_visual_block_images(
        [page_model_list],
        [{"img_pil": page_image}],
    )

    try:
        assert page_model_list == [
            leading_text,
            image_block,
            below_threshold_caption,
            invalid_caption,
            external_caption,
        ]
        assert image_block["type"] == BlockType.IMAGE
        assert all(block.get("type") != "image_block" for block in page_model_list)
        crop_image = _decode_jpeg_data_uri(image_block["image_base64"])
        try:
            assert crop_image.size == (80, 48)
        finally:
            crop_image.close()
    finally:
        page_image.close()


def test_image_block_collapse_keeps_invalid_bbox_blocks() -> None:
    """验证无效容器或子块不会参与面积吸收，但容器类型仍会规范为 image。"""
    invalid_image_block = {
        "type": "image_block",
        "bbox": None,
        "content": None,
    }
    valid_child = {
        "type": BlockType.IMAGE,
        "bbox": [0.1, 0.1, 0.2, 0.2],
    }
    invalid_child = {
        "type": BlockType.IMAGE_CAPTION,
        "bbox": [0.2, 0.2, 0.2, 0.3],
    }
    page_model_list = [invalid_image_block, valid_child, invalid_child]

    analyze._collapse_image_blocks(page_model_list)

    assert page_model_list == [invalid_image_block, valid_child, invalid_child]
    assert invalid_image_block["type"] == BlockType.IMAGE


def test_xhigh_layout_image_supplements_missing_container_before_crop() -> None:
    """验证 xhigh 可用高覆盖率 layout 整图框补容器并复用现有截图折叠流程。"""
    page_image = Image.new("RGB", (1700, 2200), "white")
    page_model_list = [
        {"type": BlockType.IMAGE, "bbox": [0.083, 0.127, 0.278, 0.243]},
        {"type": BlockType.IMAGE, "bbox": [0.281, 0.127, 0.477, 0.243]},
        {"type": BlockType.IMAGE, "bbox": [0.084, 0.246, 0.278, 0.361]},
        {"type": BlockType.IMAGE, "bbox": [0.281, 0.246, 0.477, 0.361]},
        {"type": BlockType.IMAGE, "bbox": [0.084, 0.364, 0.278, 0.48]},
        {"type": BlockType.IMAGE, "bbox": [0.281, 0.364, 0.477, 0.48]},
        {"type": BlockType.IMAGE, "bbox": [0.083, 0.483, 0.279, 0.598]},
        {"type": BlockType.IMAGE, "bbox": [0.281, 0.483, 0.477, 0.598]},
        {
            "type": BlockType.IMAGE_CAPTION,
            "bbox": [0.154, 0.6, 0.214, 0.611],
            "content": "Frame 30",
        },
        {
            "type": BlockType.IMAGE_CAPTION,
            "bbox": [0.35, 0.6, 0.41, 0.611],
            "content": "Frame 90",
        },
        {
            "type": BlockType.IMAGE_CAPTION,
            "bbox": [0.074, 0.616, 0.493, 0.678],
            "content": "Figure 2: keep external caption",
        },
    ]
    layout_blocks_list = [
        [
            {
                "type": BlockType.IMAGE,
                "bbox": [0.082352941, 0.127272727, 0.477647059, 0.611818182],
                "angle": 0,
            }
        ]
    ]

    analyze._supplement_missing_image_block_containers(
        [page_model_list],
        layout_blocks_list,
    )
    assert sum(block.get("type") == "image_block" for block in page_model_list) == 1

    analyze._attach_visual_block_images(
        [page_model_list],
        [{"img_pil": page_image}],
    )

    try:
        assert len(page_model_list) == 2
        image_block, external_caption = page_model_list
        assert image_block["type"] == BlockType.IMAGE
        assert image_block["bbox"] == layout_blocks_list[0][0]["bbox"]
        assert external_caption["content"] == "Figure 2: keep external caption"
        crop_image = _decode_jpeg_data_uri(image_block["image_base64"])
        try:
            assert crop_image.size == (674, 1068)
        finally:
            crop_image.close()
    finally:
        page_image.close()


@pytest.mark.parametrize(
    ("block_types", "block_bboxes", "layout_type", "expected_container_count"),
    [
        (
            (BlockType.IMAGE, BlockType.IMAGE),
            ([0.0, 0.0, 0.46, 1.0], [0.54, 0.0, 1.0, 1.0]),
            BlockType.IMAGE,
            1,
        ),
        (
            (BlockType.IMAGE, BlockType.IMAGE),
            ([0.0, 0.0, 0.45, 1.0], [0.55, 0.0, 1.0, 1.0]),
            BlockType.IMAGE,
            1,
        ),
        (
            (BlockType.CHART, BlockType.CHART),
            ([0.0, 0.0, 0.46, 1.0], [0.54, 0.0, 1.0, 1.0]),
            BlockType.IMAGE,
            1,
        ),
        (
            (BlockType.IMAGE, BlockType.CHART),
            ([0.0, 0.0, 0.46, 1.0], [0.54, 0.0, 1.0, 1.0]),
            BlockType.IMAGE,
            1,
        ),
        (
            (BlockType.IMAGE,),
            ([0.0, 0.0, 0.95, 1.0],),
            BlockType.IMAGE,
            0,
        ),
        (
            (BlockType.IMAGE, BlockType.CHART),
            ([0.0, 0.0, 0.46, 1.0], [0.54, 0.0, 1.0, 1.0]),
            BlockType.CHART,
            0,
        ),
    ],
)
def test_xhigh_layout_image_fallback_requires_visual_count_and_coverage(
    block_types: tuple[str, ...],
    block_bboxes: tuple[list[float], ...],
    layout_type: str,
    expected_container_count: int,
) -> None:
    """验证回退要求至少两个 image/chart，且白名单面积占比大于等于 0.9。"""
    page_model_list = [
        {"type": block_type, "bbox": bbox}
        for block_type, bbox in zip(block_types, block_bboxes)
    ]
    layout_blocks_list = [[{"type": layout_type, "bbox": [0.0, 0.0, 1.0, 1.0], "angle": 0}]]

    analyze._supplement_missing_image_block_containers(
        [page_model_list],
        layout_blocks_list,
    )

    assert sum(block.get("type") == "image_block" for block in page_model_list) == expected_container_count


def test_xhigh_layout_image_fallback_uses_whitelist_for_area_but_absorbs_all_types() -> None:
    """验证白名单只负责面积判定，回退成功后仍吸收框内所有类型。"""
    assert analyze.LOCAL_LAYOUT_IMAGE_BLOCK_AREA_TYPES == {
        BlockType.IMAGE,
        BlockType.CHART,
        BlockType.IMAGE_CAPTION,
        RAW_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
        RAW_FOOTNOTE,
    }
    layout_blocks_list = [[{"type": BlockType.IMAGE, "bbox": [0.0, 0.0, 1.0, 1.0], "angle": 0}]]
    insufficient_page = [
        {"type": BlockType.IMAGE, "bbox": [0.0, 0.0, 0.4, 1.0]},
        {"type": BlockType.CHART, "bbox": [0.4, 0.0, 0.8, 1.0]},
        {"type": BlockType.TEXT, "bbox": [0.8, 0.0, 0.9, 1.0]},
        {"type": BlockType.CHART_CAPTION, "bbox": [0.9, 0.0, 1.0, 1.0]},
    ]

    analyze._supplement_missing_image_block_containers(
        [insufficient_page],
        layout_blocks_list,
    )

    assert all(block.get("type") != "image_block" for block in insufficient_page)

    external_text = {"type": BlockType.TEXT, "bbox": [1.1, 0.0, 1.2, 0.1], "content": "keep-external"}
    qualifying_page = [
        {"type": BlockType.IMAGE, "bbox": [0.0, 0.0, 0.4, 1.0]},
        {"type": BlockType.CHART, "bbox": [0.4, 0.0, 0.8, 1.0]},
        {"type": RAW_CAPTION, "bbox": [0.8, 0.0, 0.9, 1.0]},
        {"type": BlockType.TEXT, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"type": BlockType.TABLE, "bbox": [0.2, 0.2, 0.3, 0.3]},
        {"type": BlockType.EQUATION, "bbox": [0.3, 0.3, 0.4, 0.4]},
        external_text,
    ]

    analyze._supplement_missing_image_block_containers(
        [qualifying_page],
        layout_blocks_list,
    )
    analyze._collapse_image_blocks(qualifying_page)

    assert len(qualifying_page) == 2
    assert qualifying_page[0]["type"] == BlockType.IMAGE
    assert qualifying_page[1] is external_text


def test_xhigh_layout_image_fallback_does_not_duplicate_existing_or_overlapping_containers() -> None:
    """验证已有 VLM 容器和竞争 layout 框不会重复认领相同 image。"""
    existing_image_block = {
        "type": "image_block",
        "bbox": [0.0, 0.0, 0.48, 1.0],
        "angle": 0,
    }
    page_model_list = [
        existing_image_block,
        {"type": BlockType.IMAGE, "bbox": [0.0, 0.0, 0.24, 1.0]},
        {"type": BlockType.IMAGE, "bbox": [0.24, 0.0, 0.48, 1.0]},
        {"type": BlockType.IMAGE, "bbox": [0.52, 0.0, 0.76, 1.0]},
        {"type": BlockType.IMAGE, "bbox": [0.76, 0.0, 1.0, 1.0]},
    ]
    layout_blocks_list = [
        [
            {"type": BlockType.IMAGE, "bbox": [0.0, 0.0, 0.48, 1.0], "angle": 0},
            {"type": BlockType.IMAGE, "bbox": [0.5, 0.0, 1.0, 1.0], "angle": 0},
            {"type": BlockType.IMAGE, "bbox": [0.51, 0.0, 1.0, 1.0], "angle": 0},
        ]
    ]

    analyze._supplement_missing_image_block_containers(
        [page_model_list],
        layout_blocks_list,
    )

    image_blocks = [block for block in page_model_list if block.get("type") == "image_block"]
    assert len(image_blocks) == 2
    assert existing_image_block in image_blocks
    assert sum(block.get("bbox") in ([0.5, 0.0, 1.0, 1.0], [0.51, 0.0, 1.0, 1.0]) for block in image_blocks) == 1


def test_visual_block_crop_rejects_page_count_mismatch() -> None:
    """验证 model_list 与渲染页数量不一致时抛出明确异常，避免静默漏页。"""
    with pytest.raises(ValueError, match="Hybrid visual crop page count mismatch"):
        analyze._attach_visual_block_images([[]], [])


def test_normalize_pdf_model_list_updates_in_place() -> None:
    """验证元数据、公式和无效 PDF 文本块过滤会原地更新 model JSON。"""
    model_list = [
        [
            {
                "type": BlockType.TEXT,
                "content": "前 \\(a+b\\) 中 \\(c_d\\) 后",
                "angle": 0,
                "score": 0.98,
                "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
            },
            {
                "type": BlockType.TEXT,
                "content": "已有 <eq>x</eq> 保持不变",
                "angle": 90,
                "lines": [{"bbox": [0.1, 0.2, 0.9, 0.3]}],
            },
            {
                "type": BlockType.TEXT,
                "content": "未闭合 \\(formula",
                "score": 0.75,
                "lines": [{"bbox": [0.1, 0.3, 0.9, 0.4]}],
            },
        ],
        [
            {
                "type": BlockType.TEXT,
                "content": "",
                "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
            },
            {
                "type": BlockType.TEXT,
                "content": None,
                "angle": 180,
                "score": 0.5,
                "lines": [{"bbox": [0.1, 0.2, 0.9, 0.3]}],
            },
            {
                "type": BlockType.TEXT,
                "content": ["非字符串"],
                "score": 0.25,
                "lines": [{"bbox": [0.1, 0.3, 0.9, 0.4]}],
            },
            {
                "type": BlockType.TEXT,
                "content": "跨行 \\(a\nb\\)",
                "angle": 270,
                "lines": [{"bbox": [0.1, 0.4, 0.9, 0.5]}],
            },
        ],
    ]

    result = analyze._normalize_pdf_model_list(model_list)

    assert result is None
    assert model_list[0][0]["content"] == "前 <eq>a+b</eq> 中 <eq>c_d</eq> 后"
    assert model_list[0][1]["content"] == "已有 <eq>x</eq> 保持不变"
    assert model_list[0][2]["content"] == "未闭合 \\(formula"
    assert [block["content"] for block in model_list[1]] == [
        "跨行 \\(a\nb\\)"
    ]
    assert all(
        "angle" not in block and "score" not in block
        for page_model_list in model_list
        for block in page_model_list
    )


def test_normalize_pdf_model_list_converts_phonetic_and_cleans_equation() -> None:
    """验证 phonetic 转公开类型，同时 equation 保持类型并清理展示分隔符。"""
    model_list = [[
        {
            "type": RAW_PHONETIC,
            "bbox": [0.1, 0.1, 0.9, 0.2],
            "content": "phonetic",
            "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
        },
        {
            "type": BlockType.EQUATION,
            "bbox": [0.1, 0.3, 0.9, 0.4],
            "content": r"\[x+1\]",
        },
    ]]

    analyze._normalize_pdf_model_list(model_list)

    assert model_list[0][0]["type"] == BlockType.TEXT
    assert model_list[0][1] == {
        "type": BlockType.EQUATION,
        "bbox": [0.1, 0.3, 0.9, 0.4],
        "content": "x+1",
    }


@pytest.mark.parametrize("equation_type", [BlockType.EQUATION, "display_formula"])
def test_formula_number_optimizer_recognizes_canonical_and_upstream_equation_types(
    equation_type: str,
) -> None:
    """验证公式编号只需兼容 canonical equation 与上游 display_formula 标签。"""
    optimized = analyze.optimize_hybrid_formula_number_blocks(
        [
            {"type": equation_type, "content": r"\[x+1\]"},
            {"type": BlockType.FORMULA_NUMBER, "content": "(2)"},
        ]
    )

    assert optimized == [
        {
            "type": equation_type,
            "content": r"x+1\tag{2}",
        }
    ]


def test_formula_number_optimizer_does_not_accept_legacy_interline_equation() -> None:
    """验证旧 interline_equation 不再被公式编号合并逻辑视为合法公式块。"""
    optimized = analyze.optimize_hybrid_formula_number_blocks(
        [
            {"type": "interline_equation", "content": "x+1"},
            {"type": BlockType.FORMULA_NUMBER, "content": "(2)"},
        ]
    )

    assert optimized == [
        {"type": "interline_equation", "content": "x+1"},
        {"type": BlockType.TEXT, "content": "(2)"},
    ]


def test_normalize_pdf_model_list_rejects_unclassified_vlm_title() -> None:
    """验证未被 layout 结果分类的 VLM title 以页块位置明确报错。"""
    with pytest.raises(ValueError, match="page_idx=0, block_idx=0"):
        analyze._normalize_pdf_model_list(
            [[{"type": analyze._VLM_UNCLASSIFIED_TITLE_TYPE, "bbox": [0.1, 0.1, 0.9, 0.2], "content": "title"}]]
        )


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    ],
)
@pytest.mark.parametrize(
    "invalid_lines",
    [
        None,
        [],
        [{}],
        [{"bbox": [0.1, 0.2, 0.3]}],
        [{"bbox": [0.1, 0.2, float("nan"), 0.4]}],
        [{"bbox": [-0.1, 0.2, 0.3, 0.4]}],
        [{"bbox": [0.1, 0.2, 1.1, 0.4]}],
        [{"bbox": [0.3, 0.2, 0.3, 0.4]}],
        [{"bbox": [0.1, 0.4, 0.3, 0.4]}],
        [
            {"bbox": [0.1, 0.2, 0.3, 0.4]},
            {"bbox": [0.4, 0.5, 0.4, 0.6]},
        ],
    ],
)
def test_normalize_pdf_model_list_removes_five_text_types_with_invalid_lines(
    block_type: str,
    invalid_lines: object,
) -> None:
    """验证五类文本块只要任一行框非法就按 fail-closed 规则删除。"""

    model_list = [[{"type": block_type, "content": "valid", "lines": invalid_lines}]]

    analyze._normalize_pdf_model_list(model_list)

    assert model_list == [[]]


@pytest.mark.parametrize(
    "block_type",
    [
        BlockType.TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    ],
)
@pytest.mark.parametrize("invalid_content", [None, "", "   ", ["not a string"]])
def test_normalize_pdf_model_list_removes_five_text_types_with_empty_content(
    block_type: str,
    invalid_content: object,
) -> None:
    """验证五类文本块即使行框有效，正文为空或非字符串时仍会删除。"""

    model_list = [
        [
            {
                "type": block_type,
                "content": invalid_content,
                "lines": [{"bbox": [0.1, 0.2, 0.3, 0.4]}],
            }
        ]
    ]

    analyze._normalize_pdf_model_list(model_list)

    assert model_list == [[]]


def test_normalize_pdf_model_list_keeps_other_types_and_valid_text_order() -> None:
    """验证过滤不影响其他类型，并保持合法文本块的相对顺序。"""

    equation = {"type": BlockType.EQUATION, "content": "", "lines": []}
    header = {"type": BlockType.HEADER, "content": "", "lines": []}
    first_text = {
        "type": BlockType.TEXT,
        "content": "first",
        "lines": [{"bbox": [0.1, 0.1, 0.9, 0.2]}],
    }
    invalid_text = {
        "type": RAW_CAPTION,
        "content": "caption",
        "lines": [],
    }
    second_text = {
        "type": RAW_FOOTNOTE,
        "content": "second",
        "lines": [{"bbox": [0.1, 0.8, 0.9, 0.9]}],
    }
    model_list = [[equation, first_text, invalid_text, header, second_text]]

    analyze._normalize_pdf_model_list(model_list)

    assert model_list == [[equation, first_text, header, second_text]]


@pytest.mark.parametrize(
    ("page_count", "elapsed", "expected_cost", "expected_speed"),
    [
        (3, 0.004, "cost=0.004000s", "speed=750.000 page/s"),
        (3, 0.0, "cost=0.000000s", "speed=0.000 page/s"),
    ],
)
def test_log_infer_performance_uses_unrounded_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    page_count: int,
    elapsed: float,
    expected_cost: str,
    expected_speed: str,
) -> None:
    """验证性能日志使用原始耗时计算吞吐，并安全处理零耗时。"""
    debug_log = MagicMock()
    monkeypatch.setattr(analyze.logger, "debug", debug_log)

    analyze._log_infer_performance("docx", page_count, elapsed)

    message = debug_log.call_args.args[0]
    assert "file_suffix=docx" in message
    assert f"pages={page_count}" in message
    assert expected_cost in message
    assert expected_speed in message


@pytest.mark.parametrize("file_suffix", ["docx", "pptx", "xlsx"])
def test_doc_analyze_office_returns_model_list_without_pdf_processing(
    monkeypatch: pytest.MonkeyPatch,
    file_suffix: str,
) -> None:
    """验证三类 Office 文件直接返回模型结果，且不进入任何 PDF 处理阶段。"""
    source_model_list = [[{"type": BlockType.TEXT, "content": "原始 \\(office\\) 内容"}]]
    events: list[str] = []
    model_factories: dict[str, MagicMock] = {}
    selected_model = MagicMock()

    def fake_office_predict(_file_stream: BytesIO) -> list[list[dict[str, object]]]:
        """记录 Office predict 所处的计时区间并返回固定模型结果。"""
        events.append("office_predict")
        return source_model_list

    selected_model.predict.side_effect = fake_office_predict
    for suffix in ("docx", "pptx", "xlsx"):
        model = selected_model if suffix == file_suffix else MagicMock()
        model_factories[suffix] = MagicMock(return_value=model)

    perf_counter_values = iter([10.0, 12.5])

    def fake_perf_counter() -> float:
        """返回稳定计时值并记录计时调用顺序。"""
        value = next(perf_counter_values)
        events.append(f"timer_{value}")
        return value

    pdf_document = MagicMock()
    hybrid_model_factory = MagicMock()
    window_size_reader = MagicMock()
    window_builder = MagicMock()
    image_loader = MagicMock()
    visual_image_attacher = MagicMock()
    model_list_normalizer = MagicMock()
    monkeypatch.setattr(analyze, "_OFFICE_MODEL_MAP", model_factories)
    monkeypatch.setattr(analyze, "PDFDocument", pdf_document)
    monkeypatch.setattr(analyze, "HybridLocalModelContextSingleton", hybrid_model_factory)
    monkeypatch.setattr(analyze, "get_processing_window_size", window_size_reader)
    monkeypatch.setattr(analyze, "_build_processing_windows", window_builder)
    monkeypatch.setattr(analyze, "load_images_from_pdf_bytes_range", image_loader)
    monkeypatch.setattr(analyze, "_attach_visual_block_images", visual_image_attacher)
    monkeypatch.setattr(analyze, "_normalize_pdf_model_list", model_list_normalizer)
    monkeypatch.setattr(analyze.time, "perf_counter", fake_perf_counter)

    middle_json, model_list = analyze.doc_analyze(
        b"office-bytes",
        effort="xhigh",
        parse_mode="ocr",
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )

    assert isinstance(middle_json, MiddleJson)
    assert len(middle_json.pages) == 1
    assert middle_json.file_suffix == file_suffix
    assert middle_json.effort == "flash"
    assert middle_json.parse_mode == "txt"
    assert model_list is source_model_list
    assert model_list[0][0]["content"] == "原始 \\(office\\) 内容"
    for suffix, model_factory in model_factories.items():
        assert model_factory.call_count == (1 if suffix == file_suffix else 0)
    file_stream = selected_model.predict.call_args.args[0]
    assert isinstance(file_stream, BytesIO)
    assert file_stream.getvalue() == b"office-bytes"
    assert not file_stream.closed
    assert events == ["timer_10.0", "office_predict", "timer_12.5"]
    pdf_document.assert_not_called()
    hybrid_model_factory.assert_not_called()
    window_size_reader.assert_not_called()
    window_builder.assert_not_called()
    image_loader.assert_not_called()
    visual_image_attacher.assert_not_called()
    model_list_normalizer.assert_not_called()


def test_doc_analyze_rejects_unsupported_suffix_before_resource_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非法后缀会在创建 PDF 文档或 Office 模型前直接报错。"""
    pdf_document = MagicMock()
    model_factories = {suffix: MagicMock() for suffix in ("docx", "pptx", "xlsx")}
    monkeypatch.setattr(analyze, "PDFDocument", pdf_document)
    monkeypatch.setattr(analyze, "_OFFICE_MODEL_MAP", model_factories)

    with pytest.raises(ValueError, match="Unsupported file suffix: 'PDF'"):
        analyze.doc_analyze(b"unknown", file_suffix="PDF")  # type: ignore[arg-type]

    pdf_document.assert_not_called()
    for model_factory in model_factories.values():
        model_factory.assert_not_called()


def test_pdf_infer_timer_excludes_hybrid_vlm_initialization_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 High PDF 的基础/VLM 初始化和资源清理位于 model_list 计时区间之外。"""
    events: list[str] = []
    fake_document = MagicMock()
    fake_document.page_count = 0

    def fake_document_close() -> None:
        """记录 PDFDocument 关闭顺序。"""
        events.append("document_close")

    fake_document.close.side_effect = fake_document_close
    hybrid_model = MagicMock()
    hybrid_model.device = "cpu"
    hybrid_singleton = MagicMock()

    def fake_hybrid_get_model() -> MagicMock:
        """记录 Hybrid 基础模型初始化顺序。"""
        events.append("hybrid_init")
        return hybrid_model

    hybrid_singleton.get_model.side_effect = fake_hybrid_get_model
    vlm_predictor = MagicMock()
    vlm_singleton = MagicMock()

    def fake_vlm_get_model(**_kwargs: object) -> MagicMock:
        """记录 VLM predictor 初始化顺序。"""
        events.append("vlm_init")
        return vlm_predictor

    vlm_singleton.get_model.side_effect = fake_vlm_get_model

    def fake_enable_serial_execution(predictor: MagicMock, _backend: str) -> MagicMock:
        """记录 VLM predictor 包装顺序并原样返回。"""
        events.append("vlm_wrap")
        return predictor

    perf_counter_values = iter([20.0, 22.0])

    def fake_perf_counter() -> float:
        """返回稳定计时值并记录计时调用顺序。"""
        value = next(perf_counter_values)
        events.append(f"timer_{value}")
        return value

    def fake_normalize_model_list(_model_list: list[list[dict[str, object]]]) -> None:
        """记录 PDF model_list 规范化顺序。"""
        events.append("normalize_model_list")

    def fake_clean_memory(_device: str) -> None:
        """记录设备缓存清理顺序。"""
        events.append("clean_memory")

    monkeypatch.setattr(analyze, "PDFDocument", MagicMock(return_value=fake_document))
    monkeypatch.setattr(analyze, "HybridLocalModelContextSingleton", MagicMock(return_value=hybrid_singleton))
    monkeypatch.setattr(
        analyze,
        "_load_vlm_runtime",
        lambda: {
            "ModelSingleton": MagicMock(return_value=vlm_singleton),
            "_maybe_enable_serial_execution": fake_enable_serial_execution,
        },
    )
    monkeypatch.setattr(analyze, "get_vlm_engine", MagicMock(return_value="transformers"))
    monkeypatch.setattr(analyze.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(analyze, "_normalize_pdf_model_list", fake_normalize_model_list)
    monkeypatch.setattr(analyze, "clean_memory", fake_clean_memory)

    middle_json, model_list = analyze.doc_analyze(
        b"empty-pdf",
        effort="high",
        parse_mode="txt",
    )

    assert isinstance(middle_json, MiddleJson)
    assert middle_json.pages == []
    assert model_list == []
    assert events == [
        "hybrid_init",
        "vlm_init",
        "vlm_wrap",
        "timer_20.0",
        "normalize_model_list",
        "timer_22.0",
        "document_close",
        "clean_memory",
    ]


@pytest.mark.parametrize(
    ("file_suffix", "expected_page_count"),
    [("docx", 3), ("pptx", 6), ("xlsx", 3)],
)
def test_doc_analyze_office_real_samples(file_suffix: str, expected_page_count: int) -> None:
    """验证统一入口可直接分析三类真实 Office 样例并返回完整分页结果。"""
    sample_path = _OFFICE_SAMPLE_DIR / f"{file_suffix}_01.{file_suffix}"

    middle_json, model_list = analyze.doc_analyze(
        sample_path.read_bytes(),
        effort="high",
        parse_mode="ocr",
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )

    assert isinstance(middle_json, MiddleJson)
    assert len(middle_json.pages) == expected_page_count
    assert all(page.page_idx == page_idx for page_idx, page in enumerate(middle_json.pages))
    assert len(model_list) == expected_page_count
    assert all(isinstance(page, list) for page in model_list)
    if file_suffix == "docx":
        model_equations = [
            block
            for page_model_list in model_list
            for block in page_model_list
            if block.get("type") == BlockType.EQUATION
        ]
        middle_equations = [
            block
            for page in middle_json.pages
            for block in page.blocks
            if block.type == BlockType.EQUATION
        ]
        assert model_equations
        assert len(middle_equations) == len(model_equations)
        assert "interline_equation" not in middle_json.to_json()


def test_doc_analyze_flash_real_pdf_returns_typed_middle_json() -> None:
    """验证一页真实 PDF 经 Flash Analyze 后返回严格对象且 raw 结果无废弃字段。"""
    sample_path = _PROJECT_ROOT / "demo" / "pdfs" / "2407.00079v4_origi-10.pdf"

    middle_json, model_list = analyze.doc_analyze(
        sample_path.read_bytes(),
        effort="flash",
        parse_mode="txt",
        file_suffix="pdf",
    )

    assert isinstance(middle_json, MiddleJson)
    assert len(middle_json.pages) == len(model_list) == 1
    assert all(block.bbox is not None for block in middle_json.pages[0].blocks)
    assert all("merge_prev" not in block for page in model_list for block in page)
    assert MiddleJson.model_validate_json(middle_json.to_json()) == middle_json


def test_doc_analyze_flash_demo1_uses_canonical_equation_type() -> None:
    """验证真实 demo1.pdf 的 model list 与 MiddleJson 统一输出 equation。"""
    sample_path = _PROJECT_ROOT / "demo" / "pdfs" / "demo1.pdf"

    middle_json, model_list = analyze.doc_analyze(
        sample_path.read_bytes(),
        effort="flash",
        parse_mode="txt",
        file_suffix="pdf",
    )

    model_equations = [
        block
        for page_model_list in model_list
        for block in page_model_list
        if block.get("type") == BlockType.EQUATION
    ]
    middle_equations = [
        block
        for page in middle_json.pages
        for block in page.blocks
        if block.type == BlockType.EQUATION
    ]

    assert model_equations
    assert len(middle_equations) == len(model_equations)
    assert "interline_equation" not in middle_json.to_json()


def test_doc_analyze_flash_returns_complete_model_list_and_typed_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Flash 多窗口补充完整 raw model list，并返回严格 MiddleJson。"""
    from mineru.model import flash as flash_model

    events: list[str] = []
    source_model_list = [
        [
                {
                    "type": BlockType.TEXT,
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "content": "第一页 \\(x+y\\)",
                    "lines": [{"bbox": [0.0, 0.0, 1.0, 1.0]}],
                }
        ],
        [
            {
                "type": BlockType.IMAGE,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "angle": 0,
            }
        ],
        [
            {
                "type": BlockType.EQUATION,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "angle": 90,
                "content": "第三页 \\(z\\)",
            }
        ],
    ]
    fake_pdf_doc = MagicMock()
    fake_pdf_doc.page_count = len(source_model_list)
    fake_pdf_doc.__getitem__.side_effect = lambda page_idx: MagicMock(page_idx=page_idx)

    def fake_document_close() -> None:
        """记录 Flash PDF 文档关闭顺序。"""
        events.append("document_close")

    fake_pdf_doc.close.side_effect = fake_document_close
    fake_pdf_model = MagicMock()

    def fake_pdf_predict(_document: MagicMock) -> list[list[dict[str, object]]]:
        """记录 PdfModel 推理顺序并返回完整模型结果。"""
        events.append("pdf_predict")
        return source_model_list

    fake_pdf_model.predict.side_effect = fake_pdf_predict
    rendered_images: list[Image.Image] = []
    requested_ranges: list[tuple[int, int]] = []

    def fake_load_images_for_window(
        *,
        pdf_bytes: bytes,
        start_page_id: int,
        end_page_id: int,
        image_type: str,
    ) -> list[dict[str, Image.Image]]:
        """按请求范围生成测试页图，并记录窗口以验证分段与释放行为。"""
        assert pdf_bytes == b"fake-pdf"
        assert image_type == "pil_img"
        events.append("render_window")
        requested_ranges.append((start_page_id, end_page_id))
        window_images = [
            Image.new("RGB", (40, 20), (page_idx * 40, 100, 160))
            for page_idx in range(start_page_id, end_page_id + 1)
        ]
        rendered_images.extend(window_images)
        return [{"img_pil": image} for image in window_images]

    original_attach_visual_block_images = analyze._attach_visual_block_images

    def tracked_attach_visual_block_images(*args: object, **kwargs: object) -> None:
        """记录视觉块补图顺序并调用真实实现。"""
        events.append("attach_visual")
        original_attach_visual_block_images(*args, **kwargs)  # type: ignore[arg-type]

    original_normalize_model_list = analyze._normalize_pdf_model_list

    def tracked_normalize_model_list(model_list: list[list[dict[str, object]]]) -> None:
        """记录 PDF model_list 规范化顺序并调用真实实现。"""
        events.append("normalize_model_list")
        original_normalize_model_list(model_list)  # type: ignore[arg-type]

    perf_counter_values = iter([30.0, 33.0])

    def fake_perf_counter() -> float:
        """返回稳定计时值并记录计时调用顺序。"""
        value = next(perf_counter_values)
        events.append(f"timer_{value}")
        return value

    monkeypatch.setattr(analyze, "PDFDocument", lambda _: fake_pdf_doc)
    monkeypatch.setattr(analyze, "get_processing_window_size", lambda default: 2)
    monkeypatch.setattr(analyze, "load_images_from_pdf_bytes_range", fake_load_images_for_window)
    monkeypatch.setattr(analyze, "_attach_visual_block_images", tracked_attach_visual_block_images)
    monkeypatch.setattr(analyze, "_normalize_pdf_model_list", tracked_normalize_model_list)
    monkeypatch.setattr(analyze.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(flash_model, "PdfModel", MagicMock(return_value=fake_pdf_model))

    middle_json, model_list = analyze.doc_analyze(
        b"fake-pdf",
        effort="flash",
        parse_mode="txt",
        page_index_map=[7, 8, 9],
    )

    assert isinstance(middle_json, MiddleJson)
    assert [page.page_idx for page in middle_json.pages] == [7, 8, 9]
    assert model_list is source_model_list
    assert requested_ranges == [(0, 1), (2, 2)]
    assert model_list[0][0]["content"] == "第一页 <eq>x+y</eq>"
    assert model_list[2][0]["content"] == "第三页 <eq>z</eq>"
    assert model_list[2][0]["type"] == BlockType.EQUATION
    assert middle_json.pages[2].blocks[0].type == BlockType.EQUATION
    assert "interline_equation" not in middle_json.to_json()
    assert "image_base64" not in model_list[0][0]
    for block in (model_list[1][0], model_list[2][0]):
        crop_image = _decode_jpeg_data_uri(block["image_base64"])
        crop_image.close()
    for image in rendered_images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))

    fake_pdf_model.predict.assert_called_once_with(fake_pdf_doc)
    fake_pdf_doc.close.assert_called_once_with()
    assert events == [
        "timer_30.0",
        "pdf_predict",
        "render_window",
        "attach_visual",
        "render_window",
        "attach_visual",
        "normalize_model_list",
        "timer_33.0",
        "document_close",
    ]
    assert not hasattr(analyze, "append_pages")
