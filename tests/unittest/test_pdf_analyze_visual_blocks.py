from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mineru.backend import analyze
from mineru.types import BlockType


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
    """验证像素框会裁到页面范围，无效框会清理旧载荷且不影响其他块。"""
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
        "image_base64": "stale",
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


def test_visual_block_crop_rejects_page_count_mismatch() -> None:
    """验证 model_list 与渲染页数量不一致时抛出明确异常，避免静默漏页。"""
    with pytest.raises(ValueError, match="Hybrid visual crop page count mismatch"):
        analyze._attach_visual_block_images([[]], [])


def test_replace_inline_formula_delimiters_updates_model_list_in_place() -> None:
    """验证行内公式定界符会在 model JSON 原对象上统一替换为 eq 标签。"""
    model_list = [
        [
            {"type": BlockType.TEXT, "content": "前 \\(a+b\\) 中 \\(c_d\\) 后"},
            {"type": BlockType.TEXT, "content": "已有 <eq>x</eq> 保持不变"},
            {"type": BlockType.TEXT, "content": "未闭合 \\(formula"},
        ],
        [
            {"type": BlockType.TEXT, "content": ""},
            {"type": BlockType.TEXT, "content": None},
            {"type": BlockType.TEXT, "content": ["非字符串"]},
            {"type": BlockType.TEXT, "content": "跨行 \\(a\nb\\)"},
        ],
    ]

    result = analyze._replace_inline_formula_delimiters(model_list)

    assert result is None
    assert model_list[0][0]["content"] == "前 <eq>a+b</eq> 中 <eq>c_d</eq> 后"
    assert model_list[0][1]["content"] == "已有 <eq>x</eq> 保持不变"
    assert model_list[0][2]["content"] == "未闭合 \\(formula"
    assert model_list[1][0]["content"] == ""
    assert model_list[1][1]["content"] is None
    assert model_list[1][2]["content"] == ["非字符串"]
    assert model_list[1][3]["content"] == "跨行 \\(a\nb\\)"


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
    formula_replacer = MagicMock()
    monkeypatch.setattr(analyze, "_OFFICE_MODEL_MAP", model_factories)
    monkeypatch.setattr(analyze, "PDFDocument", pdf_document)
    monkeypatch.setattr(analyze, "HybridLocalModelContextSingleton", hybrid_model_factory)
    monkeypatch.setattr(analyze, "get_processing_window_size", window_size_reader)
    monkeypatch.setattr(analyze, "_build_processing_windows", window_builder)
    monkeypatch.setattr(analyze, "load_images_from_pdf_bytes_range", image_loader)
    monkeypatch.setattr(analyze, "_attach_visual_block_images", visual_image_attacher)
    monkeypatch.setattr(analyze, "_replace_inline_formula_delimiters", formula_replacer)
    monkeypatch.setattr(analyze.time, "perf_counter", fake_perf_counter)

    middle_json, model_list = analyze.doc_analyze(
        b"office-bytes",
        effort="xhigh",
        parse_mode="ocr",
        file_suffix=file_suffix,  # type: ignore[arg-type]
    )

    assert middle_json == []
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
    formula_replacer.assert_not_called()


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

    def fake_replace_formula(_model_list: list[list[dict[str, object]]]) -> None:
        """记录 PDF model_list 公式规范化顺序。"""
        events.append("formula_replace")

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
    monkeypatch.setattr(analyze, "_replace_inline_formula_delimiters", fake_replace_formula)
    monkeypatch.setattr(analyze, "clean_memory", fake_clean_memory)

    middle_json, model_list = analyze.doc_analyze(
        b"empty-pdf",
        effort="high",
        parse_mode="txt",
    )

    assert middle_json == []
    assert model_list == []
    assert events == [
        "hybrid_init",
        "vlm_init",
        "vlm_wrap",
        "timer_20.0",
        "formula_replace",
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

    assert middle_json == []
    assert len(model_list) == expected_page_count
    assert all(isinstance(page, list) for page in model_list)


def test_doc_analyze_flash_returns_complete_model_list_without_middle_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Flash 多窗口仅补充完整 model list，并固定返回空 Middle JSON。"""
    from mineru.model import flash as flash_model

    events: list[str] = []
    source_model_list = [
        [
            {
                "type": BlockType.TEXT,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "content": "第一页 \\(x+y\\)",
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

    original_replace_formula = analyze._replace_inline_formula_delimiters

    def tracked_replace_formula(model_list: list[list[dict[str, object]]]) -> None:
        """记录 PDF 公式规范化顺序并调用真实实现。"""
        events.append("formula_replace")
        original_replace_formula(model_list)  # type: ignore[arg-type]

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
    monkeypatch.setattr(analyze, "_replace_inline_formula_delimiters", tracked_replace_formula)
    monkeypatch.setattr(analyze.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(flash_model, "PdfModel", MagicMock(return_value=fake_pdf_model))

    middle_json, model_list = analyze.doc_analyze(
        b"fake-pdf",
        effort="flash",
        parse_mode="txt",
        page_index_map=[9, 8, 7],
    )

    assert middle_json == []
    assert model_list is source_model_list
    assert requested_ranges == [(0, 1), (2, 2)]
    assert model_list[0][0]["content"] == "第一页 <eq>x+y</eq>"
    assert model_list[2][0]["content"] == "第三页 <eq>z</eq>"
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
        "formula_replace",
        "timer_33.0",
        "document_close",
    ]
    assert not hasattr(analyze, "append_pages")
