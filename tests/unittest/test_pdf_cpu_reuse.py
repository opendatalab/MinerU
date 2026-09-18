"""验证窗口字符/矢量共享和 OCR 图片复制的消费边界。"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from docvortex.document.pdf import PDFDocument, PDFPage, PDFPageTextGeometry, PDFPageVectorGeometry
from mineru.backend.analysis.pdf import tables, window
from mineru.model.ocr import results


def _text_pdf() -> bytes:
    """生成包含三行方向证据和矩形路径的小型表格页。"""
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(200, 200))
    for y in (150, 120, 90):
        canvas.drawString(20, y, "Table text 123")
    canvas.rect(10, 70, 180, 100)
    canvas.save()
    return stream.getvalue()


def test_orientation_and_native_table_share_single_character_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """同页方向与表格恢复共用扩展字符，正文随后可直接消费相同 sidecar。"""
    original = PDFPage.get_chars_with_geometry
    calls = []

    def extract(page: PDFPage) -> PDFPageTextGeometry:
        """记录真实字符提取次数，不替换其结果。"""
        calls.append(page)
        return original(page)

    monkeypatch.setattr(PDFPage, "get_chars_with_geometry", extract)
    monkeypatch.setattr(tables, "recover_table_region", lambda *_args, **_kwargs: None)
    with PDFDocument(_text_pdf()) as document, Image.new("RGB", (200, 200)) as image:
        page = document[0]
        geometries: list[PDFPageTextGeometry | None] = [None]
        vectors: list[PDFPageVectorGeometry | None] = [None]
        layout = {"label": "table", "bbox": [0, 0, 200, 200]}
        items = [{"page_idx": 0, "layout_item": layout}]
        images = [{"img_pil": image, "scale": 1.0}]
        assert tables._resolve_txt_table_orientations(items, [page], images, geometries) == []
        tables._apply_native_txt_table_priority(
            [[{"type": "table", "bbox": [0.0, 0.0, 1.0, 1.0]}]],
            [[layout]],
            [page],
            images,
            effort="high",
            page_text_geometries=geometries,
            page_vector_geometries=vectors,
        )
        assert len(calls) == 1
        assert geometries[0] is not None and vectors[0] is not None
        assert len(vectors[0].path_infos) > 0


def test_geometry_failure_keeps_plain_character_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    """扩展几何失败后仍使用普通字符，不能额外触发视觉方向模型。"""
    with PDFDocument(_text_pdf()) as document, Image.new("RGB", (200, 200)) as image:
        page = document[0]
        monkeypatch.setattr(page, "get_chars_with_geometry", MagicMock(side_effect=ValueError("geometry")))
        # 方向阶段原本不受正文字符数上限限制，复用时也不能新增限制。
        monkeypatch.setattr(page, "get_char_count", lambda: 65536)
        geometries: list[PDFPageTextGeometry | None] = [None]
        item = {"page_idx": 0, "layout_item": {"bbox": [0, 0, 200, 200]}}
        assert tables._resolve_txt_table_orientations([item], [page], [{"img_pil": image, "scale": 1}], geometries) == []
        assert item["layout_item"]["angle"] == 0
        assert geometries == [None]


def test_non_table_page_does_not_eagerly_extract_native_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Layout 后没有表格时，不将正文 CPU 工作移到 VLM 之前。"""
    page = MagicMock()
    document = MagicMock()
    document.__getitem__.return_value = page
    model = MagicMock()
    model.layout_model.batch_predict.return_value = [[]]
    image = Image.new("RGB", (20, 20))
    monkeypatch.setattr(window, "load_images_from_pdf_bytes_range", lambda **_kwargs: [{"img_pil": image, "scale": 1}])
    state = window._prepare_pdf_window(
        b"",
        document,
        window._ProcessingWindow(0, 1, 0, 0),
        page_count=1,
        effort="high",
        parse_mode="txt",
        hybrid_model=model,
    )
    try:
        page.get_chars.assert_not_called()
        page.get_chars_with_geometry.assert_not_called()
        page.get_vector_geometry.assert_not_called()
        assert state.page_text_geometries == [None]
        assert state.page_vector_geometries == [None]
    finally:
        state.close()
    assert state.page_text_geometries == []
    assert state.page_vector_geometries == []


class _CopyCountingImage(np.ndarray):
    """通过真实数组子类观察全图复制次数。"""

    copies = 0

    def copy(self, order: str = "C") -> np.ndarray:
        """记录复制，同时保持真实 NumPy 返回行为。"""
        if self.shape == (40, 40, 3):
            type(self).copies += 1
        return super().copy(order)


@pytest.mark.parametrize("recognize", [False, True])
def test_ocr_image_is_copied_only_when_recognition_needs_it(recognize: bool) -> None:
    """检测回填不复制全图；多个识别框共享一次独立副本，源图保持不变。"""
    image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3).view(_CopyCountingImage)
    before = image.tobytes()
    _CopyCountingImage.copies = 0
    boxes = [[[1, 1], [20, 1], [20, 10], [1, 10]], [[1, 12], [20, 12], [20, 22], [1, 22]]]
    output = results.get_ocr_result_list(boxes, [0, 0, 0, 0, 40, 40, 40, 40], recognize, image)
    assert _CopyCountingImage.copies == int(recognize)
    assert all(("np_img" in entry) is recognize for entry in output)
    assert image.tobytes() == before


def test_already_recognized_boxes_do_not_copy_images() -> None:
    """即使允许识别，已有文本结果也不需要复制和裁图。"""
    image = np.zeros((40, 40, 3), dtype=np.uint8).view(_CopyCountingImage)
    _CopyCountingImage.copies = 0
    output = results.get_ocr_result_list(
        [[[[1, 1], [20, 1], [20, 10], [1, 10]], ["abc", 0.99]]],
        [0, 0, 0, 0, 40, 40, 40, 40],
        True,
        image,
    )
    assert _CopyCountingImage.copies == 0
    assert output[0]["text"] == "abc"
