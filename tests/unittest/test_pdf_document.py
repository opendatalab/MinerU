from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from pdftext.schema import Bbox
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from mineru.utils import pdf_document


def test_pdf_page_exposes_path_infos_without_raw_pdfium_access() -> None:
    """验证 PDFPage 只读代理当前页 Path 摘要并保留页索引。"""

    document = MagicMock()
    expected = [MagicMock()]
    document.get_page_path_infos.return_value = expected

    assert pdf_document.PDFPage(document, 3).get_path_infos() is expected
    document.get_page_path_infos.assert_called_once_with(3)


def _build_drawing_pdf() -> bytes:
    """构造包含描边、填充细矩形、相邻线段、Form 矩阵和斜线的测试 PDF。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=(100, 200))
    canvas.setLineWidth(1)
    canvas.line(10, 180, 90, 180)
    canvas.line(10, 160, 50, 160)
    canvas.line(51, 160, 90, 160)
    canvas.rect(10, 139.5, 80, 0.5, stroke=0, fill=1)

    canvas.beginForm("NestedLine", 0, 0, 20, 10)
    canvas.setLineWidth(1)
    canvas.line(0, 0, 20, 0)
    canvas.endForm()
    canvas.saveState()
    canvas.translate(30, 100)
    canvas.scale(2, 1)
    canvas.doForm("NestedLine")
    canvas.restoreState()

    # alpha 为 0 的描边不可见，公共接口应过滤。
    canvas.saveState()
    canvas.setStrokeAlpha(0)
    canvas.line(10, 120, 90, 120)
    canvas.restoreState()

    # 斜线不属于表格横竖线，公共接口应过滤。
    canvas.line(10, 10, 90, 50)
    # 闭合贝塞尔用于验证 Path 信息保留控制点形成的几何范围。
    curve = canvas.beginPath()
    curve.moveTo(10, 80)
    curve.curveTo(20, 95, 30, 95, 40, 80)
    curve.lineTo(10, 80)
    curve.close()
    canvas.drawPath(curve, stroke=0, fill=1)
    canvas.save()
    return output.getvalue()


def _build_rotated_cropped_drawing_pdf() -> bytes:
    """构造带 CropBox 与 90 度页面旋转的测试 PDF。"""
    source = BytesIO()
    canvas = Canvas(source, pagesize=(100, 200))
    canvas.setLineWidth(2)
    canvas.line(10, 20, 90, 20)
    canvas.save()

    reader = PdfReader(BytesIO(source.getvalue()))
    page = reader.pages[0]
    page.rotate(90)
    page.cropbox.lower_left = (5, 10)
    page.cropbox.upper_right = (95, 190)
    writer = PdfWriter()
    writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _build_colored_path_pdf() -> bytes:
    """构造可见浅色填充与透明填充 Path，验证 RGBA 元数据。"""

    output = BytesIO()
    canvas = Canvas(output, pagesize=(100, 100))
    canvas.setFillColorRGB(242 / 255, 255 / 255, 242 / 255)
    canvas.rect(10, 60, 80, 20, stroke=0, fill=1)
    canvas.saveState()
    canvas.setFillAlpha(0)
    canvas.rect(10, 20, 80, 20, stroke=0, fill=1)
    canvas.restoreState()
    canvas.save()
    return output.getvalue()


def _build_rotated_cropped_image_pdf() -> bytes:
    """构造普通、嵌套 Form、部分页外和完全页外点阵图，并应用 CropBox 与旋转。"""
    image = Image.new("RGB", (3, 4), "red")
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    image_reader = ImageReader(image_buffer)

    source = BytesIO()
    canvas = Canvas(source, pagesize=(100, 200))
    canvas.drawImage(image_reader, 10, 20, width=30, height=40)
    canvas.drawImage(image_reader, -10, 170, width=30, height=40)
    canvas.drawImage(image_reader, -30, -30, width=5, height=5)
    canvas.beginForm("NestedImage", 0, 0, 20, 20)
    canvas.drawImage(image_reader, 1, 2, width=3, height=4)
    canvas.endForm()
    canvas.saveState()
    canvas.translate(50, 80)
    canvas.scale(2, 3)
    canvas.doForm("NestedImage")
    canvas.restoreState()
    canvas.save()

    reader = PdfReader(BytesIO(source.getvalue()))
    page = reader.pages[0]
    page.rotate(90)
    page.cropbox.lower_left = (5, 10)
    page.cropbox.upper_right = (95, 190)
    writer = PdfWriter()
    writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _build_rotated_cropped_signature_pdf() -> bytes:
    """构造带可见签名和各类无效注释的 CropBox 旋转测试 PDF。"""

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=200)
    page.rotate(90)
    page.cropbox.lower_left = (5, 10)
    page.cropbox.upper_right = (95, 190)

    def add_appearance(width: float, height: float) -> object:
        """为测试签名创建最小正常 Form 外观流。"""

        appearance = DecodedStreamObject()
        appearance.set_data(b"q 1 0 0 rg 0 0 1 1 re f Q")
        appearance.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): ArrayObject(
                    [
                        FloatObject(0),
                        FloatObject(0),
                        FloatObject(width),
                        FloatObject(height),
                    ]
                ),
                NameObject("/Resources"): DictionaryObject(),
            }
        )
        return writer._add_object(appearance)

    annotations = ArrayObject()
    form_fields = ArrayObject()

    def add_widget(
        rect: tuple[float, float, float, float],
        *,
        flags: int = 4,
        field_type: str = "/Sig",
        subtype: str = "/Widget",
        with_appearance: bool = True,
        inherited_field_type: bool = False,
    ) -> None:
        """追加一个可配置的测试 Widget，覆盖可见性和结构过滤分支。"""

        annotation = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject(subtype),
                NameObject("/Rect"): ArrayObject([FloatObject(value) for value in rect]),
                NameObject("/F"): NumberObject(flags),
            }
        )
        parent_field = None
        parent_reference = None
        if inherited_field_type:
            parent_field = DictionaryObject(
                {
                    NameObject("/FT"): NameObject(field_type),
                    NameObject("/T"): TextStringObject("InheritedSignature"),
                }
            )
            parent_reference = writer._add_object(parent_field)
            annotation[NameObject("/Parent")] = parent_reference
        else:
            annotation[NameObject("/FT")] = NameObject(field_type)
        if with_appearance:
            annotation[NameObject("/AP")] = DictionaryObject(
                {
                    NameObject("/N"): add_appearance(
                        abs(rect[2] - rect[0]),
                        abs(rect[3] - rect[1]),
                    )
                }
            )
        annotation_reference = writer._add_object(annotation)
        annotations.append(annotation_reference)
        if parent_field is not None and parent_reference is not None:
            parent_field[NameObject("/Kids")] = ArrayObject([annotation_reference])
            form_fields.append(parent_reference)
        elif subtype == "/Widget":
            form_fields.append(annotation_reference)

    add_widget((10, 20, 40, 60), inherited_field_type=True)
    add_widget((-10, 170, 20, 210))
    add_widget((20, 80, 40, 100), flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_HIDDEN)
    add_widget((20, 80, 40, 100), flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE)
    add_widget((20, 80, 40, 100), flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_NOVIEW)
    add_widget((20, 80, 40, 100), with_appearance=False)
    add_widget((20, 80, 40, 100), field_type="/Btn")
    add_widget((20, 80, 40, 100), subtype="/Text")
    page[NameObject("/Annots")] = annotations
    writer._root_object[NameObject("/AcroForm")] = DictionaryObject(
        {NameObject("/Fields"): form_fields}
    )

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _build_rotated_cropped_link_pdf() -> bytes:
    """构造带 QuadPoints、Rect 回退和无效动作的旋转 Link 注解测试 PDF。"""

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=200)
    page.rotate(90)
    page.cropbox.lower_left = (5, 10)
    page.cropbox.upper_right = (95, 190)
    annotations = ArrayObject()

    def add_uri_link(
        target: str,
        rect: tuple[float, float, float, float],
        *,
        quad_points: tuple[float, ...] | None = None,
        flags: int = 0,
    ) -> None:
        """追加一个可配置 URI Link，用于覆盖目标校验和区域读取分支。"""

        annotation = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Link"),
                NameObject("/Rect"): ArrayObject(
                    [FloatObject(value) for value in rect]
                ),
                NameObject("/A"): DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/URI"),
                        NameObject("/URI"): TextStringObject(target),
                    }
                ),
            }
        )
        if quad_points is not None:
            annotation[NameObject("/QuadPoints")] = ArrayObject(
                [FloatObject(value) for value in quad_points]
            )
        if flags:
            annotation[NameObject("/F")] = NumberObject(flags)
        annotations.append(writer._add_object(annotation))

    add_uri_link(
        "https://example.com/a?x=1&y=2",
        (10, 20, 60, 40),
        quad_points=(20, 35, 40, 35, 20, 25, 40, 25),
    )
    add_uri_link("mailto:user@example.com", (10, 50, 40, 60))
    add_uri_link("tel:+123456", (10, 70, 40, 80))
    add_uri_link("javascript:alert(1)", (10, 90, 40, 100))
    add_uri_link("relative/path", (10, 110, 40, 120))
    add_uri_link(
        "https://hidden.example.com",
        (10, 130, 40, 140),
        flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_HIDDEN,
    )
    add_uri_link(
        "https://invisible.example.com",
        (45, 130, 75, 140),
        flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE,
    )
    add_uri_link(
        "https://noview.example.com",
        (10, 145, 40, 155),
        flags=pdf_document.pdfium_c.FPDF_ANNOT_FLAG_NOVIEW,
    )
    annotations.append(
        writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): ArrayObject(
                        [FloatObject(10), FloatObject(20)]
                    ),
                    NameObject("/A"): DictionaryObject(
                        {
                            NameObject("/S"): NameObject("/URI"),
                            NameObject("/URI"): TextStringObject(
                                "https://broken.example.com"
                            ),
                        }
                    ),
                }
            )
        )
    )
    annotations.append(
        writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): ArrayObject(
                        [
                            FloatObject(10),
                            FloatObject(165),
                            FloatObject(40),
                            FloatObject(175),
                        ]
                    ),
                    NameObject("/Dest"): TextStringObject("missing-destination"),
                }
            )
        )
    )
    page[NameObject("/Annots")] = annotations

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class _TrackingLock:
    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self) -> None:
        self.depth += 1

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.depth -= 1


def test_pdf_document_methods_keep_page_access_inside_pdfium_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = _TrackingLock()
    monkeypatch.setattr(pdf_document, "_pdfium_lock", lock)

    events: list[str] = []

    class _FakeBitmap:
        def to_pil(self) -> Image.Image:
            events.append(f"bitmap.to_pil:{lock.depth}")
            return Image.new("RGB", (2, 2), "white")

        def close(self) -> None:
            events.append(f"bitmap.close:{lock.depth}")

    class _FakePage:
        def get_bbox(self) -> tuple[float, float, float, float]:
            events.append(f"page.get_bbox:{lock.depth}")
            return (0.0, 10.0, 20.0, 0.0)

        def get_size(self) -> tuple[int, int]:
            events.append(f"page.get_size:{lock.depth}")
            return 20, 10

        def get_textpage(self) -> "_FakeTextPage":
            events.append(f"page.get_textpage:{lock.depth}")
            return _FakeTextPage()

        def get_rotation(self) -> int:
            events.append(f"page.get_rotation:{lock.depth}")
            return 0

        def render(self, *, scale: float) -> _FakeBitmap:
            events.append(f"page.render:{lock.depth}:{scale}")
            return _FakeBitmap()

    class _FakeTextPage:
        def close(self) -> None:
            events.append(f"textpage.close:{lock.depth}")

    class _FakeDoc:
        def __init__(self, pdf_bytes: bytes) -> None:
            events.append(f"doc.open:{lock.depth}:{pdf_bytes!r}")
            self.page = _FakePage()
            self.raw = object()

        def __len__(self) -> int:
            events.append(f"doc.__len__:{lock.depth}")
            return 1

        def __getitem__(self, page_idx: int) -> _FakePage:
            events.append(f"doc.__getitem__:{lock.depth}:{page_idx}")
            return self.page

        def close(self) -> None:
            events.append(f"doc.close:{lock.depth}")

    def fake_get_chars(textpage: _FakeTextPage, page_bbox: list[float], page_rotation: int) -> list[dict[str, Any]]:
        """记录文本抽取时的锁深度，避免依赖旧模块级 get_page_chars 钩子。"""
        events.append(f"get_chars:{lock.depth}:{page_bbox}:{page_rotation}")
        return [
            {
                "char": "A",
                "bbox": Bbox([0.0, 0.0, 1.0, 1.0]),
                "rotation": 0,
                "font": {"name": "Helvetica", "flags": 0, "size": 10, "weight": 400},
                "char_idx": 0,
            }
        ]

    def fake_extract_page_drawing_lines(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[pdf_document.PDFDrawingLine]:
        """记录绘图对象遍历时仍由 PDFDocument 持有 PDFium 锁。"""
        events.append(f"drawing_lines:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_image_bboxes(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[tuple[float, float, float, float]]:
        """记录点阵图遍历时仍由 PDFDocument 持有 PDFium 锁。"""
        events.append(f"image_bboxes:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_image_infos(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[pdf_document.PDFImageInfo]:
        """记录图片指纹元数据遍历时仍由 PDFDocument 持有 PDFium 锁。"""

        events.append(f"image_infos:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_path_infos(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[pdf_document.PDFPathInfo]:
        """记录完整 Path 几何遍历时仍由 PDFDocument 持有 PDFium 锁。"""
        events.append(f"path_infos:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_form_bboxes(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[tuple[float, float, float, float]]:
        """记录 Form 遍历时仍由 PDFDocument 持有 PDFium 锁。"""
        events.append(f"form_bboxes:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_signature_bboxes(
        page: _FakePage,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
        *,
        form_handle: object | None = None,
    ) -> list[tuple[float, float, float, float]]:
        """记录签名注释遍历时仍由 PDFDocument 持有 PDFium 锁。"""

        assert form_handle is None
        events.append(f"signature_bboxes:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    def fake_extract_page_link_annotations(
        page: _FakePage,
        raw_doc: object,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> list[pdf_document.PDFLinkAnnotation]:
        """记录 Link 注解遍历时页面和文档句柄仍处于 PDFium 锁内。"""

        assert raw_doc is doc._pdf_doc.raw
        events.append(f"link_annotations:{lock.depth}:{page_bbox}:{page_rotation}")
        return []

    monkeypatch.setattr(pdf_document.pdfium, "PdfDocument", _FakeDoc)
    monkeypatch.setattr(pdf_document, "get_chars", fake_get_chars, raising=False)
    monkeypatch.setattr(pdf_document, "pdftext_get_chars", fake_get_chars, raising=False)
    monkeypatch.setattr(pdf_document, "_extract_page_drawing_lines", fake_extract_page_drawing_lines)
    monkeypatch.setattr(pdf_document, "_extract_page_path_infos", fake_extract_page_path_infos)
    monkeypatch.setattr(pdf_document, "_extract_page_image_bboxes", fake_extract_page_image_bboxes)
    monkeypatch.setattr(pdf_document, "_extract_page_image_infos", fake_extract_page_image_infos)
    monkeypatch.setattr(pdf_document, "_extract_page_form_bboxes", fake_extract_page_form_bboxes)
    monkeypatch.setattr(pdf_document, "_extract_page_signature_bboxes", fake_extract_page_signature_bboxes)
    monkeypatch.setattr(pdf_document, "_extract_page_link_annotations", fake_extract_page_link_annotations)

    doc = pdf_document.PDFDocument(b"%PDF")

    assert doc.page_size(0) == (20.0, 10.0)
    image = doc.render_page(0, scale=3)
    assert image.pil_image.size == (2, 2)
    assert image.scale == 3
    assert doc.get_page_chars(0)[0]["char"] == "A"
    assert doc.get_page_drawing_lines(0) == []
    assert doc.get_page_path_infos(0) == []
    assert doc.get_page_image_bboxes(0) == []
    assert doc.get_page_image_infos(0) == []
    assert doc.get_page_form_bboxes(0) == []
    assert doc.get_page_signature_bboxes(0) == []
    assert doc.get_page_link_annotations(0) == []

    assert any(event.startswith("doc.open:") and not event.startswith("doc.open:0:") for event in events)
    assert any(event.startswith("doc.__getitem__:") and not event.startswith("doc.__getitem__:0:") for event in events)
    assert "page.get_bbox:1" in events
    assert "page.get_size:1" in events
    assert "page.render:1:3" in events
    assert "bitmap.to_pil:1" in events
    assert "bitmap.close:1" in events
    assert "page.get_textpage:1" in events
    assert "get_chars:1:[0.0, 10.0, 20.0, 0.0]:0" in events
    assert "textpage.close:1" in events
    assert "drawing_lines:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "path_infos:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "image_bboxes:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "image_infos:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "form_bboxes:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "signature_bboxes:1:(0.0, 0.0, 20.0, 10.0):0" in events
    assert "link_annotations:1:(0.0, 0.0, 20.0, 10.0):0" in events


def test_pdf_document_does_not_expose_legacy_compat_hooks() -> None:
    assert not hasattr(pdf_document, "pdf_page_to_image")
    assert not hasattr(pdf_document, "open_pdfium_document")
    assert not hasattr(pdf_document, "get_text_quality_signal_pdfium")
    assert not hasattr(pdf_document.PDFDocument, "get_text_quality")
    assert pdf_document.PDFDocument._pdf_doc.fset is None


def test_restore_pdfium_surrogate_pairs_recovers_supplementary_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证合法 surrogate pair 可恢复，真实替换符与孤立 surrogate 不被误判。"""

    raw_codes = [ord("A"), 0xD835, 0xDF03, 0xFFFD, 0xD835, ord("B")]

    class _FakeTextPage:
        raw = object()

        def count_chars(self) -> int:
            return len(raw_codes)

    monkeypatch.setattr(
        pdf_document.pdfium_c,
        "FPDFText_GetUnicode",
        lambda _textpage, char_idx: raw_codes[char_idx],
    )
    chars = [
        {
            "char": text,
            "bbox": Bbox([float(char_idx), 0.0, float(char_idx + 1), 1.0]),
            "rotation": 0,
            "font": {},
            "char_idx": char_idx,
        }
        for char_idx, text in enumerate(["A", "\uFFFD", "\uFFFD", "\uFFFD", "\ud835", "B"])
    ]

    restored = pdf_document._restore_pdfium_surrogate_pairs(chars, _FakeTextPage())

    assert [char["char"] for char in restored] == ["A", "𝜃", "\uFFFD", "\uFFFD", "B"]
    assert [char["char_idx"] for char in restored] == [0, 1, 3, 4, 5]


def test_get_page_drawing_lines_extracts_forms_filled_rectangles_and_merges_segments() -> None:
    """验证绘图线接口支持 Form、细长填充矩形、共线合并并过滤斜线。"""
    with pdf_document.PDFDocument(_build_drawing_pdf()) as doc:
        lines = doc.get_page_drawing_lines(0)

    assert [line.orientation for line in lines] == ["horizontal"] * 4
    assert [line.start for line in lines] == pytest.approx(
        [
            (10.0, 20.0),
            (10.0, 40.0),
            (10.0, 60.25),
            (30.0, 100.0),
        ]
    )
    assert [line.end for line in lines] == pytest.approx(
        [
            (90.0, 20.0),
            (90.0, 40.0),
            (90.0, 60.25),
            (70.0, 100.0),
        ]
    )
    assert [line.width for line in lines] == pytest.approx([1.0, 1.0, 0.5, 1.0])
    assert lines[2].bbox == pytest.approx((10.0, 60.0, 90.0, 60.5))


def test_get_page_path_infos_preserves_bezier_visibility_depth_and_source_order() -> None:
    """验证完整 Path 接口保留贝塞尔 bbox、绘制模式、Form 深度和稳定源序号。"""

    with pdf_document.PDFDocument(_build_drawing_pdf()) as doc:
        path_infos = doc.get_page_path_infos(0)

    assert len(path_infos) == 7
    assert [item.source_index for item in path_infos] == [0, 1, 2, 3, 4, 6, 7]
    assert path_infos[0].bbox == pytest.approx((9.5, 19.5, 90.5, 20.5))
    assert not path_infos[0].fill_visible and path_infos[0].stroke_visible
    assert path_infos[3].bbox == pytest.approx((10.0, 60.0, 90.0, 60.5))
    assert path_infos[3].fill_visible and not path_infos[3].stroke_visible
    assert path_infos[3].fill_rgba == (0, 0, 0, 255)
    nested_path = next(item for item in path_infos if item.form_depth == 1)
    assert nested_path.bbox == pytest.approx((29.0, 99.0, 71.0, 101.0))
    bezier_path = path_infos[-1]
    assert bezier_path.segment_count == 5
    assert bezier_path.bbox == pytest.approx((10.0, 105.0, 40.0, 120.0))


def test_get_page_path_infos_exposes_fill_rgba_and_transparency() -> None:
    """验证可见 Path 保留填充 RGBA，透明填充不伪装成可见背景。"""

    with pdf_document.PDFDocument(_build_colored_path_pdf()) as doc:
        path_infos = doc.get_page_path_infos(0)

    visible = next(item for item in path_infos if item.fill_visible)
    transparent = next(item for item in path_infos if not item.fill_visible)
    assert visible.fill_rgba == (242, 255, 242, 255)
    assert transparent.fill_rgba is None


def test_raw_object_rgba_failure_returns_none() -> None:
    """验证旧 PDFium 或损坏对象读取颜色失败时返回 None。"""

    def broken_getter(*_args: Any) -> bool:
        """模拟底层颜色接口异常。"""

        raise RuntimeError("broken color")

    assert pdf_document._get_raw_object_rgba(object(), broken_getter) is None


def test_get_page_drawing_lines_applies_crop_box_and_page_rotation() -> None:
    """验证页面 CropBox 与 90 度旋转被转换为左上原点坐标。"""
    with pdf_document.PDFDocument(_build_rotated_cropped_drawing_pdf()) as doc:
        page_size = doc.page_size(0)
        page_rotation = doc.page_rotation(0)
        page_object_rotation = doc[0].rotation
        lines = doc.get_page_drawing_lines(0)

    assert page_size == pytest.approx((180.0, 90.0))
    assert page_rotation == 90
    assert page_object_rotation == 90
    assert len(lines) == 1
    line = lines[0]
    assert line.orientation == "vertical"
    assert line.start == pytest.approx((10.0, 5.0))
    assert line.end == pytest.approx((10.0, 85.0))
    assert line.bbox == pytest.approx((9.0, 5.0, 11.0, 85.0))
    assert line.width == pytest.approx(2.0)


def test_get_page_path_infos_applies_crop_box_page_rotation_and_stroke_width() -> None:
    """验证 Path bbox 在 CropBox 与页面旋转后仍保留可见描边宽度。"""

    with pdf_document.PDFDocument(_build_rotated_cropped_drawing_pdf()) as doc:
        path_infos = doc.get_page_path_infos(0)

    assert len(path_infos) == 1
    assert path_infos[0].bbox == pytest.approx((9.0, 4.0, 11.0, 86.0))
    assert path_infos[0].form_depth == 0
    assert path_infos[0].segment_count == 2


def test_get_page_image_bboxes_applies_forms_crop_box_rotation_and_clipping() -> None:
    """验证点阵图接口递归 Form，并按 CropBox、页面旋转裁剪为左上坐标。"""
    with pdf_document.PDFDocument(_build_rotated_cropped_image_pdf()) as doc:
        page_size = doc.page_size(0)
        image_bboxes = doc.get_page_image_bboxes(0)

    assert page_size == pytest.approx((180.0, 90.0))
    assert image_bboxes == pytest.approx(
        [
            (160.0, 0.0, 180.0, 15.0),
            (10.0, 5.0, 50.0, 35.0),
            (76.0, 47.0, 88.0, 53.0),
        ]
    )


def test_get_page_image_infos_preserves_bboxes_and_fingerprints_reused_images() -> None:
    """验证图片信息保持既有几何，并为普通与 Form 复用图生成相同内容指纹。"""

    with pdf_document.PDFDocument(_build_rotated_cropped_image_pdf()) as doc:
        image_infos = doc.get_page_image_infos(0)

    assert [info.bbox for info in image_infos] == pytest.approx(
        [
            (160.0, 0.0, 180.0, 15.0),
            (10.0, 5.0, 50.0, 35.0),
            (76.0, 47.0, 88.0, 53.0),
        ]
    )
    assert len({info.fingerprint for info in image_infos}) == 1
    assert image_infos[0].fingerprint is not None


def test_image_fingerprint_fails_open_when_raw_stream_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证超大图片流不分配缓冲区且返回空指纹，后续按普通图片放行。"""

    raw_data_calls: list[tuple[object | None, int]] = []

    def fake_metadata(raw_obj: object, raw_page: object, metadata_pointer: Any) -> int:
        """填入有效像素宽高，使测试只命中原始流大小上限。"""

        metadata_pointer._obj.width = 100
        metadata_pointer._obj.height = 200
        return 1

    def fake_raw_data(raw_obj: object, buffer: object | None, buffer_length: int) -> int:
        """声明超过上限的流，并记录是否发生第二次缓冲区读取。"""

        raw_data_calls.append((buffer, buffer_length))
        return pdf_document.PDF_IMAGE_FINGERPRINT_MAX_RAW_BYTES + 1

    class _FakePage:
        """提供 PDFium 元数据接口所需的最小 raw 页面句柄。"""

        raw = object()

    monkeypatch.setattr(pdf_document.pdfium_c, "FPDFImageObj_GetImageMetadata", fake_metadata)
    monkeypatch.setattr(pdf_document.pdfium_c, "FPDFImageObj_GetImageDataRaw", fake_raw_data)

    assert pdf_document._get_raw_image_fingerprint(object(), _FakePage()) is None
    assert raw_data_calls == [(None, 0)]


def test_get_page_form_bboxes_reads_root_forms_and_nested_content_bounds() -> None:
    """验证顶层 Form bbox 覆盖其嵌套绘图内容，且不重复输出内部对象。"""
    with pdf_document.PDFDocument(_build_drawing_pdf()) as doc:
        form_bboxes = doc.get_page_form_bboxes(0)

    assert form_bboxes == pytest.approx([(28.0, 99.0, 72.0, 101.0)])


def test_get_page_form_bboxes_applies_crop_box_rotation_and_clipping() -> None:
    """验证 Form bbox 按 CropBox 与页面旋转转换，并裁剪为左上原点坐标。"""
    with pdf_document.PDFDocument(_build_rotated_cropped_image_pdf()) as doc:
        page_size = doc.page_size(0)
        form_bboxes = doc.get_page_form_bboxes(0)

    assert page_size == pytest.approx((180.0, 90.0))
    assert form_bboxes == pytest.approx([(76.0, 47.0, 88.0, 53.0)])


def test_get_page_signature_bboxes_filters_visibility_and_applies_page_geometry() -> None:
    """验证仅输出可见正常签名，并正确应用 CropBox、旋转和页面裁剪。"""

    with pdf_document.PDFDocument(_build_rotated_cropped_signature_pdf()) as doc:
        page_size = doc.page_size(0)
        signature_bboxes = doc.get_page_signature_bboxes(0)

    assert page_size == pytest.approx((180.0, 90.0))
    assert signature_bboxes == pytest.approx(
        [
            (160.0, 0.0, 180.0, 15.0),
            (10.0, 5.0, 50.0, 35.0),
        ]
    )


def test_pdf_document_extracts_safe_external_link_annotations() -> None:
    """验证外部 URI 白名单、QuadPoints 优先级和旋转 CropBox 坐标转换。"""

    with pdf_document.PDFDocument(_build_rotated_cropped_link_pdf()) as document:
        page_size = document.page_size(0)
        links = document[0].get_link_annotations()

    assert page_size == pytest.approx((180.0, 90.0))
    assert [link.target for link in links] == [
        "https://example.com/a?x=1&y=2",
        "mailto:user@example.com",
        "tel:+123456",
    ]
    assert links[0].bboxes[0] == pytest.approx((15.0, 15.0, 25.0, 35.0))
    assert links[1].bboxes[0] == pytest.approx((40.0, 5.0, 50.0, 35.0))
    assert links[2].bboxes[0] == pytest.approx((60.0, 5.0, 70.0, 35.0))
    assert [link.source_index for link in links] == [0, 1, 2]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("HTTP://Example.com/path", "HTTP://Example.com/path"),
        ("mailto:user@example.com", "mailto:user@example.com"),
        ("tel:+123456", "tel:+123456"),
        ("https:///missing-host", None),
        ("mailto:", None),
        ("javascript:alert(1)", None),
        ("relative/path", None),
        ("https://example.com/a\x01b", None),
    ],
)
def test_pdf_external_link_target_validation(
    target: str,
    expected: str | None,
) -> None:
    """验证 PDF producer 只接受显式安全协议及完整目标。"""

    assert pdf_document._validate_pdf_external_link_target(target) == expected


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, (15.0, 155.0, 35.0, 165.0)),
        (90, (15.0, 15.0, 25.0, 35.0)),
        (180, (55.0, 15.0, 75.0, 25.0)),
        (270, (155.0, 55.0, 165.0, 75.0)),
    ],
)
def test_pdf_link_region_geometry_supports_standard_page_rotations(
    rotation: int,
    expected: tuple[float, float, float, float],
) -> None:
    """验证 Link 点集在四个标准页面方向下转换到统一视觉坐标。"""

    bbox = pdf_document._visual_bbox_from_pdf_points(
        [(20.0, 25.0), (40.0, 25.0), (20.0, 35.0), (40.0, 35.0)],
        (5.0, 10.0, 95.0, 190.0),
        rotation,
    )

    assert bbox == pytest.approx(expected)


def test_extract_page_signature_bboxes_closes_handles_and_skips_bad_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证损坏签名被隔离，且成功或失败路径都关闭注释句柄。"""

    bad_annot = object()
    good_annot = object()
    closed: list[object] = []

    class _FakePage:
        """提供注释原始接口所需的最小页面句柄。"""

        raw = object()

    def fake_get_annot(_raw_page: object, index: int) -> object:
        """按索引返回一个损坏注释和一个有效注释。"""

        return (bad_annot, good_annot)[index]

    def fake_signature_bbox(
        raw_annot: object,
        _page_bbox: tuple[float, float, float, float],
        _page_rotation: int,
        _form_handle: object | None,
    ) -> tuple[float, float, float, float]:
        """让首个注释抛错，验证第二个注释仍能被提取。"""

        if raw_annot is bad_annot:
            raise RuntimeError("broken annotation")
        return (10.0, 20.0, 30.0, 40.0)

    monkeypatch.setattr(pdf_document.pdfium_c, "FPDFPage_GetAnnotCount", lambda _page: 2)
    monkeypatch.setattr(pdf_document.pdfium_c, "FPDFPage_GetAnnot", fake_get_annot)
    monkeypatch.setattr(pdf_document.pdfium_c, "FPDFPage_CloseAnnot", closed.append)
    monkeypatch.setattr(pdf_document, "_signature_bbox_from_annotation", fake_signature_bbox)

    assert pdf_document._extract_page_signature_bboxes(
        _FakePage(),
        (0.0, 0.0, 100.0, 200.0),
        0,
    ) == [(10.0, 20.0, 30.0, 40.0)]
    assert closed == [bad_annot, good_annot]


def test_extract_page_form_bboxes_skips_one_bad_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证单个损坏 Form 不会阻断同页其他有效 Form 的提取。"""
    bad_object = object()
    good_object = object()

    def fake_form_bbox(
        raw_object: object,
        page_bbox: tuple[float, float, float, float],
        page_rotation: int,
    ) -> tuple[float, float, float, float]:
        """首个对象抛错，第二个对象返回可验证 bbox。"""
        assert page_bbox == (0.0, 0.0, 100.0, 200.0)
        assert page_rotation == 0
        if raw_object is bad_object:
            raise RuntimeError("broken form")
        return (10.0, 20.0, 30.0, 40.0)

    def fake_root_forms(_page: object) -> Any:
        """依次返回损坏对象与有效对象，验证逐对象异常隔离。"""
        return iter((bad_object, good_object))

    monkeypatch.setattr(
        pdf_document,
        "_iter_raw_root_form_objects",
        fake_root_forms,
    )
    monkeypatch.setattr(pdf_document, "_form_bbox_from_object", fake_form_bbox)

    assert pdf_document._extract_page_form_bboxes(
        object(),
        (0.0, 0.0, 100.0, 200.0),
        0,
    ) == [(10.0, 20.0, 30.0, 40.0)]


def test_get_page_drawing_lines_skips_one_bad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证单个 Path 解析异常不会丢失同页其他有效绘图线。"""
    original_extract = pdf_document._extract_path_drawing_lines
    call_count = 0

    def flaky_extract(*args: Any, **kwargs: Any) -> list[pdf_document.PDFDrawingLine]:
        """仅让首个 Path 失败，后续对象仍调用真实实现。"""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("broken path")
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(pdf_document, "_extract_path_drawing_lines", flaky_extract)
    with pdf_document.PDFDocument(_build_drawing_pdf()) as doc:
        lines = doc.get_page_drawing_lines(0)

    assert call_count >= 5
    assert len(lines) == 3
    assert all(line.start[1] != pytest.approx(20.0) for line in lines)
    assert [line.start[1] for line in lines] == pytest.approx([40.0, 60.25, 100.0])


def test_get_page_path_infos_skips_one_bad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证单个 Path 信息解析失败不会丢失同页其他有效对象。"""

    original_extract = pdf_document._path_info_from_object
    call_count = 0

    def flaky_extract(*args: Any, **kwargs: Any) -> pdf_document.PDFPathInfo | None:
        """仅让首个 Path 失败，后续对象仍调用真实实现。"""

        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("broken path")
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(pdf_document, "_path_info_from_object", flaky_extract)
    with pdf_document.PDFDocument(_build_drawing_pdf()) as doc:
        path_infos = doc.get_page_path_infos(0)

    assert call_count >= 8
    assert len(path_infos) == 6
    assert all(item.source_index != 0 for item in path_infos)
