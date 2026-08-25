# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterator, Literal, TypeAlias, cast
from urllib.parse import urlsplit

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_lines, get_spans
from pdftext.schema import Bbox, Char, Line
from PIL import Image, ImageOps

from ..types import BBox, PageInfo
from .pdf_classify import classify, get_sample_page_indices
from .pdf_image_tools import get_crop_img, load_images_from_pdf_bytes_range
from .pdf_reader import image_to_bytes
from .pdfium_guard import _pdfium_lock, safe_rewrite_pdf_bytes_with_pdfium

POINTS_PER_INCH: int = 72
DEFAULT_RENDER_DPI: int = 200
DEFAULT_RENDER_SCALE: float = DEFAULT_RENDER_DPI / POINTS_PER_INCH
DEFAULT_RENDER_MAX_EDGE: int = 3500
NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE = 1.0
OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE = 2.5
OFFSET_DUPLICATE_TRANSLATION_TOLERANCE = 0.1
OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO = 0.45
DRAWING_FORM_MAX_DEPTH = 15
DRAWING_LINE_MERGE_TOLERANCE = 2.0
DRAWING_LINE_AXIS_ABSOLUTE_TOLERANCE = 1.0
DRAWING_LINE_AXIS_RATIO_TOLERANCE = 0.02
DRAWING_LINE_MIN_LENGTH = 1.0
DRAWING_THIN_RECT_MAX_THICKNESS = 2.0
DRAWING_THIN_RECT_MIN_ASPECT_RATIO = 4.0
PDF_IMAGE_FINGERPRINT_MAX_RAW_BYTES = 16 * 1024 * 1024
_PDF_EXTERNAL_LINK_SCHEMES = frozenset({"http", "https", "mailto", "tel"})

try:
    from pdftext.pdf.chars import PageChars
except ImportError:
    PageChars = None

# See: pdfium.PdfDocument.METADATA_KEYS
PDFMetadataKey: TypeAlias = Literal[
    "Title",
    "Author",
    "Subject",
    "Keywords",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
]


class PDFPageImage:
    def __init__(self, pil_image: Image.Image, scale: float) -> None:
        self.pil_image = pil_image
        self.scale = scale


@dataclass(frozen=True)
class PDFDrawingLine:
    """PDF 页面中可见的水平或竖直绘图线，坐标使用页面左上原点的 PDF point。"""

    start: tuple[float, float]
    end: tuple[float, float]
    bbox: BBox
    width: float
    orientation: Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class PDFLinkAnnotation:
    """PDF 外部 URI Link 注解，区域坐标使用页面左上原点的 PDF point。"""

    target: str
    bboxes: tuple[BBox, ...]
    source_index: int


@dataclass(frozen=True)
class PDFPathInfo:
    """PDF Path 的可见几何和绘制特征，bbox 使用页面左上原点坐标。"""

    bbox: BBox
    segment_count: int
    fill_visible: bool
    stroke_visible: bool
    form_depth: int
    source_index: int
    fill_rgba: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class PDFImageInfo:
    """PDF 点阵图的页面几何与稳定内容指纹，指纹读取失败时为 None。"""

    bbox: BBox
    fingerprint: str | None


@dataclass
class _PathSubpath:
    """保存一个 PDF Path 子路径的点、直线段和闭合状态。"""

    points: list[tuple[float, float]]
    straight_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    closed: bool = False


class PDFPage:
    def __init__(self, pdf_doc: "PDFDocument", idx: int) -> None:
        self.pdf_doc = pdf_doc
        self._idx = idx

    @property
    def size(self) -> tuple[float, float]:
        return self.pdf_doc.page_size(self._idx)

    @property
    def rotation(self) -> Literal[0, 90, 180, 270]:
        """只读返回当前页声明的标准旋转角度。"""

        return self.pdf_doc.page_rotation(self._idx)

    def get_char_count(self) -> int:
        return self.pdf_doc.page_char_count(self._idx)

    def get_chars(self) -> list[Char]:
        return self.pdf_doc.get_page_chars(self._idx)

    def get_drawing_lines(self) -> list[PDFDrawingLine]:
        """读取当前页已规范到视觉页面坐标的横竖 drawing。"""

        return self.pdf_doc.get_page_drawing_lines(self._idx)

    def get_path_infos(self) -> list[PDFPathInfo]:
        """读取当前页及嵌套 Form 中已规范到视觉页面坐标的 Path 摘要。"""

        return self.pdf_doc.get_page_path_infos(self._idx)

    def get_link_annotations(self) -> list[PDFLinkAnnotation]:
        """读取当前页已规范到视觉页面坐标的外部 URI Link 注解。"""

        return self.pdf_doc.get_page_link_annotations(self._idx)


class PDFDocument:
    """A PDF file loaded in memory, with lazy pypdfium2 access.

    This object is responsible for all access to, and lifecycle management of,
    the associated PDFium document/page objects.

    All pypdfium2 operations are serialized under a module-level lock for
    thread safety. Call ``close()`` when done, or use as a context manager.

    The class does not expose raw PDFium objects or methods without wrapping
    them first. Callers may use this class to read or operate on the
    underlying PDFium state, but they must not directly access PDFium objects
    through this API.
    """

    def __init__(
        self,
        pdf_bytes_or_path: bytes | str,
        render_scale: float = DEFAULT_RENDER_SCALE,
        render_max_edge: int = DEFAULT_RENDER_MAX_EDGE,
    ) -> None:
        if isinstance(pdf_bytes_or_path, bytes):
            self._pdf_bytes: bytes = pdf_bytes_or_path
        else:
            assert isinstance(pdf_bytes_or_path, str)
            with open(pdf_bytes_or_path, "rb") as f:
                self._pdf_bytes = f.read()

        self._pdf_doc_opened: pdfium.PdfDocument | None = None
        self._page_count: int | None = None
        self.render_scale = render_scale
        self.render_max_edge = render_max_edge

    # ------------------------------------------------------------------ #
    #  Factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_image(image_bytes: bytes) -> "PDFDocument":
        return PDFDocument(images_bytes_to_pdf_bytes(image_bytes))

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._pdf_doc_opened is not None:
            with _pdfium_lock:
                _try_close(self._pdf_doc_opened)
                self._pdf_doc_opened = None

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.page_count

    def __getitem__(self, idx: int) -> PDFPage:
        return PDFPage(self, idx)

    @property
    def page_count(self) -> int:
        with _pdfium_lock:
            return len(self._pdf_doc)

    @property
    def metadata(self) -> dict[PDFMetadataKey, str]:
        with _pdfium_lock:
            metadata = self._pdf_doc.get_metadata_dict()
        return cast(dict[PDFMetadataKey, str], metadata)

    @property
    def bytes(self) -> bytes:
        # TODO: some invoker expected PDF bytes even if input bytes is an Image.
        return self._pdf_bytes

    # ------------------------------------------------------------------ #
    #  Metadata
    # ------------------------------------------------------------------ #

    def page_size(self, page_idx: int) -> tuple[float, float]:
        with self._open_page(page_idx) as page:
            # rect: (left, bottom, right, top)
            rect: tuple[float, float, float, float] = page.get_bbox()
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
        width = abs(rect[2] - rect[0])
        height = abs(rect[1] - rect[3])
        # PDFium 文本与渲染坐标已经应用页面旋转，页面尺寸必须使用相同视觉方向。
        return (height, width) if page_rotation in {90, 270} else (width, height)

    def page_rotation(self, page_idx: int) -> Literal[0, 90, 180, 270]:
        """在线程锁保护下返回 PDF 页面字典声明的标准旋转角度。"""

        with self._open_page(page_idx) as page:
            try:
                rotation = int(page.get_rotation()) % 360
            except Exception:
                rotation = 0
        return rotation if rotation in {0, 90, 180, 270} else 0

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def render_page(self, page_idx: int, *, scale: float | None = None) -> PDFPageImage:
        if scale is None:
            scale = self.render_scale
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        with self._open_page(page_idx) as page:
            return _page_to_image(page, scale, self.render_max_edge)

    def render_pages(self, start: int = 0, end: int | None = None, *, scale: float | None = None) -> list[PDFPageImage]:
        if end is None:
            end = self.page_count - 1
        if scale is None:
            scale = self.render_scale
        if scale <= 0:
            raise ValueError("scale must be greater than 0")
        results = load_images_from_pdf_bytes_range(
            pdf_bytes=self.bytes,
            dpi=max(1, int(round(scale * POINTS_PER_INCH))),
            start_page_id=start,
            end_page_id=end,
        )
        return [PDFPageImage(pil_image=r["img_pil"], scale=r["scale"]) for r in results]

    async def render_page_async(self, page_idx: int, *, scale: float | None = None) -> PDFPageImage:
        return await asyncio.to_thread(self.render_page, page_idx, scale=scale)

    async def render_pages_async(
        self, start: int = 0, end: int | None = None, *, scale: float | None = None
    ) -> list[PDFPageImage]:
        return await asyncio.to_thread(self.render_pages, start, end, scale=scale)

    # TODO: move
    def crop_image(self, bbox: BBox, page_idx: int, *, scale: int = 2) -> bytes:
        image = self.render_page(page_idx, scale=scale)
        crop = None
        try:
            crop = get_crop_img(bbox, image.pil_image, scale=image.scale)
            return image_to_bytes(crop, image_format="JPEG")
        finally:
            if crop is not None:
                crop.close()
            image.pil_image.close()

    # ------------------------------------------------------------------ #
    #  Text
    # ------------------------------------------------------------------ #

    def page_char_count(self, page_idx: int) -> int:
        with self._open_page(page_idx) as page:
            textpage = None
            try:
                textpage = page.get_textpage()
                n_chars = textpage.count_chars()
            finally:
                _try_close(textpage)
        return cast(int, n_chars)

    def get_page_chars(self, page_idx: int) -> list[Char]:
        with self._open_page(page_idx) as page:
            textpage = None
            try:
                textpage = page.get_textpage()
                page_bbox: list[float] = list(page.get_bbox())
                page_rotation: int = 0
                try:
                    page_rotation = page.get_rotation()
                except Exception:
                    pass
                chars = get_chars(textpage, page_bbox, page_rotation)
                chars = _deduplicate_pdftext_chars(chars)
                chars = _ensure_legacy_chars(chars)
                chars = _restore_pdfium_surrogate_pairs(chars, textpage)
            finally:
                _try_close(textpage)
        return _deduplicate_near_identical_chars(chars)

    def get_page_lines(self, page_idx: int) -> list[Line]:
        chars = self.get_page_chars(page_idx)
        return get_lines_from_chars(chars)

    def get_page_text(self, page_idx: int) -> str:
        with self._open_page(page_idx) as page:
            textpage = None
            try:
                textpage = page.get_textpage()
                text = textpage.get_text_range()
            finally:
                _try_close(textpage)
        return text or ""

    # ------------------------------------------------------------------ #
    #  Drawing geometry
    # ------------------------------------------------------------------ #

    def get_page_drawing_lines(self, page_idx: int) -> list[PDFDrawingLine]:
        """提取页面中可见的水平、竖直绘图线，并转换为页面左上原点坐标。"""
        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_drawing_lines(page, page_bbox, page_rotation)

    def get_page_path_infos(self, page_idx: int) -> list[PDFPathInfo]:
        """提取页面及嵌套 Form 中的 Path 几何和绘制特征。"""
        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_path_infos(page, page_bbox, page_rotation)

    def get_page_image_bboxes(self, page_idx: int) -> list[BBox]:
        """提取页面及嵌套 Form 中的点阵图 bbox，并转换为页面左上原点坐标。"""
        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_image_bboxes(page, page_bbox, page_rotation)

    def get_page_image_infos(self, page_idx: int) -> list[PDFImageInfo]:
        """提取点阵图 bbox 与内容指纹，供 Flash 在认领正文前识别跨页重复图。"""

        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_image_infos(page, page_bbox, page_rotation)

    def get_page_form_bboxes(self, page_idx: int) -> list[BBox]:
        """提取页面顶层 Form XObject bbox，并转换为页面左上原点坐标。"""
        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_form_bboxes(page, page_bbox, page_rotation)

    def get_page_signature_bboxes(self, page_idx: int) -> list[BBox]:
        """提取带可见正常外观的数字签名控件，并转换为页面左上原点坐标。"""

        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_signature_bboxes(
                page,
                page_bbox,
                page_rotation,
                form_handle=getattr(self._pdf_doc, "formenv", None),
            )

    def get_page_link_annotations(self, page_idx: int) -> list[PDFLinkAnnotation]:
        """提取可见且目标安全的外部 URI Link 注解。"""

        with self._open_page(page_idx) as page:
            page_bbox = _normalize_pdf_page_bbox(page.get_bbox())
            try:
                page_rotation = int(page.get_rotation()) % 360
            except Exception:
                page_rotation = 0
            return _extract_page_link_annotations(
                page,
                self._pdf_doc.raw,
                page_bbox,
                page_rotation,
            )

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify(self) -> Literal["ocr", "txt"]:
        pdf_class = classify(self._pdf_doc, self.bytes)
        return cast(Literal["ocr", "txt"], pdf_class)

    # ------------------------------------------------------------------ #
    #  Page extraction
    # ------------------------------------------------------------------ #

    # TODO: no caller
    def extract_page_range(self, start: int, end: int) -> "PDFDocument":
        new_bytes = safe_rewrite_pdf_bytes_with_pdfium(
            self._pdf_bytes,
            start_page_id=start,
            end_page_id=end,
        )
        return PDFDocument(new_bytes)

    # TODO: no caller
    def sample_pages(self, max_pages: int = 3) -> "PDFDocument":
        """按 PDF 分类抽样规则提取代表性页面，返回新的 PDFDocument。"""
        if max_pages <= 0:
            return PDFDocument(b"")

        page_indices = get_sample_page_indices(self.page_count, max_pages)
        if page_indices:
            new_bytes = safe_rewrite_pdf_bytes_with_pdfium(
                self._pdf_bytes,
                page_indices=page_indices,
            )
            if new_bytes:
                return PDFDocument(new_bytes)
        return PDFDocument(b"")

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def draw_layout_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        from .draw_bbox import draw_layout_bbox

        out_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        draw_layout_bbox(pages, self._pdf_bytes, out_dir, filename)

    def draw_span_bbox(self, pages: list[PageInfo], output_path: str) -> None:
        from .draw_bbox import draw_span_bbox

        out_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        draw_span_bbox(pages, self._pdf_bytes, out_dir, filename)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    @property
    def _pdf_doc(self) -> pdfium.PdfDocument:
        if self._pdf_doc_opened is None:
            with _pdfium_lock:
                if self._pdf_doc_opened is None:
                    pdf_doc = pdfium.PdfDocument(self._pdf_bytes)
                    try:
                        # 表单环境必须在打开页面前初始化，签名字段类型才能沿 Parent 正确继承。
                        pdf_doc.init_forms()
                    except Exception:
                        # 异常表单不能阻断普通文本、绘图和渲染接口。
                        pass
                    self._pdf_doc_opened = pdf_doc
        return self._pdf_doc_opened

    @contextmanager
    def _open_page(self, page_idx: int) -> Iterator[pdfium.PdfPage]:
        """Open and process page with _pdfium_lock"""
        with _pdfium_lock:
            page = None
            try:
                page = self._pdf_doc[page_idx]
                yield page
            finally:
                _try_close(page)


def _normalize_pdf_page_bbox(bbox: tuple[float, float, float, float]) -> BBox:
    """规范化 PDFium 页面框，兼容上下坐标顺序相反的测试或异常文档。"""
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _drawing_page_size(page_bbox: BBox, page_rotation: int) -> tuple[float, float]:
    """根据未旋转页面框和页面旋转角计算左上坐标系中的页面尺寸。"""
    width = page_bbox[2] - page_bbox[0]
    height = page_bbox[3] - page_bbox[1]
    if page_rotation in (90, 270):
        return height, width
    return width, height


def _transform_drawing_point(
    point: tuple[float, float],
    page_bbox: BBox,
    page_rotation: int,
) -> tuple[float, float]:
    """把 PDF 底左原点坐标转换为应用页面旋转后的左上原点坐标。"""
    x, y = point
    left, bottom, right, top = page_bbox
    if page_rotation == 90:
        return y - bottom, x - left
    if page_rotation == 180:
        return right - x, y - bottom
    if page_rotation == 270:
        return top - y, right - x
    return x - left, top - y


def _multiply_pdf_matrices(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """按 PDF 行向量约定合并对象矩阵与父 Form 矩阵。"""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _apply_pdf_matrix(
    point: tuple[float, float],
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """将 PDF 仿射矩阵应用到一个路径点。"""
    x, y = point
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _get_raw_object_matrix(raw_obj: Any) -> tuple[float, float, float, float, float, float] | None:
    """读取一个原始 PDFium 页面对象矩阵，读取失败时返回 None。"""
    matrix = pdfium_c.FS_MATRIX()
    try:
        ok = pdfium_c.FPDFPageObj_GetMatrix(raw_obj, ctypes.byref(matrix))
    except Exception:
        return None
    if not ok:
        return None
    return (
        float(matrix.a),
        float(matrix.b),
        float(matrix.c),
        float(matrix.d),
        float(matrix.e),
        float(matrix.f),
    )


def _walk_raw_page_objects_with_depth(
    container: Any,
    *,
    is_form: bool,
    parent_matrix: tuple[float, float, float, float, float, float],
    depth: int,
    target_type: int,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float], int]]:
    """递归遍历目标对象，并携带累积矩阵及所在 Form 深度。"""
    if depth >= DRAWING_FORM_MAX_DEPTH:
        return

    count_objects = pdfium_c.FPDFFormObj_CountObjects if is_form else pdfium_c.FPDFPage_CountObjects
    get_object = pdfium_c.FPDFFormObj_GetObject if is_form else pdfium_c.FPDFPage_GetObject
    try:
        object_count = int(count_objects(container))
    except Exception:
        return
    if object_count < 0:
        return

    for object_index in range(object_count):
        try:
            raw_obj = get_object(container, object_index)
            if not raw_obj:
                continue
            object_type = int(pdfium_c.FPDFPageObj_GetType(raw_obj))
            object_matrix = _get_raw_object_matrix(raw_obj)
            if object_matrix is None:
                continue
            combined_matrix = _multiply_pdf_matrices(object_matrix, parent_matrix)
        except Exception:
            # 单个损坏页面对象不能中断同页其他目标对象遍历。
            continue

        if object_type == target_type:
            yield raw_obj, combined_matrix, depth
        if object_type == pdfium_c.FPDF_PAGEOBJ_FORM:
            yield from _walk_raw_page_objects_with_depth(
                raw_obj,
                is_form=True,
                parent_matrix=combined_matrix,
                depth=depth + 1,
                target_type=target_type,
            )


def _walk_raw_page_objects(
    container: Any,
    *,
    is_form: bool,
    parent_matrix: tuple[float, float, float, float, float, float],
    depth: int,
    target_type: int,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float]]]:
    """兼容既有调用，仅返回目标对象及累积到页面坐标的矩阵。"""

    for raw_obj, matrix, _form_depth in _walk_raw_page_objects_with_depth(
        container,
        is_form=is_form,
        parent_matrix=parent_matrix,
        depth=depth,
        target_type=target_type,
    ):
        yield raw_obj, matrix


def _walk_raw_path_objects(
    container: Any,
    *,
    is_form: bool,
    parent_matrix: tuple[float, float, float, float, float, float],
    depth: int,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float]]]:
    """递归遍历页面或 Form 中的 Path，并携带累积到页面坐标的矩阵。"""
    yield from _walk_raw_page_objects(
        container,
        is_form=is_form,
        parent_matrix=parent_matrix,
        depth=depth,
        target_type=pdfium_c.FPDF_PAGEOBJ_PATH,
    )


def _iter_raw_path_objects(
    page: pdfium.PdfPage,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float]]]:
    """从页面根对象开始遍历全部 Path，包括嵌套 Form 中的 Path。"""
    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    yield from _walk_raw_path_objects(
        page,
        is_form=False,
        parent_matrix=identity,
        depth=0,
    )


def _iter_raw_path_objects_with_depth(
    page: pdfium.PdfPage,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float], int]]:
    """遍历全部 Path，并保留其所在 Form 深度供上层过滤图标。"""

    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    yield from _walk_raw_page_objects_with_depth(
        page,
        is_form=False,
        parent_matrix=identity,
        depth=0,
        target_type=pdfium_c.FPDF_PAGEOBJ_PATH,
    )


def _iter_raw_image_objects(
    page: pdfium.PdfPage,
) -> Iterator[tuple[Any, tuple[float, float, float, float, float, float]]]:
    """从页面根对象开始遍历全部点阵图，包括嵌套 Form 中的图片。"""
    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    yield from _walk_raw_page_objects(
        page,
        is_form=False,
        parent_matrix=identity,
        depth=0,
        target_type=pdfium_c.FPDF_PAGEOBJ_IMAGE,
    )


def _iter_raw_root_form_objects(page: pdfium.PdfPage) -> Iterator[Any]:
    """只遍历页面根层的 Form，避免把同一矢量图的嵌套子 Form 重复输出。"""

    try:
        object_count = int(pdfium_c.FPDFPage_CountObjects(page))
    except Exception:
        return
    if object_count < 0:
        return

    for object_index in range(object_count):
        try:
            raw_obj = pdfium_c.FPDFPage_GetObject(page, object_index)
            if raw_obj and int(pdfium_c.FPDFPageObj_GetType(raw_obj)) == pdfium_c.FPDF_PAGEOBJ_FORM:
                yield raw_obj
        except Exception:
            # 单个损坏对象不能阻断同页其他顶层 Form 的提取。
            continue


def _get_raw_object_rgba(
    raw_obj: Any,
    color_getter: Any,
) -> tuple[int, int, int, int] | None:
    """读取对象 RGBA；旧 PDFium、不支持的对象或读取失败时返回 None。"""

    red = ctypes.c_uint()
    green = ctypes.c_uint()
    blue = ctypes.c_uint()
    alpha = ctypes.c_uint()
    try:
        ok = color_getter(
            raw_obj,
            ctypes.byref(red),
            ctypes.byref(green),
            ctypes.byref(blue),
            ctypes.byref(alpha),
        )
    except Exception:
        return None
    if not ok:
        return None
    return (
        int(red.value),
        int(green.value),
        int(blue.value),
        int(alpha.value),
    )


def _get_raw_object_alpha(raw_obj: Any, color_getter: Any) -> int:
    """读取对象颜色 alpha；旧 PDFium 或读取失败时按不透明处理。"""

    rgba = _get_raw_object_rgba(raw_obj, color_getter)
    return rgba[3] if rgba is not None else 255


def _get_path_visibility(raw_obj: Any) -> tuple[bool, bool]:
    """分别判断 Path 的填充与描边是否实际可见。"""
    fill_mode = ctypes.c_int()
    stroke = ctypes.c_int()
    try:
        ok = pdfium_c.FPDFPath_GetDrawMode(raw_obj, ctypes.byref(fill_mode), ctypes.byref(stroke))
    except Exception:
        return False, False
    if not ok:
        return False, False

    fill_visible = (
        fill_mode.value != pdfium_c.FPDF_FILLMODE_NONE and _get_raw_object_alpha(raw_obj, pdfium_c.FPDFPageObj_GetFillColor) > 0
    )
    stroke_visible = bool(stroke.value) and _get_raw_object_alpha(raw_obj, pdfium_c.FPDFPageObj_GetStrokeColor) > 0
    return fill_visible, stroke_visible


def _get_raw_stroke_width(raw_obj: Any) -> float:
    """读取 Path 在对象局部坐标中的描边宽度。"""
    stroke_width = ctypes.c_float()
    try:
        ok = pdfium_c.FPDFPageObj_GetStrokeWidth(raw_obj, ctypes.byref(stroke_width))
    except Exception:
        return 0.0
    if not ok:
        return 0.0
    width = abs(float(stroke_width.value))
    return width if math.isfinite(width) else 0.0


def _get_segment_stroke_width(
    raw_width: float,
    start: tuple[float, float],
    end: tuple[float, float],
    matrix: tuple[float, float, float, float, float, float],
) -> float:
    """按线段局部法向量换算非等比矩阵下的实际描边宽度。"""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    segment_length = math.hypot(delta_x, delta_y)
    a, b, c, d, _, _ = matrix
    if segment_length <= 0:
        scale = math.sqrt(abs(a * d - b * c))
    else:
        normal_x = -delta_y / segment_length
        normal_y = delta_x / segment_length
        transformed_x = a * normal_x + c * normal_y
        transformed_y = b * normal_x + d * normal_y
        scale = math.hypot(transformed_x, transformed_y)
    width = raw_width * scale
    return width if math.isfinite(width) else 0.0


def _read_raw_path_subpaths(raw_obj: Any) -> list[_PathSubpath]:
    """读取 Path 段并拆为子路径，只把 LINETO 与闭合边记录为直线段。"""
    try:
        segment_count = int(pdfium_c.FPDFPath_CountSegments(raw_obj))
    except Exception:
        return []
    if segment_count <= 0:
        return []

    subpaths: list[_PathSubpath] = []
    current_subpath: _PathSubpath | None = None
    current_point: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None

    for segment_index in range(segment_count):
        try:
            segment = pdfium_c.FPDFPath_GetPathSegment(raw_obj, segment_index)
            if not segment:
                continue
            x = ctypes.c_float()
            y = ctypes.c_float()
            if not pdfium_c.FPDFPathSegment_GetPoint(segment, ctypes.byref(x), ctypes.byref(y)):
                continue
            point = (float(x.value), float(y.value))
            segment_type = int(pdfium_c.FPDFPathSegment_GetType(segment))
            segment_closes = bool(pdfium_c.FPDFPathSegment_GetClose(segment))
        except Exception:
            continue

        if segment_type == pdfium_c.FPDF_SEGMENT_MOVETO:
            if current_subpath is not None and current_subpath.points:
                subpaths.append(current_subpath)
            current_subpath = _PathSubpath(points=[point], straight_segments=[])
            current_point = point
            subpath_start = point
        else:
            if current_subpath is None:
                current_subpath = _PathSubpath(points=[point], straight_segments=[])
                current_point = point
                subpath_start = point
            else:
                current_subpath.points.append(point)
                if segment_type == pdfium_c.FPDF_SEGMENT_LINETO and current_point is not None:
                    current_subpath.straight_segments.append((current_point, point))
                current_point = point

        if segment_closes and current_subpath is not None and current_point is not None and subpath_start is not None:
            if current_point != subpath_start:
                current_subpath.straight_segments.append((current_point, subpath_start))
            current_subpath.closed = True
            current_point = subpath_start

    if current_subpath is not None and current_subpath.points:
        subpaths.append(current_subpath)
    return subpaths


def _transform_path_subpath(
    subpath: _PathSubpath,
    matrix: tuple[float, float, float, float, float, float],
    page_bbox: BBox,
    page_rotation: int,
) -> _PathSubpath:
    """将一个子路径从对象局部坐标转换为页面左上坐标。"""

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        """先应用对象/Form 矩阵，再应用页面坐标与旋转转换。"""
        return _transform_drawing_point(_apply_pdf_matrix(point, matrix), page_bbox, page_rotation)

    return _PathSubpath(
        points=[transform(point) for point in subpath.points],
        straight_segments=[(transform(start), transform(end)) for start, end in subpath.straight_segments],
        closed=subpath.closed,
    )


def _path_info_from_object(
    raw_obj: Any,
    matrix: tuple[float, float, float, float, float, float],
    page_bbox: BBox,
    page_rotation: int,
    form_depth: int,
    source_index: int,
) -> PDFPathInfo | None:
    """将一个原始 Path 转换为页面几何；贝塞尔控制点也纳入保守 bbox。"""

    try:
        segment_count = int(pdfium_c.FPDFPath_CountSegments(raw_obj))
    except Exception:
        return None
    if segment_count <= 0:
        return None

    fill_visible, stroke_visible = _get_path_visibility(raw_obj)
    points = [
        point
        for raw_subpath in _read_raw_path_subpaths(raw_obj)
        for point in _transform_path_subpath(
            raw_subpath,
            matrix,
            page_bbox,
            page_rotation,
        ).points
    ]
    if not points:
        return None
    coordinates = [coordinate for point in points for coordinate in point]
    if not all(math.isfinite(value) for value in coordinates):
        return None

    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    stroke_margin = 0.0
    if stroke_visible:
        a, b, c, d, _, _ = matrix
        stroke_scale = max(math.hypot(a, b), math.hypot(c, d))
        stroke_margin = 0.5 * _get_raw_stroke_width(raw_obj) * stroke_scale
    bbox = (
        max(0.0, min(point[0] for point in points) - stroke_margin),
        max(0.0, min(point[1] for point in points) - stroke_margin),
        min(page_width, max(point[0] for point in points) + stroke_margin),
        min(page_height, max(point[1] for point in points) + stroke_margin),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return PDFPathInfo(
        bbox=bbox,
        segment_count=segment_count,
        fill_visible=fill_visible,
        stroke_visible=stroke_visible,
        form_depth=form_depth,
        source_index=source_index,
        fill_rgba=(
            _get_raw_object_rgba(
                raw_obj,
                pdfium_c.FPDFPageObj_GetFillColor,
            )
            if fill_visible
            else None
        ),
    )


def _get_thin_filled_subpath_line(
    subpath: _PathSubpath,
    page_size: tuple[float, float],
) -> PDFDrawingLine | None:
    """把闭合的细长填充子路径折叠为一条中心线，避免把矩形四边重复输出。"""
    if not subpath.closed or len(subpath.points) < 4:
        return None
    x_values = [point[0] for point in subpath.points]
    y_values = [point[1] for point in subpath.points]
    x0, x1 = min(x_values), max(x_values)
    y0, y1 = min(y_values), max(y_values)
    width = x1 - x0
    height = y1 - y0
    long_side = max(width, height)
    short_side = min(width, height)
    if (
        long_side < DRAWING_LINE_MIN_LENGTH
        or short_side > DRAWING_THIN_RECT_MAX_THICKNESS
        or long_side < DRAWING_THIN_RECT_MIN_ASPECT_RATIO * max(short_side, 0.01)
    ):
        return None
    if width >= height:
        return _make_axis_drawing_line((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2), short_side, page_size)
    return _make_axis_drawing_line(((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), short_side, page_size)


def _make_axis_drawing_line(
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
    page_size: tuple[float, float],
) -> PDFDrawingLine | None:
    """将近水平或近竖直线段吸附到坐标轴、裁剪到页面并生成公开结果。"""
    page_width, page_height = page_size
    x0, y0 = start
    x1, y1 = end
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1, width)):
        return None
    delta_x = abs(x1 - x0)
    delta_y = abs(y1 - y0)

    if delta_x >= delta_y and delta_y <= max(
        DRAWING_LINE_AXIS_ABSOLUTE_TOLERANCE,
        delta_x * DRAWING_LINE_AXIS_RATIO_TOLERANCE,
    ):
        coordinate = (y0 + y1) / 2
        if coordinate < 0 or coordinate > page_height:
            return None
        main_start = max(0.0, min(x0, x1))
        main_end = min(page_width, max(x0, x1))
        if main_end - main_start < DRAWING_LINE_MIN_LENGTH:
            return None
        line_width = max(0.0, width, delta_y)
        half_width = line_width / 2
        bbox = (
            main_start,
            max(0.0, coordinate - half_width),
            main_end,
            min(page_height, coordinate + half_width),
        )
        return PDFDrawingLine(
            start=(main_start, coordinate),
            end=(main_end, coordinate),
            bbox=bbox,
            width=line_width,
            orientation="horizontal",
        )

    if delta_y > delta_x and delta_x <= max(
        DRAWING_LINE_AXIS_ABSOLUTE_TOLERANCE,
        delta_y * DRAWING_LINE_AXIS_RATIO_TOLERANCE,
    ):
        coordinate = (x0 + x1) / 2
        if coordinate < 0 or coordinate > page_width:
            return None
        main_start = max(0.0, min(y0, y1))
        main_end = min(page_height, max(y0, y1))
        if main_end - main_start < DRAWING_LINE_MIN_LENGTH:
            return None
        line_width = max(0.0, width, delta_x)
        half_width = line_width / 2
        bbox = (
            max(0.0, coordinate - half_width),
            main_start,
            min(page_width, coordinate + half_width),
            main_end,
        )
        return PDFDrawingLine(
            start=(coordinate, main_start),
            end=(coordinate, main_end),
            bbox=bbox,
            width=line_width,
            orientation="vertical",
        )
    return None


def _extract_path_drawing_lines(
    raw_obj: Any,
    matrix: tuple[float, float, float, float, float, float],
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFDrawingLine]:
    """从单个 Path 提取可见直线，坏 Path 返回空结果且不影响同页其他对象。"""
    fill_visible, stroke_visible = _get_path_visibility(raw_obj)
    if not fill_visible and not stroke_visible:
        return []

    page_size = _drawing_page_size(page_bbox, page_rotation)
    raw_stroke_width = _get_raw_stroke_width(raw_obj) if stroke_visible else 0.0
    drawing_lines: list[PDFDrawingLine] = []
    for raw_subpath in _read_raw_path_subpaths(raw_obj):
        subpath = _transform_path_subpath(raw_subpath, matrix, page_bbox, page_rotation)
        if fill_visible:
            filled_line = _get_thin_filled_subpath_line(subpath, page_size)
            if filled_line is not None:
                drawing_lines.append(filled_line)
                # 细长填充矩形已经折叠为中心线，不再输出其描边四条边。
                continue
        if not stroke_visible:
            continue
        for (raw_start, raw_end), (segment_start, segment_end) in zip(
            raw_subpath.straight_segments,
            subpath.straight_segments,
        ):
            stroke_width = _get_segment_stroke_width(raw_stroke_width, raw_start, raw_end, matrix)
            drawing_line = _make_axis_drawing_line(segment_start, segment_end, stroke_width, page_size)
            if drawing_line is not None:
                drawing_lines.append(drawing_line)
    return drawing_lines


def _line_axis_coordinate(line: PDFDrawingLine) -> float:
    """返回绘图线在垂直于自身方向的轴坐标。"""
    return line.start[1] if line.orientation == "horizontal" else line.start[0]


def _line_main_interval(line: PDFDrawingLine) -> tuple[float, float]:
    """返回绘图线沿自身方向的起止区间。"""
    if line.orientation == "horizontal":
        return line.start[0], line.end[0]
    return line.start[1], line.end[1]


def _combine_collinear_line_group(
    lines: list[PDFDrawingLine],
    page_size: tuple[float, float],
) -> PDFDrawingLine | None:
    """把坐标接近且区间相连的一组线合并为一条稳定中心线。"""
    orientation = lines[0].orientation
    intervals = [_line_main_interval(line) for line in lines]
    lengths = [max(end - start, 0.01) for start, end in intervals]
    coordinates = [_line_axis_coordinate(line) for line in lines]
    total_length = sum(lengths)
    coordinate = sum(value * length for value, length in zip(coordinates, lengths)) / total_length
    coordinate_span = max(coordinates) - min(coordinates)
    width = max(max(line.width for line in lines), coordinate_span + max(line.width for line in lines))
    main_start = min(start for start, _ in intervals)
    main_end = max(end for _, end in intervals)
    if orientation == "horizontal":
        return _make_axis_drawing_line((main_start, coordinate), (main_end, coordinate), width, page_size)
    return _make_axis_drawing_line((coordinate, main_start), (coordinate, main_end), width, page_size)


def _merge_orientation_lines(
    lines: list[PDFDrawingLine],
    page_size: tuple[float, float],
) -> list[PDFDrawingLine]:
    """按轴坐标聚类，再合并间距不超过约 2pt 的同方向线段。"""
    if not lines:
        return []
    coordinate_clusters: list[list[PDFDrawingLine]] = []
    for line in sorted(lines, key=lambda item: (_line_axis_coordinate(item), _line_main_interval(item)[0])):
        if (
            not coordinate_clusters
            or _line_axis_coordinate(line) - min(_line_axis_coordinate(item) for item in coordinate_clusters[-1])
            > DRAWING_LINE_MERGE_TOLERANCE
        ):
            coordinate_clusters.append([line])
        else:
            coordinate_clusters[-1].append(line)

    merged: list[PDFDrawingLine] = []
    for cluster in coordinate_clusters:
        interval_group: list[PDFDrawingLine] = []
        interval_end = -math.inf
        for line in sorted(cluster, key=lambda item: _line_main_interval(item)[0]):
            line_start, line_end = _line_main_interval(line)
            if interval_group and line_start > interval_end + DRAWING_LINE_MERGE_TOLERANCE:
                combined = _combine_collinear_line_group(interval_group, page_size)
                if combined is not None:
                    merged.append(combined)
                interval_group = []
                interval_end = -math.inf
            interval_group.append(line)
            interval_end = max(interval_end, line_end)
        if interval_group:
            combined = _combine_collinear_line_group(interval_group, page_size)
            if combined is not None:
                merged.append(combined)
    return merged


def _merge_collinear_drawing_lines(
    lines: list[PDFDrawingLine],
    page_size: tuple[float, float],
) -> list[PDFDrawingLine]:
    """合并水平和竖直共线段，并按页面视觉位置稳定排序。"""
    horizontal = _merge_orientation_lines(
        [line for line in lines if line.orientation == "horizontal"],
        page_size,
    )
    vertical = _merge_orientation_lines(
        [line for line in lines if line.orientation == "vertical"],
        page_size,
    )
    return sorted(
        [*horizontal, *vertical],
        key=lambda line: (line.bbox[1], line.bbox[0], line.orientation),
    )


def _image_bbox_from_matrix(
    matrix: tuple[float, float, float, float, float, float],
    page_bbox: BBox,
    page_rotation: int,
) -> BBox | None:
    """把点阵图单位矩形经对象/Form 矩阵转换并裁剪为页面左上坐标 bbox。"""
    page_points = [
        _transform_drawing_point(
            _apply_pdf_matrix(point, matrix),
            page_bbox,
            page_rotation,
        )
        for point in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    ]
    coordinates = [coordinate for point in page_points for coordinate in point]
    if not all(math.isfinite(value) for value in coordinates):
        return None

    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    left = max(0.0, min(point[0] for point in page_points))
    top = max(0.0, min(point[1] for point in page_points))
    right = min(page_width, max(point[0] for point in page_points))
    bottom = min(page_height, max(point[1] for point in page_points))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _form_bbox_from_object(
    raw_obj: Any,
    page_bbox: BBox,
    page_rotation: int,
) -> BBox | None:
    """读取顶层 Form 的页面坐标边界，并转换、裁剪到视觉页面范围。"""

    left = ctypes.c_float()
    bottom = ctypes.c_float()
    right = ctypes.c_float()
    top = ctypes.c_float()
    try:
        ok = pdfium_c.FPDFPageObj_GetBounds(
            raw_obj,
            ctypes.byref(left),
            ctypes.byref(bottom),
            ctypes.byref(right),
            ctypes.byref(top),
        )
    except Exception:
        return None
    if not ok:
        return None

    raw_bbox = (float(left.value), float(bottom.value), float(right.value), float(top.value))
    if not all(math.isfinite(value) for value in raw_bbox):
        return None
    page_points = [
        _transform_drawing_point(point, page_bbox, page_rotation)
        for point in (
            (raw_bbox[0], raw_bbox[1]),
            (raw_bbox[0], raw_bbox[3]),
            (raw_bbox[2], raw_bbox[1]),
            (raw_bbox[2], raw_bbox[3]),
        )
    ]
    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    visual_left = max(0.0, min(point[0] for point in page_points))
    visual_top = max(0.0, min(point[1] for point in page_points))
    visual_right = min(page_width, max(point[0] for point in page_points))
    visual_bottom = min(page_height, max(point[1] for point in page_points))
    if visual_right <= visual_left or visual_bottom <= visual_top:
        return None
    return visual_left, visual_top, visual_right, visual_bottom


def _validate_pdf_external_link_target(value: str) -> str | None:
    """校验 PDF URI Action，只保留显式且可安全输出的外部链接。"""

    target = value.strip()
    if not target or any(
        ord(char) < 0x20
        or ord(char) == 0x7F
        or 0xD800 <= ord(char) <= 0xDFFF
        for char in target
    ):
        return None
    try:
        parsed = urlsplit(target)
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in _PDF_EXTERNAL_LINK_SCHEMES:
        return None
    if scheme in {"http", "https"}:
        try:
            if not parsed.netloc or parsed.hostname is None:
                return None
        except ValueError:
            return None
    elif not parsed.path:
        return None
    return target


def _get_pdfium_uri_path(raw_doc: Any, raw_action: Any) -> str | None:
    """按 PDFium 两阶段缓冲区协议读取 URI Action 的 UTF-8 目标。"""

    try:
        required = int(
            pdfium_c.FPDFAction_GetURIPath(
                raw_doc,
                raw_action,
                None,
                0,
            )
        )
    except Exception:
        return None
    if required <= 1:
        return None
    try:
        buffer = ctypes.create_string_buffer(required)
        actual = int(
            pdfium_c.FPDFAction_GetURIPath(
                raw_doc,
                raw_action,
                buffer,
                required,
            )
        )
    except Exception:
        return None
    if actual <= 1 or actual > required:
        return None
    raw_value = bytes(buffer.raw[: actual - 1])
    try:
        return raw_value.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _pdf_link_annotation_is_visible(page: pdfium.PdfPage, raw_link: Any) -> bool:
    """读取 Link 注解可见性；损坏或显式隐藏的注解均不参与文本富化。"""

    raw_annot = None
    try:
        raw_annot = pdfium_c.FPDFLink_GetAnnot(page.raw, raw_link)
        if not raw_annot:
            return False
        flags = int(pdfium_c.FPDFAnnot_GetFlags(raw_annot))
    except Exception:
        return False
    finally:
        if raw_annot:
            try:
                pdfium_c.FPDFPage_CloseAnnot(raw_annot)
            except Exception:
                pass
    hidden_flags = (
        pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE
        | pdfium_c.FPDF_ANNOT_FLAG_HIDDEN
        | pdfium_c.FPDF_ANNOT_FLAG_NOVIEW
    )
    return not bool(flags & hidden_flags)


def _visual_bbox_from_pdf_points(
    points: list[tuple[float, float]],
    page_bbox: BBox,
    page_rotation: int,
) -> BBox | None:
    """把 PDF 底左坐标点集转换并裁剪为视觉页面左上坐标 bbox。"""

    if not points or not all(
        math.isfinite(coordinate)
        for point in points
        for coordinate in point
    ):
        return None
    visual_points = [
        _transform_drawing_point(point, page_bbox, page_rotation)
        for point in points
    ]
    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    visual_bbox = (
        max(0.0, min(point[0] for point in visual_points)),
        max(0.0, min(point[1] for point in visual_points)),
        min(page_width, max(point[0] for point in visual_points)),
        min(page_height, max(point[1] for point in visual_points)),
    )
    if visual_bbox[2] <= visual_bbox[0] or visual_bbox[3] <= visual_bbox[1]:
        return None
    return visual_bbox


def _pdf_link_region_bboxes(
    raw_link: Any,
    page_bbox: BBox,
    page_rotation: int,
) -> tuple[BBox, ...]:
    """优先读取 Link QuadPoints，并在缺失时回退到注解矩形。"""

    bboxes: list[BBox] = []
    try:
        quad_count = max(0, int(pdfium_c.FPDFLink_CountQuadPoints(raw_link)))
    except Exception:
        quad_count = 0
    for quad_index in range(quad_count):
        quad = pdfium_c.FS_QUADPOINTSF()
        try:
            ok = pdfium_c.FPDFLink_GetQuadPoints(
                raw_link,
                quad_index,
                ctypes.byref(quad),
            )
        except Exception:
            continue
        if not ok:
            continue
        bbox = _visual_bbox_from_pdf_points(
            [
                (float(quad.x1), float(quad.y1)),
                (float(quad.x2), float(quad.y2)),
                (float(quad.x3), float(quad.y3)),
                (float(quad.x4), float(quad.y4)),
            ],
            page_bbox,
            page_rotation,
        )
        if bbox is not None and bbox not in bboxes:
            bboxes.append(bbox)

    if not bboxes:
        rect = pdfium_c.FS_RECTF()
        try:
            ok = pdfium_c.FPDFLink_GetAnnotRect(raw_link, ctypes.byref(rect))
        except Exception:
            ok = False
        if ok:
            bbox = _visual_bbox_from_pdf_points(
                [
                    (float(rect.left), float(rect.bottom)),
                    (float(rect.left), float(rect.top)),
                    (float(rect.right), float(rect.bottom)),
                    (float(rect.right), float(rect.top)),
                ],
                page_bbox,
                page_rotation,
            )
            if bbox is not None:
                bboxes.append(bbox)
    return tuple(
        sorted(
            bboxes,
            key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]),
        )
    )


def _extract_page_link_annotations(
    page: pdfium.PdfPage,
    raw_doc: Any,
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFLinkAnnotation]:
    """枚举页面外部 URI Link；单条损坏数据不会中断同页其他链接。"""

    annotations: list[PDFLinkAnnotation] = []
    position = ctypes.c_int(0)
    source_index = 0
    while True:
        raw_link = pdfium_c.FPDF_LINK()
        previous_position = int(position.value)
        try:
            found = pdfium_c.FPDFLink_Enumerate(
                page.raw,
                ctypes.byref(position),
                ctypes.byref(raw_link),
            )
        except Exception:
            break
        if not found:
            break
        current_source_index = source_index
        source_index += 1
        if int(position.value) <= previous_position:
            break
        try:
            if not raw_link or not _pdf_link_annotation_is_visible(page, raw_link):
                continue
            raw_action = pdfium_c.FPDFLink_GetAction(raw_link)
            if not raw_action or int(pdfium_c.FPDFAction_GetType(raw_action)) != pdfium_c.PDFACTION_URI:
                continue
            raw_target = _get_pdfium_uri_path(raw_doc, raw_action)
            target = (
                _validate_pdf_external_link_target(raw_target)
                if raw_target is not None
                else None
            )
            if target is None:
                continue
            bboxes = _pdf_link_region_bboxes(
                raw_link,
                page_bbox,
                page_rotation,
            )
            if not bboxes:
                continue
            annotations.append(
                PDFLinkAnnotation(
                    target=target,
                    bboxes=bboxes,
                    source_index=current_source_index,
                )
            )
        except Exception:
            continue
    return annotations


def _get_annotation_string(raw_annot: Any, key: bytes) -> str | None:
    """读取 PDFium 注释字典中的 UTF-16 字符串或名称，异常和空值统一返回 None。"""

    try:
        required = int(pdfium_c.FPDFAnnot_GetStringValue(raw_annot, key, None, 0))
    except Exception:
        return None
    if required <= 2:
        return None
    try:
        buffer = (ctypes.c_ushort * ((required + 1) // 2))()
        actual = int(
            pdfium_c.FPDFAnnot_GetStringValue(
                raw_annot,
                key,
                buffer,
                ctypes.sizeof(buffer),
            )
        )
    except Exception:
        return None
    if actual <= 2 or actual > ctypes.sizeof(buffer):
        return None
    try:
        return bytes(buffer)[: actual - 2].decode("utf-16-le")
    except UnicodeDecodeError:
        return None


def _signature_bbox_from_annotation(
    raw_annot: Any,
    page_bbox: BBox,
    page_rotation: int,
    form_handle: Any | None,
) -> BBox | None:
    """校验一个可见签名 Widget，并把注释矩形裁剪到视觉页面坐标。"""

    try:
        if int(pdfium_c.FPDFAnnot_GetSubtype(raw_annot)) != pdfium_c.FPDF_ANNOT_WIDGET:
            return None
        flags = int(pdfium_c.FPDFAnnot_GetFlags(raw_annot))
    except Exception:
        return None
    hidden_flags = (
        pdfium_c.FPDF_ANNOT_FLAG_INVISIBLE
        | pdfium_c.FPDF_ANNOT_FLAG_HIDDEN
        | pdfium_c.FPDF_ANNOT_FLAG_NOVIEW
    )
    if flags & hidden_flags:
        return None
    field_type: int | None = None
    if form_handle is not None:
        try:
            field_type = int(
                pdfium_c.FPDFAnnot_GetFormFieldType(form_handle, raw_annot)
            )
        except Exception:
            field_type = None
    if field_type == pdfium_c.FPDF_FORMFIELD_SIGNATURE:
        pass
    elif field_type is None or field_type <= pdfium_c.FPDF_FORMFIELD_UNKNOWN:
        if _get_annotation_string(raw_annot, b"FT") != "Sig":
            return None
    else:
        return None

    # 正常外观长度只有 UTF-16 终止符时等价于无 /AP /N，不能形成可见签名。
    try:
        appearance_length = int(
            pdfium_c.FPDFAnnot_GetAP(
                raw_annot,
                pdfium_c.FPDF_ANNOT_APPEARANCEMODE_NORMAL,
                None,
                0,
            )
        )
    except Exception:
        return None
    if appearance_length <= 2:
        return None

    rect = pdfium_c.FS_RECTF()
    try:
        if not pdfium_c.FPDFAnnot_GetRect(raw_annot, ctypes.byref(rect)):
            return None
    except Exception:
        return None
    raw_bbox = _normalize_pdf_page_bbox(
        (float(rect.left), float(rect.bottom), float(rect.right), float(rect.top))
    )
    if not all(math.isfinite(value) for value in raw_bbox):
        return None
    page_points = [
        _transform_drawing_point(point, page_bbox, page_rotation)
        for point in (
            (raw_bbox[0], raw_bbox[1]),
            (raw_bbox[0], raw_bbox[3]),
            (raw_bbox[2], raw_bbox[1]),
            (raw_bbox[2], raw_bbox[3]),
        )
    ]
    page_width, page_height = _drawing_page_size(page_bbox, page_rotation)
    visual_bbox = (
        max(0.0, min(point[0] for point in page_points)),
        max(0.0, min(point[1] for point in page_points)),
        min(page_width, max(point[0] for point in page_points)),
        min(page_height, max(point[1] for point in page_points)),
    )
    if visual_bbox[2] <= visual_bbox[0] or visual_bbox[3] <= visual_bbox[1]:
        return None
    return visual_bbox


def _extract_page_signature_bboxes(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
    *,
    form_handle: Any | None = None,
) -> list[BBox]:
    """遍历页面签名注释；逐个关闭句柄，并隔离损坏注释造成的异常。"""

    try:
        annot_count = max(0, int(pdfium_c.FPDFPage_GetAnnotCount(page.raw)))
    except Exception:
        return []
    signature_bboxes: list[BBox] = []
    for annot_index in range(annot_count):
        raw_annot = None
        try:
            raw_annot = pdfium_c.FPDFPage_GetAnnot(page.raw, annot_index)
            if not raw_annot:
                continue
            signature_bbox = _signature_bbox_from_annotation(
                raw_annot,
                page_bbox,
                page_rotation,
                form_handle,
            )
            if signature_bbox is not None:
                signature_bboxes.append(signature_bbox)
        except Exception:
            # 单个损坏注释不能影响同页其他有效签名框。
            continue
        finally:
            if raw_annot:
                try:
                    pdfium_c.FPDFPage_CloseAnnot(raw_annot)
                except Exception:
                    pass
    return sorted(signature_bboxes, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _extract_page_image_bboxes(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
) -> list[BBox]:
    """在调用方持有 PDFium 锁时提取全部有效点阵图 bbox，并隔离单对象异常。"""
    image_bboxes: list[BBox] = []
    for _raw_obj, matrix in _iter_raw_image_objects(page):
        try:
            image_bbox = _image_bbox_from_matrix(matrix, page_bbox, page_rotation)
        except Exception:
            # 单个损坏 Image 对象不能中断同页其他点阵图提取。
            continue
        if image_bbox is not None:
            image_bboxes.append(image_bbox)
    return sorted(image_bboxes, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _get_raw_image_fingerprint(raw_obj: Any, page: pdfium.PdfPage) -> str | None:
    """读取受限大小的图片原始流，并结合像素宽高生成 SHA-256 指纹。"""

    metadata = pdfium_c.FPDF_IMAGEOBJ_METADATA()
    try:
        if not pdfium_c.FPDFImageObj_GetImageMetadata(
            raw_obj,
            page.raw,
            ctypes.byref(metadata),
        ):
            return None
        width = int(metadata.width)
        height = int(metadata.height)
        raw_size = int(pdfium_c.FPDFImageObj_GetImageDataRaw(raw_obj, None, 0))
    except Exception:
        return None
    if (
        width <= 0
        or height <= 0
        or raw_size <= 0
        or raw_size > PDF_IMAGE_FINGERPRINT_MAX_RAW_BYTES
    ):
        return None

    try:
        buffer = ctypes.create_string_buffer(raw_size)
    except (MemoryError, OverflowError):
        return None
    try:
        bytes_read = int(pdfium_c.FPDFImageObj_GetImageDataRaw(raw_obj, buffer, raw_size))
    except Exception:
        return None
    if bytes_read <= 0 or bytes_read > raw_size:
        return None

    digest = hashlib.sha256()
    digest.update(f"{width}:{height}:".encode("ascii"))
    digest.update(buffer.raw[:bytes_read])
    return digest.hexdigest()


def _extract_page_image_infos(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFImageInfo]:
    """提取全部有效点阵图几何；单图指纹失败时保留 bbox 并按放行处理。"""

    image_infos: list[PDFImageInfo] = []
    for raw_obj, matrix in _iter_raw_image_objects(page):
        try:
            image_bbox = _image_bbox_from_matrix(matrix, page_bbox, page_rotation)
        except Exception:
            # 单个损坏 Image 对象不能中断同页其他点阵图提取。
            continue
        if image_bbox is None:
            continue
        image_infos.append(
            PDFImageInfo(
                bbox=image_bbox,
                fingerprint=_get_raw_image_fingerprint(raw_obj, page),
            )
        )
    return sorted(
        image_infos,
        key=lambda info: (info.bbox[1], info.bbox[0], info.bbox[3], info.bbox[2]),
    )


def _extract_page_form_bboxes(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
) -> list[BBox]:
    """在调用方持有 PDFium 锁时提取顶层 Form bbox，并隔离单对象异常。"""

    form_bboxes: list[BBox] = []
    for raw_obj in _iter_raw_root_form_objects(page):
        try:
            form_bbox = _form_bbox_from_object(raw_obj, page_bbox, page_rotation)
        except Exception:
            # PDFium 遇到个别损坏 Form 时，保留同页其他有效结果。
            continue
        if form_bbox is not None:
            form_bboxes.append(form_bbox)
    return sorted(form_bboxes, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _extract_page_path_infos(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFPathInfo]:
    """在调用方持有 PDFium 锁时提取 Path 信息，并隔离单对象异常。"""

    path_infos: list[PDFPathInfo] = []
    for source_index, (raw_obj, matrix, form_depth) in enumerate(
        _iter_raw_path_objects_with_depth(page)
    ):
        try:
            path_info = _path_info_from_object(
                raw_obj,
                matrix,
                page_bbox,
                page_rotation,
                form_depth,
                source_index,
            )
        except Exception:
            # PDFium 遇到个别损坏 Path 时，保留同页其他对象的有效结果。
            continue
        if path_info is not None:
            path_infos.append(path_info)
    return path_infos


def _extract_page_drawing_lines(
    page: pdfium.PdfPage,
    page_bbox: BBox,
    page_rotation: int,
) -> list[PDFDrawingLine]:
    """在调用方持有 PDFium 锁时提取整页绘图线，并隔离单个对象异常。"""
    drawing_lines: list[PDFDrawingLine] = []
    for raw_obj, matrix in _iter_raw_path_objects(page):
        try:
            drawing_lines.extend(
                _extract_path_drawing_lines(
                    raw_obj,
                    matrix,
                    page_bbox,
                    page_rotation,
                )
            )
        except Exception:
            # PDFium 遇到个别损坏 Path 时，保留同页其他对象的有效结果。
            continue
    return _merge_collinear_drawing_lines(
        drawing_lines,
        _drawing_page_size(page_bbox, page_rotation),
    )


def _try_close(obj: object) -> None:
    if callable(close := getattr(obj, "close", None)):
        try:
            close()
        except Exception:
            pass


def _get_visible_char_signature(
    char: Char,
) -> tuple[str, tuple[Any, Any, Any, Any], float]:
    """生成可见字符去重签名，不把 bbox 放入签名以便单独做近重合判断。"""
    font = char.get("font") or {}
    font_key = (
        font.get("name"),
        font.get("flags"),
        font.get("size"),
        font.get("weight"),
    )
    rotation_key = round(float(char.get("rotation") or 0.0), 3)
    return str(char.get("char", "")), font_key, rotation_key


def _is_near_identical_bbox(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> bool:
    """判断两个字符 bbox 是否属于同一视觉位置的一点内抖动。"""
    return all(
        abs(coord_a - coord_b) <= NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        for coord_a, coord_b in zip(bbox_a, bbox_b)
    )


def _calculate_bbox_overlap_in_smaller_area(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    """计算两个字符框交集占较小字符框面积的比例。"""
    intersection_width = max(
        0.0,
        min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]),
    )
    intersection_height = max(
        0.0,
        min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]),
    )
    bbox_a_area = max(0.0, bbox_a[2] - bbox_a[0]) * max(
        0.0,
        bbox_a[3] - bbox_a[1],
    )
    bbox_b_area = max(0.0, bbox_b[2] - bbox_b[0]) * max(
        0.0,
        bbox_b[3] - bbox_b[1],
    )
    smaller_area = min(bbox_a_area, bbox_b_area)
    if smaller_area == 0:
        return 0.0
    return intersection_width * intersection_height / smaller_area


def _is_adjacent_offset_duplicate_char(
    previous_char: Char,
    current_char: Char,
) -> bool:
    """识别相邻字符中由对角平移阴影产生的第二个重复字符。"""
    if _get_visible_char_signature(previous_char) != _get_visible_char_signature(current_char):
        return False

    previous_bbox = _char_bbox_values(previous_char.get("bbox"))
    current_bbox = _char_bbox_values(current_char.get("bbox"))
    if previous_bbox is None or current_bbox is None:
        return False

    x_start_offset = current_bbox[0] - previous_bbox[0]
    y_start_offset = current_bbox[1] - previous_bbox[1]
    x_end_offset = current_bbox[2] - previous_bbox[2]
    y_end_offset = current_bbox[3] - previous_bbox[3]

    # 阴影层应是同一字符框的刚性平移，避免把大小不同的相邻同字误判为重复。
    if (
        abs(x_start_offset - x_end_offset) > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
        or abs(y_start_offset - y_end_offset) > OFFSET_DUPLICATE_TRANSLATION_TOLERANCE
    ):
        return False

    if not (
        NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        < abs(x_start_offset)
        <= OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE
        and NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE
        < abs(y_start_offset)
        <= OFFSET_DUPLICATE_CHAR_BBOX_TOLERANCE
    ):
        return False

    return (
        _calculate_bbox_overlap_in_smaller_area(previous_bbox, current_bbox)
        >= OFFSET_DUPLICATE_MIN_BBOX_OVERLAP_RATIO
    )


def _get_near_identical_bbox_bucket_key(
    bbox_coords: tuple[float, float, float, float],
) -> tuple[int, int]:
    """按字符 bbox 左上角生成空间桶 key，缩小近重合判断的候选范围。"""
    return (
        math.floor(bbox_coords[0] / NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE),
        math.floor(bbox_coords[1] / NEAR_IDENTICAL_CHAR_BBOX_TOLERANCE),
    )


def _iter_neighbor_bbox_bucket_keys(
    bucket_key: tuple[int, int],
) -> Iterator[tuple[int, int]]:
    """遍历当前桶及周围 8 个邻近桶，覆盖 bbox 容差范围内的候选字符。"""
    bucket_x, bucket_y = bucket_key
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            yield bucket_x + offset_x, bucket_y + offset_y


def _deduplicate_near_identical_chars(chars: list[Char]) -> list[Char]:
    """清理 PDFium 文本层边界处同字符、同位置及对角阴影重复字符。"""
    seen_visible_char_bboxes: dict[
        tuple[str, tuple[Any, Any, Any, Any], float],
        dict[tuple[int, int], list[tuple[float, float, float, float]]],
    ] = {}
    deduplicated_chars: list[Char] = []

    for char in chars:
        text = str(char.get("char", ""))
        if not text or text.isspace():
            deduplicated_chars.append(char)
            continue

        visible_char_key = _get_visible_char_signature(char)
        bbox_coords = _char_bbox_values(char.get("bbox"))
        if bbox_coords is None:
            deduplicated_chars.append(char)
            continue

        if deduplicated_chars and _is_adjacent_offset_duplicate_char(
            deduplicated_chars[-1],
            char,
        ):
            continue

        bbox_bucket_key = _get_near_identical_bbox_bucket_key(bbox_coords)
        visible_char_bbox_buckets = seen_visible_char_bboxes.setdefault(
            visible_char_key,
            {},
        )
        if any(
            _is_near_identical_bbox(bbox_coords, seen_bbox)
            for neighbor_bucket_key in _iter_neighbor_bbox_bucket_keys(bbox_bucket_key)
            for seen_bbox in visible_char_bbox_buckets.get(neighbor_bucket_key, [])
        ):
            continue

        visible_char_bbox_buckets.setdefault(bbox_bucket_key, []).append(bbox_coords)
        deduplicated_chars.append(char)

    return deduplicated_chars


def _is_pdftext_page_chars(chars: Any) -> bool:
    """判断对象是否为 pdftext 0.7 引入的 PageChars 列式字符容器。"""
    return PageChars is not None and isinstance(chars, PageChars)


def _deduplicate_pdftext_chars(chars: Any) -> Any:
    """按当前 pdftext 返回类型调用官方去重，兼容测试或旧版本的 list 字符。"""
    if _is_pdftext_page_chars(chars) or PageChars is None:
        return deduplicate_chars(chars)
    return chars


def _materialize_page_chars(chars: Any) -> list[Char]:
    """将 pdftext 0.7 的 PageChars 物化为 MinerU 既有 char dict 列表。"""
    boxes = chars.boxes.tolist()
    rotations = chars.rotations.tolist()
    font_ids = chars.font_ids.tolist()
    char_indices = chars.char_indices.tolist()

    return [
        cast(
            Char,
            {
                "bbox": Bbox([float(coord) for coord in boxes[index]]),
                "char": chars.text[index],
                "rotation": float(rotations[index]),
                "font": chars.fonts[int(font_ids[index])],
                "char_idx": int(char_indices[index]),
            },
        )
        for index in range(len(chars))
    ]


def _ensure_legacy_chars(chars: Any) -> list[Char]:
    """统一输出旧版 char dict 列表，隔离 pdftext 0.7 的返回结构变化。"""
    if _is_pdftext_page_chars(chars):
        return _materialize_page_chars(chars)
    return cast(list[Char], chars)


def _restore_pdfium_surrogate_pairs(
    chars: list[Char],
    textpage: pdfium.PdfTextPage,
) -> list[Char]:
    """利用 PDFium 原始 UTF-16 code unit 恢复 pdftext 丢失的补充平面字符。"""
    if not any(
        len(text := str(char.get("char", ""))) == 1
        and (text == "\uFFFD" or 0xD800 <= ord(text) <= 0xDFFF)
        for char in chars
    ):
        return chars

    try:
        textpage_raw = textpage.raw
        char_count = int(textpage.count_chars())
    except Exception:
        textpage_raw = None
        char_count = 0

    restored_chars: list[Char] = []
    consumed_char_indices: set[int] = set()
    get_unicode = pdfium_c.FPDFText_GetUnicode

    for char in chars:
        text = str(char.get("char", ""))
        raw_char_idx = char.get("char_idx")
        try:
            char_idx = int(raw_char_idx) if raw_char_idx is not None else -1
        except (TypeError, ValueError):
            char_idx = -1

        if char_idx in consumed_char_indices:
            continue

        raw_code = None
        if (
            textpage_raw is not None
            and 0 <= char_idx < char_count
            and len(text) == 1
            and (text == "\uFFFD" or 0xD800 <= ord(text) <= 0xDFFF)
        ):
            raw_code = int(get_unicode(textpage_raw, char_idx))

        high_surrogate = None
        low_surrogate = None
        if raw_code is not None and 0xD800 <= raw_code <= 0xDBFF and char_idx + 1 < char_count:
            next_code = int(get_unicode(textpage_raw, char_idx + 1))
            if 0xDC00 <= next_code <= 0xDFFF:
                high_surrogate = raw_code
                low_surrogate = next_code
                consumed_char_indices.add(char_idx + 1)
        elif raw_code is not None and 0xDC00 <= raw_code <= 0xDFFF and char_idx > 0:
            previous_code = int(get_unicode(textpage_raw, char_idx - 1))
            if 0xD800 <= previous_code <= 0xDBFF:
                high_surrogate = previous_code
                low_surrogate = raw_code

        if high_surrogate is not None and low_surrogate is not None:
            restored_char = cast(Char, dict(char))
            restored_char["char"] = chr(
                0x10000
                + ((high_surrogate - 0xD800) << 10)
                + (low_surrogate - 0xDC00)
            )
            restored_chars.append(restored_char)
            continue

        if len(text) == 1 and 0xD800 <= ord(text) <= 0xDFFF:
            restored_char = cast(Char, dict(char))
            restored_char["char"] = "\uFFFD"
            restored_chars.append(restored_char)
            continue

        restored_chars.append(char)

    return restored_chars


def _char_bbox_values(bbox: object) -> tuple[float, float, float, float] | None:
    """将 tuple/list 或 pdftext Bbox 对象统一转换为四元组坐标。"""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    attrs = ("x_start", "y_start", "x_end", "y_end")
    if all(hasattr(bbox, attr) for attr in attrs):
        return tuple(float(getattr(bbox, attr)) for attr in attrs)  # type: ignore[return-value]
    return None


def _get_single_char_text(char: Char) -> str:
    """提取单个 PDF 字符文本，异常空值用替换符保证 PageChars 长度一致。"""
    text = str(char.get("char", ""))
    if len(text) == 1:
        return text
    return text[:1] or "\uFFFD"


def _get_char_font_id(
    char: Char,
    fonts: list[dict[str, Any]],
    font_cache: dict[tuple[Any, Any, Any, Any], int],
) -> int:
    """为旧版字符 font 生成 PageChars 需要的页内 font id。"""
    font = char.get("font") or {}
    font_key = (
        font.get("name"),
        font.get("flags"),
        font.get("size"),
        font.get("weight"),
    )
    font_id = font_cache.get(font_key)
    if font_id is None:
        font_id = len(fonts)
        font_cache[font_key] = font_id
        fonts.append(
            {
                "name": font.get("name"),
                "flags": font.get("flags"),
                "size": font.get("size"),
                "weight": font.get("weight"),
            }
        )
    return font_id


def _get_char_index(char: Char, fallback_idx: int) -> int:
    """提取旧版字符索引，缺失或为空时回退到当前列表位置。"""
    char_idx = char.get("char_idx")
    if char_idx is None:
        char_idx = fallback_idx
    return int(char_idx)


def _legacy_chars_to_page_chars(chars: Any) -> Any:
    """将旧版 char dict 列表打包回 pdftext 0.7 get_spans 所需的 PageChars。"""
    if PageChars is None or _is_pdftext_page_chars(chars):
        return chars

    fonts: list[dict[str, Any]] = []
    font_cache: dict[tuple[Any, Any, Any, Any], int] = {}
    text_parts: list[str] = []
    codes: list[int] = []
    rotations: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    font_ids: list[int] = []
    char_indices: list[int] = []

    for fallback_idx, char in enumerate(cast(list[Char], chars)):
        char_text = _get_single_char_text(char)
        bbox_values = _char_bbox_values(char.get("bbox"))
        if bbox_values is None:
            bbox_values = (0.0, 0.0, 0.0, 0.0)
        text_parts.append(char_text)
        codes.append(ord(char_text))
        rotations.append(float(char.get("rotation") or 0.0))
        boxes.append(bbox_values)
        font_ids.append(_get_char_font_id(char, fonts, font_cache))
        char_indices.append(_get_char_index(char, fallback_idx))

    return PageChars(
        "".join(text_parts),
        np.array(codes, dtype=np.uint32),
        np.array(rotations, dtype=np.float64),
        np.array(boxes, dtype=np.float64).reshape((len(boxes), 4)),
        np.array(font_ids, dtype=np.int32),
        fonts,
        np.array(char_indices, dtype=np.int64),
    )


def get_lines_from_chars(
    chars: list[Char],
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> list[Line]:
    """从已提取的字符构建 pdftext lines，避免重复读取 PDFium textpage。"""
    chars = _legacy_chars_to_page_chars(chars)
    spans = get_spans(
        chars,
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    lines = get_lines(spans)
    assign_scripts(
        lines,
        height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    return lines


def images_bytes_to_pdf_bytes(image_bytes: bytes) -> bytes:
    # 载入并转换所有图像为 RGB 模式
    image = Image.open(BytesIO(image_bytes))
    # 根据 EXIF 信息自动转正（处理手机拍摄的带 Orientation 标记的图片）
    image = ImageOps.exif_transpose(image) or image

    # 只在必要时转换
    if image.mode != "RGB":
        image = image.convert("RGB")

    with BytesIO() as pdf_buffer:
        # 第一张图保存为 PDF，其余追加
        image.save(
            pdf_buffer,
            format="PDF",
            resolution=DEFAULT_RENDER_DPI,
            quality=95,
            subsampling=0,
        )
        return pdf_buffer.getvalue()


def _page_to_image(page: pdfium.PdfPage, scale: float, max_edge: int) -> PDFPageImage:
    long_edge_length = max(*page.get_size())
    if (long_edge_length * scale) > max_edge:
        scale = max_edge / long_edge_length

    bitmap = None
    try:
        bitmap = page.render(scale=scale)  # type: ignore
        bitmap = cast(pdfium.PdfBitmap, bitmap)
        pil_image = bitmap.to_pil()
    finally:
        _try_close(bitmap)

    return PDFPageImage(pil_image=pil_image, scale=scale)
