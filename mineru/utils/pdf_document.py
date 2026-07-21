# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Iterator, Literal, TypeAlias, cast

import numpy as np
import pypdfium2 as pdfium
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_lines, get_spans
from pdftext.schema import Bbox, Char, Line
from PIL import Image, ImageOps

from ..types import BBox, PageInfo
from .draw_bbox import draw_layout_bbox, draw_span_bbox
from .pdf_classify import classify, get_sample_page_indices
from .pdf_image_tools import get_crop_img, load_images_from_pdf_bytes_range
from .pdf_reader import image_to_bytes
from .pdfium_guard import _pdfium_lock, safe_rewrite_pdf_bytes_with_pdfium

POINTS_PER_INCH: int = 72
DEFAULT_RENDER_DPI: int = 200
DEFAULT_RENDER_SCALE: float = DEFAULT_RENDER_DPI / POINTS_PER_INCH
DEFAULT_RENDER_MAX_EDGE: int = 3500
NEAR_DUPLICATE_CHAR_BBOX_TOLERANCE = 0.5

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


class PDFPage:
    def __init__(self, pdf_doc: "PDFDocument", idx: int) -> None:
        self.pdf_doc = pdf_doc
        self._idx = idx

    @property
    def size(self) -> tuple[float, float]:
        return self.pdf_doc.page_size(self._idx)

    def get_char_count(self) -> int:
        return self.pdf_doc.page_char_count(self._idx)

    def get_chars(self) -> list[Char]:
        return self.pdf_doc.get_page_chars(self._idx)


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
    def from_image(
        image_bytes: bytes,
        render_scale: float = DEFAULT_RENDER_SCALE,
        render_max_edge: int = DEFAULT_RENDER_MAX_EDGE,
    ) -> "PDFDocument":
        image = Image.open(BytesIO(image_bytes))
        # 根据 EXIF 信息自动转正（处理手机拍摄的带 Orientation 标记的图片）
        image = ImageOps.exif_transpose(image) or image

        # 只在必要时转换
        if image.mode != "RGB":
            image = image.convert("RGB")

        render_dpi = max(1, int(round(render_scale * POINTS_PER_INCH)))

        with BytesIO() as pdf_buffer:
            # 第一张图保存为 PDF，其余追加
            image.save(
                pdf_buffer,
                format="PDF",
                resolution=render_dpi,
                quality=95,
                subsampling=0,
            )
            pdf_bytes = pdf_buffer.getvalue()

        return PDFDocument(
            pdf_bytes,
            render_scale=render_scale,
            render_max_edge=render_max_edge,
        )

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
        return (abs(rect[2] - rect[0]), abs(rect[1] - rect[3]))

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
    def crop_image(self, bbox: BBox, page_idx: int, *, scale: float = 2) -> bytes:
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
            finally:
                _try_close(textpage)
        chars = _deduplicate_pdftext_chars(chars)
        chars = _ensure_legacy_chars(chars)
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
        out_dir = os.path.dirname(output_path) or "."
        filename = os.path.basename(output_path)
        draw_layout_bbox(pages, self._pdf_bytes, out_dir, filename)

    def draw_span_bbox(self, pages: list[PageInfo], output_path: str) -> None:
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
                    self._pdf_doc_opened = pdfium.PdfDocument(self._pdf_bytes)
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


def _try_close(obj: object) -> None:
    if callable(close := getattr(obj, "close", None)):
        try:
            close()
        except Exception:
            pass


def _deduplicate_near_identical_chars(chars: list[Char]) -> list[Char]:
    """移除坐标近乎重合的重复可见字符，降低 PDF 叠印文本造成的重复抽取。"""
    unique_chars: list[Char] = []
    seen_keys: set[tuple[str, int, int, int, int, str, str]] = set()
    for char in chars:
        text = str(char.get("char", ""))
        if not text or text.isspace():
            unique_chars.append(char)
            continue

        bbox_values = _char_bbox_values(char.get("bbox"))
        if bbox_values is None:
            unique_chars.append(char)
            continue

        rounded_bbox = tuple(int(round(float(value) / NEAR_DUPLICATE_CHAR_BBOX_TOLERANCE)) for value in bbox_values)
        key = (
            text,
            *rounded_bbox,
            str(char.get("font", "")),
            str(char.get("rotation", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_chars.append(char)
    return unique_chars


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
    return text[:1] or "\ufffd"


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
