# Copyright (c) Opendatalab. All rights reserved.
"""Flash Office 文档的图片识别、转码与占位图生成。"""

from functools import lru_cache
from io import BytesIO
from pathlib import PurePosixPath
import struct
from typing import Final

from loguru import logger
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from ....utils.platform import is_windows_environment
from .._shared.image import image_to_b64str

VECTOR_IMAGE_FORMATS = frozenset({"WMF", "EMF"})
VECTOR_IMAGE_EXTENSIONS = frozenset({".wmf", ".emf"})
VECTOR_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/x-wmf",
        "image/wmf",
        "image/x-emf",
        "image/emf",
        "application/x-msmetafile",
    }
)
PIL_IMAGE_LOAD_ERRORS = (UnidentifiedImageError, OSError, SyntaxError)
STANDARD_VECTOR_PLACEHOLDER_SIZE: Final = (320, 180)
STANDARD_VECTOR_PLACEHOLDER_LINES: Final = (
    "WMF/EMF placeholder",
    "Use Windows to parse",
    "the original image",
)


def _is_wmf_payload(image_data: bytes) -> bool:
    """根据 placeable 或标准 METAHEADER magic 判断原始载荷是否为 WMF。"""

    return image_data.startswith(b"\xd7\xcd\xc6\x9a") or (
        len(image_data) >= 4
        and image_data[:2] in {b"\x01\x00", b"\x02\x00"}
        and image_data[2:4] == b"\x09\x00"
    )


def is_vector_image(pil_image: Image.Image) -> bool:
    """判断已由 Pillow 打开的图片是否属于 WMF/EMF 矢量格式。"""
    return (getattr(pil_image, "format", None) or "").upper() in VECTOR_IMAGE_FORMATS


def is_vector_image_part(part_name: object | None = None, content_type: str | None = None) -> bool:
    """根据 OOXML 部件扩展名和内容类型判断是否为矢量图片。"""
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    if suffix in VECTOR_IMAGE_EXTENSIONS:
        return True
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    return normalized_content_type in VECTOR_IMAGE_CONTENT_TYPES


def is_valid_vector_image_payload(
    image_data: bytes,
    *,
    part_name: object | None = None,
    content_type: str | None = None,
) -> bool:
    """校验 WMF/EMF 最小文件签名，避免为任意伪装字节生成矢量占位图。"""
    label = _vector_image_format_label(part_name, content_type)
    if label == "WMF":
        return _is_wmf_payload(image_data)
    if label == "EMF":
        return (
            len(image_data) >= 44
            and struct.unpack_from("<I", image_data, 0)[0] == 1
            and image_data[40:44] == b" EMF"
        )
    return _is_wmf_payload(image_data) or (
        len(image_data) >= 44 and image_data[40:44] == b" EMF"
    )


def _vector_image_format_label(part_name: object | None = None, content_type: str | None = None) -> str:
    """从 OOXML 部件信息推断用于日志和占位图的矢量格式名称。"""
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    normalized_content_type = (content_type or "").lower()
    if suffix == ".wmf" or "wmf" in normalized_content_type:
        return "WMF"
    if suffix == ".emf" or "emf" in normalized_content_type:
        return "EMF"
    return "WMF/EMF"


def _load_placeholder_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按优先级加载占位图字体，均不可用时退回 Pillow 默认字体。"""
    for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_text_placeholder(size: tuple[int, int], lines: list[str]) -> Image.Image:
    """按目标尺寸绘制带边框和居中文案的 RGB 占位图。"""
    width = max(int(size[0]), 1)
    height = max(int(size[1]), 1)
    placeholder = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(placeholder)

    border_width = max(1, min(width, height) // 80)
    draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline=(190, 190, 190),
        width=border_width,
    )

    max_text_width = max(width - 16, 1)
    max_text_height = max(height - 16, 1)
    fallback_text = "WMF/EMF"
    text = "\n".join(line for line in lines if line)
    if not text:
        text = fallback_text

    font = None
    spacing = 4
    bbox = None
    for font_size in range(max(min(width, height) // 7, 10), 7, -1):
        font = _load_placeholder_font(font_size)
        spacing = max(2, font_size // 4)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= max_text_width and text_height <= max_text_height:
            break
    else:
        text = fallback_text
        font = _load_placeholder_font(max(min(width, height) // 5, 10))
        spacing = 2
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    origin = ((width - text_width) / 2, (height - text_height) / 2)
    draw.multiline_text(
        origin,
        text,
        fill=(90, 90, 90),
        font=font,
        spacing=spacing,
        align="center",
    )
    return placeholder


@lru_cache(maxsize=1)
def _standard_vector_placeholder_data_uri() -> str:
    """生成并缓存标准 WMF/EMF 占位图，避免每张矢量图重复绘制。"""
    placeholder = create_text_placeholder(
        STANDARD_VECTOR_PLACEHOLDER_SIZE,
        list(STANDARD_VECTOR_PLACEHOLDER_LINES),
    )
    return image_to_b64str(placeholder, image_format="JPEG")


def get_standard_vector_placeholder_data_uri() -> str:
    """返回标准 WMF/EMF 占位图 data URI，供 Office 各格式复用。"""
    return _standard_vector_placeholder_data_uri()


def serialize_vector_image_with_placeholder(pil_image: Image.Image, image_format_override: str | None = None) -> str:
    """在支持的平台序列化矢量图，失败或不支持时返回标准占位图。"""
    image_format = (image_format_override or getattr(pil_image, "format", None) or "WMF/EMF").upper()

    if is_windows_environment():
        try:
            pil_image.load()
            return image_to_b64str(pil_image, image_format="PNG")
        except PIL_IMAGE_LOAD_ERRORS as e:
            logger.warning(f"Failed to render {image_format} image: {e}, size: {pil_image.size}. Using placeholder instead.")
    else:
        logger.warning(f"Skipping {image_format} image on non-Windows environment, size: {pil_image.size}")

    return get_standard_vector_placeholder_data_uri()


def serialize_vector_part_with_placeholder(
    part_name: object | None = None,
    content_type: str | None = None,
    size: tuple[int, int] = (320, 180),
) -> str:
    """跳过未加载的矢量部件并返回可嵌入文档的标准占位图。"""
    image_format = _vector_image_format_label(part_name, content_type)
    logger.debug(
        f"Skipping {image_format} image part before Pillow load, "
        f"part_name={part_name}, content_type={content_type}, requested_size={size}"
    )
    return get_standard_vector_placeholder_data_uri()


def serialize_office_image(
    image_data: bytes,
    *,
    part_name: object | None = None,
    content_type: str | None = None,
) -> str | None:
    """识别并序列化 Office 图片，透明图保留 PNG，普通位图优先使用 JPEG。"""
    is_wmf_payload = _is_wmf_payload(image_data)
    if is_wmf_payload:
        part_name = "image.wmf"
        content_type = "image/wmf"
    if is_wmf_payload or is_vector_image_part(part_name, content_type):
        if not is_windows_environment():
            return serialize_vector_part_with_placeholder(part_name, content_type)

        try:
            pil_image = Image.open(BytesIO(image_data))
        except PIL_IMAGE_LOAD_ERRORS as e:
            logger.warning(
                f"Warning: vector image cannot be opened by Pillow: {e}, "
                f"part_name={part_name}, content_type={content_type}. "
                "Using placeholder instead."
            )
            return serialize_vector_part_with_placeholder(part_name, content_type)

        return serialize_vector_image_with_placeholder(
            pil_image,
            image_format_override=_vector_image_format_label(part_name, content_type),
        )

    try:
        pil_image = Image.open(BytesIO(image_data))
        pil_image.load()
    except PIL_IMAGE_LOAD_ERRORS as e:
        logger.warning(f"Warning: image cannot be loaded by Pillow: {e}, part_name={part_name}, content_type={content_type}")
        return None

    if is_vector_image(pil_image):
        return serialize_vector_image_with_placeholder(pil_image)

    if pil_image.mode == "RGB":
        return image_to_b64str(pil_image, image_format="JPEG")

    if pil_image.mode in {"RGBA", "LA"} or (pil_image.mode == "P" and "transparency" in pil_image.info):
        return image_to_b64str(pil_image.convert("RGBA"), image_format="PNG")

    return image_to_b64str(pil_image.convert("RGB"), image_format="JPEG")


def ensure_bmp_header(image_data: bytes) -> bytes:
    """为裸 DIB 补齐 BMP 文件头，无法可靠推断时保留原始载荷。"""
    if image_data.startswith(b"BM") or len(image_data) < 4:
        return image_data
    header_size = int(struct.unpack_from("<I", image_data, 0)[0])
    if header_size < 12 or header_size > len(image_data):
        return image_data
    palette_bytes = 0
    if header_size >= 40 and len(image_data) >= 36:
        bit_count = int(struct.unpack_from("<H", image_data, 14)[0])
        colors_used = int(struct.unpack_from("<I", image_data, 32)[0])
        colors = colors_used or ((1 << bit_count) if bit_count <= 8 else 0)
        palette_bytes = colors * 4
    pixel_offset = min(14 + header_size + palette_bytes, 14 + len(image_data))
    return b"BM" + struct.pack("<IHHI", 14 + len(image_data), 0, 0, pixel_offset) + image_data
