# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from loguru import logger

from mineru.utils.check_sys_env import is_windows_environment
from mineru.utils.pdf_reader import image_to_b64str


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


def is_vector_image(pil_image: Image.Image) -> bool:
    return (getattr(pil_image, "format", None) or "").upper() in VECTOR_IMAGE_FORMATS


def is_vector_image_part(
    part_name: object | None = None, content_type: str | None = None
) -> bool:
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    if suffix in VECTOR_IMAGE_EXTENSIONS:
        return True
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
    return normalized_content_type in VECTOR_IMAGE_CONTENT_TYPES


def _vector_image_format_label(
    part_name: object | None = None, content_type: str | None = None
) -> str:
    suffix = PurePosixPath(str(part_name or "")).suffix.lower()
    normalized_content_type = (content_type or "").lower()
    if suffix == ".wmf" or "wmf" in normalized_content_type:
        return "WMF"
    if suffix == ".emf" or "emf" in normalized_content_type:
        return "EMF"
    return "WMF/EMF"


def _load_placeholder_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in (
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_text_placeholder(
    size: tuple[int, int], lines: list[str]
) -> Image.Image:
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
        bbox = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center"
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= max_text_width and text_height <= max_text_height:
            break
    else:
        text = fallback_text
        font = _load_placeholder_font(max(min(width, height) // 5, 10))
        spacing = 2
        bbox = draw.multiline_textbbox(
            (0, 0), text, font=font, spacing=spacing, align="center"
        )

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


def serialize_vector_image_with_placeholder(
    pil_image: Image.Image, image_format_override: str | None = None
) -> str:
    image_format = (
        image_format_override or getattr(pil_image, "format", None) or "WMF/EMF"
    ).upper()

    if is_windows_environment():
        try:
            pil_image.load()
            return image_to_b64str(pil_image, image_format="PNG")
        except OSError as e:
            logger.warning(
                f"Failed to render {image_format} image: {e}, size: {pil_image.size}. Using placeholder instead."
            )
            placeholder_lines = [
                f"{image_format} placeholder",
                "Windows rendering failed",
            ]
    else:
        logger.warning(
            f"Skipping {image_format} image on non-Windows environment, size: {pil_image.size}"
        )
        placeholder_lines = [
            f"{image_format} placeholder",
            "Use Windows to parse",
            "the original image",
        ]

    placeholder = create_text_placeholder(pil_image.size, placeholder_lines)
    return image_to_b64str(placeholder, image_format="JPEG")


def serialize_vector_part_with_placeholder(
    part_name: object | None = None,
    content_type: str | None = None,
    size: tuple[int, int] = (320, 180),
) -> str:
    image_format = _vector_image_format_label(part_name, content_type)
    logger.warning(
        f"Skipping {image_format} image part before Pillow load, "
        f"part_name={part_name}, content_type={content_type}"
    )
    placeholder = create_text_placeholder(
        size,
        [
            f"{image_format} placeholder",
            "Use Windows to parse",
            "the original image",
        ],
    )
    return image_to_b64str(placeholder, image_format="JPEG")


def serialize_office_image(
    image_data: bytes,
    *,
    part_name: object | None = None,
    content_type: str | None = None,
) -> str | None:
    if is_vector_image_part(part_name, content_type):
        return serialize_vector_part_with_placeholder(part_name, content_type)

    try:
        pil_image = Image.open(BytesIO(image_data))
        pil_image.load()
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(
            f"Warning: image cannot be loaded by Pillow: {e}, "
            f"part_name={part_name}, content_type={content_type}"
        )
        return None

    if is_vector_image(pil_image):
        return serialize_vector_image_with_placeholder(pil_image)

    if pil_image.mode == "RGB":
        return image_to_b64str(pil_image, image_format="JPEG")

    if pil_image.mode in {"RGBA", "LA"} or (
        pil_image.mode == "P" and "transparency" in pil_image.info
    ):
        return image_to_b64str(pil_image.convert("RGBA"), image_format="PNG")

    return image_to_b64str(pil_image.convert("RGB"), image_format="JPEG")
