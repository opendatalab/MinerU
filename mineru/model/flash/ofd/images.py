# Copyright (c) Opendatalab. All rights reserved.
"""OFD ImageObject 资源解码与安全图片投影。"""

from __future__ import annotations

import math
from io import BytesIO

from loguru import logger
from PIL import Image, UnidentifiedImageError
from lxml import etree  # type: ignore[reportMissingImports]

from .._shared.image import image_to_b64str
from ....types import BBox
from .geometry import Affine, bbox_intersection, canonical_angle, parse_affine, parse_st_box, transform_angle, transform_bbox
from .models import ImageItem, ResourceRegistry
from .package import OfdPackage, parse_int

_JBIG2_SIGNATURE = b"\x97JB2\r\n\x1a\n"


def _serialize_raster(data: bytes, *, alpha: int, angle: int) -> str | None:
    """使用 Pillow 校验、旋转并编码常见位图。"""
    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            image = source.copy()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None
    if angle == 90:
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif angle == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif angle == 270:
        image = image.transpose(Image.Transpose.ROTATE_90)
    if alpha < 255:
        rgba = image.convert("RGBA")
        current_alpha = rgba.getchannel("A")
        current_alpha = current_alpha.point(lambda value: round(value * alpha / 255))
        rgba.putalpha(current_alpha)
        image = rgba
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        return image_to_b64str(image.convert("RGBA"), image_format="PNG")
    return image_to_b64str(image.convert("RGB"), image_format="JPEG")


def build_image_item(
    image_object: etree._Element,
    *,
    parent_transform: Affine,
    parent_clip: BBox,
    resources: ResourceRegistry,
    package: OfdPackage,
    paint_order: int,
    layer_type: str,
    template_id: int | None,
    resolved_style: dict[str, str] | None = None,
) -> ImageItem | None:
    """解析一个 ImageObject，并保留不支持图片的占位几何。"""
    style = resolved_style or {}
    if (style.get("Visible") or image_object.get("Visible") or "true").casefold() in {"false", "0"}:
        return None
    boundary = parse_st_box(image_object.get("Boundary"))
    if boundary is None:
        return None
    page_bbox = transform_bbox(boundary, parent_transform)
    if page_bbox is None:
        return None
    page_bbox = bbox_intersection(page_bbox, parent_clip)
    if page_bbox is None:
        return None
    resource_id = parse_int(image_object.get("ResourceID") or image_object.get("ResouceID"))
    media = resources.media.get(resource_id) if resource_id is not None else None
    object_id = parse_int(image_object.get("ID"))
    if media is None:
        return ImageItem(
            bbox=page_bbox,
            image_base64=None,
            paint_order=paint_order,
            object_id=object_id,
            layer_type=layer_type,
            template_id=template_id,
            diagnostic="missing_media_resource",
        )
    data = package.read_part(media.media_part, asset=True)
    if data is None:
        diagnostic = "missing_media_part"
        payload = None
    elif data.startswith(_JBIG2_SIGNATURE):
        diagnostic = "unsupported_jbig2"
        payload = None
    else:
        try:
            alpha = min(255, max(0, int(float(style.get("Alpha") or image_object.get("Alpha") or 255))))
        except ValueError:
            alpha = 255
        ctm = parse_affine(image_object.get("CTM"))
        raw_angle = transform_angle(parent_transform.compose(ctm))
        angle = canonical_angle(raw_angle)
        if angle not in {0, 90, 180, 270} or not math.isclose(math.remainder(raw_angle, 90.0), 0.0, abs_tol=5.0):
            diagnostic = "unsupported_image_affine"
            payload = None
        else:
            payload = _serialize_raster(data, alpha=alpha, angle=angle)
            diagnostic = None if payload is not None else "unsupported_image_payload"
    if diagnostic:
        logger.warning(f"OFD_IMAGE_SKIPPED: code={diagnostic}, object_id={object_id}, part={media.media_part!r}")
    return ImageItem(
        bbox=page_bbox,
        image_base64=payload,
        paint_order=paint_order,
        object_id=object_id,
        layer_type=layer_type,
        template_id=template_id,
        diagnostic=diagnostic,
    )


__all__ = ["build_image_item"]
