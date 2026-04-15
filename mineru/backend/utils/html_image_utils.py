# Copyright (c) Opendatalab. All rights reserved.
import base64
import re

from loguru import logger

from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.hash_utils import str_sha256


def _normalize_image_extension(fmt: str) -> str:
    normalized = fmt.lower().split("+", 1)[0]
    return "jpg" if normalized == "jpeg" else normalized


def save_base64_image(b64_data_uri: str, image_writer, page_index: int):
    """Persist a data-URI image via image_writer and return a relative path."""
    if not image_writer:
        return None

    match = re.match(r"data:image/([^;]+);base64,(.+)", b64_data_uri, re.DOTALL)
    if not match:
        logger.warning(f"Unrecognized image_base64 format in page {page_index}, skipping.")
        return None

    fmt = _normalize_image_extension(match.group(1))
    try:
        img_bytes = base64.b64decode(match.group(2))
    except Exception as exc:
        logger.warning(f"Failed to decode image_base64 on page {page_index}: {exc}")
        return None

    img_path = f"{str_sha256(b64_data_uri)}.{fmt}"
    image_writer.write(img_path, img_bytes)
    return img_path


def replace_inline_base64_img_src(markup: str, image_writer, page_index: int) -> str:
    """Replace inline base64 image sources in HTML-like markup with local paths."""
    if not markup or not image_writer or "base64," not in markup:
        return markup

    def _replace_src(match, _writer=image_writer, _idx=page_index):
        img_path = save_base64_image(match.group(1), _writer, _idx)
        if img_path:
            return f'src="{img_path}"'
        return match.group(0)

    return re.sub(
        r'src="(data:image/[^"]+)"',
        _replace_src,
        markup,
    )


def replace_inline_table_images(
    blocks: list[dict],
    image_writer,
    page_index: int,
    table_block_type=BlockType.TABLE,
    table_body_type=BlockType.TABLE_BODY,
    table_span_type=ContentType.TABLE,
) -> None:
    """Persist inline base64 images embedded inside table HTML."""
    if not image_writer:
        return

    for block in blocks:
        if block.get("type") != table_block_type:
            continue

        for sub_block in block.get("blocks", []):
            if sub_block.get("type") != table_body_type:
                continue

            for line in sub_block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") != table_span_type:
                        continue
                    span["html"] = replace_inline_base64_img_src(
                        span.get("html", ""),
                        image_writer,
                        page_index,
                    )


def save_span_image_if_needed(
    span: dict,
    image_writer,
    page_index: int,
    source_key: str = "image_base64",
    target_key: str = "image_path",
) -> None:
    """Persist a span-level base64 image and normalize the target path field."""
    image_base64 = span.get(source_key, "")
    if not image_base64:
        span.setdefault(target_key, "")
        return

    img_path = save_base64_image(image_base64, image_writer, page_index)
    if img_path:
        span[target_key] = img_path
        del span[source_key]
        return

    span.setdefault(target_key, "")
