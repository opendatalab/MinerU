# Copyright (c) Opendatalab. All rights reserved.
"""Preserve PDF link annotations in MinerU's middle JSON representation."""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from typing import Any

from pypdf import PdfReader

from mineru.utils.enum_class import BlockType, ContentType


_PDF_LINK_TYPE = "/Link"
_URI_ACTION = "/URI"
_GOTO_ACTION = "/GoTo"
_PAGE_ANCHOR_PREFIX = "page-"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return value.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _iter_spans(blocks: list[dict] | None) -> Iterator[dict]:
    for block in blocks or []:
        for line in block.get("lines", []):
            line_bbox = line.get("bbox")
            for span in line.get("spans", []):
                if line_bbox and not span.get("bbox"):
                    span["bbox"] = line_bbox
                yield span
        yield from _iter_spans(block.get("blocks"))


def _page_reference_key(reference: Any) -> tuple[int, int] | None:
    reference = getattr(reference, "indirect_reference", reference)
    idnum = getattr(reference, "idnum", None)
    generation = getattr(reference, "generation", 0)
    if idnum is None:
        return None
    return int(idnum), int(generation)


def _build_page_reference_map(reader: PdfReader) -> dict[tuple[int, int], int]:
    result = {}
    for index, page in enumerate(reader.pages):
        key = _page_reference_key(page)
        if key is not None:
            result[key] = index
    return result


def _destination_page_index(
    destination: Any,
    reader: PdfReader,
    page_reference_map: dict[tuple[int, int], int],
) -> int | None:
    if destination is None:
        return None

    try:
        named_destination = reader.named_destinations.get(_as_text(destination))
    except Exception:
        named_destination = None
    if named_destination is not None:
        try:
            return reader.get_destination_page_number(named_destination)
        except Exception:
            pass

    if isinstance(destination, (list, tuple)) and destination:
        page_reference = destination[0]
        if isinstance(page_reference, int):
            return page_reference if 0 <= page_reference < len(reader.pages) else None
        return page_reference_map.get(_page_reference_key(page_reference))

    if isinstance(destination, int):
        return destination if 0 <= destination < len(reader.pages) else None

    return page_reference_map.get(_page_reference_key(destination))


def _resolve_link_target(
    annotation: Any,
    reader: PdfReader,
    page_reference_map: dict[tuple[int, int], int],
) -> tuple[str, int | None]:
    action = annotation.get("/A")
    if action is not None:
        action_type = _as_text(action.get("/S"))
        if action_type == _URI_ACTION:
            return _as_text(action.get("/URI")), None
        if action_type == _GOTO_ACTION:
            destination = action.get("/D")
            target_page = _destination_page_index(
                destination,
                reader,
                page_reference_map,
            )
            return (
                f"#{_PAGE_ANCHOR_PREFIX}{target_page + 1}"
                if target_page is not None
                else _as_text(destination),
                target_page,
            )

    destination = annotation.get("/Dest")
    target_page = _destination_page_index(
        destination,
        reader,
        page_reference_map,
    )
    return (
        f"#{_PAGE_ANCHOR_PREFIX}{target_page + 1}"
        if target_page is not None
        else _as_text(destination),
        target_page,
    )


def _annotation_bbox(
    page: Any,
    rect: Any,
    output_size: list[float] | tuple[float, float] | None,
) -> list[float] | None:
    if not rect or len(rect) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(rect[index]) for index in range(4))
        crop_box = page.cropbox
        crop_left = float(crop_box.left)
        crop_bottom = float(crop_box.bottom)
        crop_width = float(crop_box.width)
        crop_height = float(crop_box.height)
        x0 -= crop_left
        x1 -= crop_left
        y0 -= crop_bottom
        y1 -= crop_bottom
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        top_y0 = crop_height - y1
        top_y1 = crop_height - y0
        rotation = int(page.get("/Rotate", 0) or 0) % 360
        if rotation == 90:
            left, top, right, bottom = (
                crop_height - top_y1,
                x0,
                crop_height - top_y0,
                x1,
            )
            display_width, display_height = crop_height, crop_width
        elif rotation == 180:
            left, top, right, bottom = (
                crop_width - x1,
                crop_height - top_y1,
                crop_width - x0,
                crop_height - top_y0,
            )
            display_width, display_height = crop_width, crop_height
        elif rotation == 270:
            left, top, right, bottom = (
                top_y0,
                crop_width - x1,
                top_y1,
                crop_width - x0,
            )
            display_width, display_height = crop_height, crop_width
        else:
            left, top, right, bottom = x0, top_y0, x1, top_y1
            display_width, display_height = crop_width, crop_height

        if output_size and len(output_size) >= 2:
            scale_x = float(output_size[0]) / max(display_width, 1.0)
            scale_y = float(output_size[1]) / max(display_height, 1.0)
            left, right = left * scale_x, right * scale_x
            top, bottom = top * scale_y, bottom * scale_y
        return [left, top, right, bottom]
    except (TypeError, ValueError, AttributeError):
        return None


def _bbox_intersects(left: list[float], right: list[float]) -> bool:
    if len(left) < 4 or len(right) < 4:
        return False
    intersection_width = min(left[2], right[2]) - max(left[0], right[0])
    intersection_height = min(left[3], right[3]) - max(left[1], right[1])
    if intersection_width <= 0 or intersection_height <= 0:
        return False
    intersection_area = intersection_width * intersection_height
    right_area = max(right[2] - right[0], 1) * max(right[3] - right[1], 1)
    return intersection_area / right_area >= 0.15


def _fallback_link_label(target: str) -> str:
    if target.startswith("#"):
        return target[1:]
    return target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or target


def enrich_pdf_hyperlinks(pdf_info: list[dict], pdf_bytes: bytes) -> None:
    """Attach PDF annotations to overlapping text spans and page anchors."""
    if not pdf_info or not pdf_bytes:
        return

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        return

    page_reference_map = _build_page_reference_map(reader)
    for page_index, page_info in enumerate(pdf_info):
        if page_index >= len(reader.pages):
            break
        annotations = reader.pages[page_index].get("/Annots") or []
        if not annotations:
            continue

        output_size = page_info.get("page_size")
        spans = list(_iter_spans(page_info.get("para_blocks")))
        spans.extend(_iter_spans(page_info.get("discarded_blocks")))
        page_links = []
        for annotation_ref in annotations:
            annotation = annotation_ref.get_object()
            if _as_text(annotation.get("/Subtype")) != _PDF_LINK_TYPE:
                continue
            target, target_page = _resolve_link_target(
                annotation,
                reader,
                page_reference_map,
            )
            if not target or target.lower().startswith(("javascript:", "data:", "vbscript:")):
                continue
            bbox = _annotation_bbox(page=reader.pages[page_index], rect=annotation.get("/Rect"), output_size=output_size)
            if bbox is None:
                continue

            if target_page is not None and target_page < len(pdf_info):
                pdf_info[target_page]["page_anchor"] = f"{_PAGE_ANCHOR_PREFIX}{target_page + 1}"

            matched = False
            for span in spans:
                span_bbox = span.get("bbox")
                if not span_bbox or not _bbox_intersects(bbox, span_bbox):
                    continue
                if span.get("type") == ContentType.HYPERLINK and span.get("url") == target:
                    matched = True
                    continue
                if span.get("type") != ContentType.TEXT or not str(span.get("content", "")).strip():
                    continue
                span["type"] = ContentType.HYPERLINK
                span["url"] = target
                matched = True

            if not matched:
                page_links.append((bbox, target))

        if page_links:
            blocks = page_info.setdefault("para_blocks", [])
            next_index = max((block.get("index", -1) for block in blocks), default=-1) + 1
            for bbox, target in page_links:
                label = _fallback_link_label(target)
                blocks.append(
                    {
                        "type": BlockType.TEXT,
                        "index": next_index,
                        "bbox": bbox,
                        "lines": [
                            {
                                "bbox": bbox,
                                "spans": [
                                    {
                                        "type": ContentType.HYPERLINK,
                                        "content": label,
                                        "url": target,
                                        "bbox": bbox,
                                    }
                                ],
                            }
                        ],
                    }
                )
                next_index += 1
