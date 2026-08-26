# Copyright (c) Opendatalab. All rights reserved.

"""从 DOC Data/PICF 与 Word OfficeArt drawing 中恢复图片。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from loguru import logger

from ..legacy import LegacyOfficeResourceLimitError
from ..legacy.binary import bounded_slice, get_u16, get_u32
from ..legacy.limits import MAX_ASSET_TOTAL_BYTES
from ..legacy.officeart import (
    OfficeImagePayload,
    decode_bstore,
    extract_word_shapes,
    first_blip,
    record_at,
)
from ..image_equation import (
    OfficeImageEquationDecoder,
)

from .models import DocImage, DocImagePayload
from .records import DocBudget, parse_plc

_PLACEABLE_WMF_MAGIC = b"\xd7\xcd\xc6\x9a"


@dataclass(slots=True)
class ImageStore:
    """按内容去重并限制 DOC 图片累计字节数。"""

    total: int = 0
    cache: dict[bytes, DocImagePayload] = field(default_factory=dict)
    equation_decoder: OfficeImageEquationDecoder = field(
        default_factory=OfficeImageEquationDecoder
    )

    def add(self, payload: OfficeImagePayload) -> DocImagePayload:
        """计入一张唯一图片并返回内部载荷。"""

        digest = hashlib.sha256(payload.data).digest()
        cached = self.cache.get(digest)
        if cached is not None:
            return cached
        if self.total + len(payload.data) > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"embedded assets exceed max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )
        converted = DocImagePayload(
            payload.data,
            payload.extension,
            payload.content_type,
            self.equation_decoder.decode(
                payload.data,
                part_name=f"image.{payload.extension}",
                content_type=payload.content_type,
            ),
        )
        self.total += len(payload.data)
        self.cache[digest] = converted
        return converted


def inline_picture(
    data_stream: bytes,
    *,
    offset: int,
    store: ImageStore,
    budget: DocBudget,
) -> DocImagePayload | None:
    """解析 sprmCPicLocation 指向的 PICFAndOfficeArtData。"""

    total_length = get_u32(data_stream, offset)
    header_length = get_u16(data_stream, offset + 4)
    if total_length is None or header_length is None or total_length < header_length:
        return None
    picf = bounded_slice(data_stream, offset, total_length)
    if picf is None:
        logger.warning(f"DOC picture at Data offset {offset} is out of bounds")
        return None
    art = picf[min(header_length, len(picf)) :]
    decoded = first_blip(art, charge=budget.charge)
    if decoded is None:
        # 少量旧文件把原始位图直接放在 PICF 尾部，按 magic 尽力保留。
        signatures = (
            (_PLACEABLE_WMF_MAGIC, "wmf", "image/wmf"),
            (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
            (b"\xff\xd8\xff", "jpg", "image/jpeg"),
            (b"GIF8", "gif", "image/gif"),
            (b"BM", "bmp", "image/bmp"),
            (b"II*\x00", "tiff", "image/tiff"),
            (b"MM\x00*", "tiff", "image/tiff"),
        )
        for signature, extension, content_type in signatures:
            position = art.find(signature)
            if position >= 0:
                decoded = OfficeImagePayload(art[position:], extension, content_type)
                break
    return store.add(decoded) if decoded is not None else None


def floating_pictures(
    table_stream: bytes,
    *,
    word_document: bytes,
    shape_plc_offset: int,
    shape_plc_size: int,
    drawing_offset: int,
    drawing_size: int,
    store: ImageStore,
    budget: DocBudget,
) -> list[DocImage]:
    """按 PlcfSpaMom anchor CP 将 floating shape 的 BStore 图片绑定到正文。"""

    plc_payload = bounded_slice(table_stream, shape_plc_offset, shape_plc_size)
    drawing = bounded_slice(table_stream, drawing_offset, drawing_size)
    if plc_payload is None or drawing is None:
        return []
    cps, items = parse_plc(plc_payload, item_size=26, budget=budget)
    if not items:
        return []
    assets = decode_bstore(
        drawing,
        charge=budget.charge,
        delay_stream=word_document,
    )
    shapes = {}
    first = record_at(drawing, 0, charge=budget.charge)
    cursor = (8 + len(first.payload)) if first is not None else len(drawing)
    while cursor < len(drawing):
        # OfficeArtWordDrawing 在 DgContainer 前有一个 main/header 标签字节。
        cursor += 1
        container = record_at(drawing, cursor, charge=budget.charge)
        if container is None:
            break
        for shape in extract_word_shapes(container.payload, charge=budget.charge):
            if shape.shape_id is not None:
                shapes.setdefault(shape.shape_id, shape)
        cursor += 8 + len(container.payload)
    anchors = [
        (shape_id, cps[index])
        for index, item in enumerate(items)
        if index < len(cps) and (shape_id := get_u32(item, 0)) is not None
    ]
    images: list[DocImage] = []
    for shape_id, shape in sorted(shapes.items()):
        if shape.hidden or shape.pib is None:
            continue
        payload = assets.get(shape.pib)
        if payload is None:
            continue
        anchor_cp = next((cp for anchor_id, cp in anchors if anchor_id == shape_id), None)
        if anchor_cp is None:
            candidates = [(anchor_id, cp) for anchor_id, cp in anchors if anchor_id <= shape_id]
            if candidates:
                anchor_cp = max(candidates)[1]
        if anchor_cp is None:
            continue
        images.append(DocImage(cp=anchor_cp, payload=store.add(payload)))
    return images
