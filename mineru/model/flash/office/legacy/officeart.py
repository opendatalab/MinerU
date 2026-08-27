# Copyright (c) Opendatalab. All rights reserved.

"""旧版 Office 二进制格式共享的 OfficeArt 记录与图片解码。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import struct
import zlib

from loguru import logger

from ..image import ensure_bmp_header
from ..errors import LegacyOfficeResourceLimitError
from ..limits import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
    MAX_PICTURE_RECORDS,
    MAX_RECORD_DEPTH,
)

OFFICEART_CONTAINER_VERSION = 0xF
OFFICEART_DGG_CONTAINER = 0xF000
OFFICEART_BSTORE_CONTAINER = 0xF001
OFFICEART_SP_CONTAINER = 0xF004
OFFICEART_BSE = 0xF007
OFFICEART_FSP = 0xF00A
OFFICEART_FOPT = 0xF00B
OFFICEART_CLIENT_ANCHOR = 0xF010
OFFICEART_TERTIARY_FOPT = 0xF122

FOPT_PIB = 0x0104
FOPT_GROUP_SHAPE = 0x03BF
F_HIDDEN = 0x0000_0002
F_USE_HIDDEN = 0x0002_0000


@dataclass(frozen=True, slots=True)
class OfficeArtRecord:
    """一条已经通过长度边界校验的 OfficeArt 记录。"""

    offset: int
    version: int
    instance: int
    record_type: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class OfficeImagePayload:
    """从 BLIP 中恢复出的原始图片及其媒体类型。"""

    data: bytes
    extension: str
    content_type: str


@dataclass(frozen=True, slots=True)
class OfficeArtShape:
    """Excel drawing 中可绑定到 OBJ 的形状属性。"""

    shape_id: int | None
    anchor: tuple[int, int, int, int] | None
    pib: int | None
    hidden: bool


def record_at(
    data: bytes,
    offset: int,
    *,
    end: int | None = None,
    charge: Callable[[], None] | None = None,
) -> OfficeArtRecord | None:
    """从指定偏移读取一条 OfficeArt 记录，坏边界返回空值。"""

    limit = len(data) if end is None else min(end, len(data))
    if offset < 0 or offset + 8 > limit:
        return None
    version_instance, record_type, length = struct.unpack_from("<HHI", data, offset)
    payload_start = offset + 8
    payload_end = payload_start + int(length)
    if payload_end < payload_start or payload_end > limit:
        return None
    if charge is not None:
        charge()
    return OfficeArtRecord(
        offset=offset,
        version=version_instance & 0xF,
        instance=version_instance >> 4,
        record_type=record_type,
        payload=data[payload_start:payload_end],
    )


def iter_records(
    data: bytes,
    *,
    start: int = 0,
    end: int | None = None,
    charge: Callable[[], None] | None = None,
) -> Iterator[OfficeArtRecord]:
    """顺序遍历同一 OfficeArt 容器内的直接子记录。"""

    limit = len(data) if end is None else min(end, len(data))
    cursor = start
    while cursor < limit:
        record = record_at(data, cursor, end=limit, charge=charge)
        if record is None:
            return
        yield record
        cursor += 8 + len(record.payload)


def iter_descendants(
    data: bytes,
    *,
    charge: Callable[[], None] | None = None,
) -> Iterator[OfficeArtRecord]:
    """以显式栈深度优先遍历 OfficeArt 记录树并限制嵌套深度。"""

    stack: list[Iterator[OfficeArtRecord]] = [iter_records(data, charge=charge)]
    while stack:
        if len(stack) > MAX_RECORD_DEPTH:
            raise LegacyOfficeResourceLimitError(
                f"record nesting exceeds max_record_depth={MAX_RECORD_DEPTH}"
            )
        try:
            record = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        yield record
        if record.version == OFFICEART_CONTAINER_VERSION:
            stack.append(iter_records(record.payload, charge=charge))


def _simple_properties(record: OfficeArtRecord) -> dict[int, int]:
    """读取 FOPT 简单属性并让后出现的同名属性覆盖前值。"""

    properties: dict[int, int] = {}
    for index in range(record.instance):
        offset = index * 6
        if offset + 6 > len(record.payload):
            break
        opid, value = struct.unpack_from("<HI", record.payload, offset)
        properties[opid & 0x3FFF] = int(value)
    return properties


def _excel_client_anchor(payload: bytes) -> tuple[int, int, int, int] | None:
    """把 OfficeArtClientAnchorChart 转成起止行列坐标。"""

    if len(payload) < 18:
        return None
    start_col = struct.unpack_from("<H", payload, 2)[0]
    start_row = struct.unpack_from("<H", payload, 6)[0]
    end_col = struct.unpack_from("<H", payload, 10)[0]
    end_row = struct.unpack_from("<H", payload, 14)[0]
    return int(start_row), int(start_col), int(end_row), int(end_col)


def _shape_from_container(
    record: OfficeArtRecord,
    *,
    charge: Callable[[], None] | None = None,
) -> OfficeArtShape | None:
    """从单个 SpContainer 提取 shape id、anchor、pib 与隐藏状态。"""

    shape_id: int | None = None
    anchor: tuple[int, int, int, int] | None = None
    properties: dict[int, int] = {}
    for child in iter_records(record.payload, charge=charge):
        if child.record_type == OFFICEART_FSP and len(child.payload) >= 4:
            shape_id = int(struct.unpack_from("<I", child.payload, 0)[0])
        elif child.record_type in {OFFICEART_FOPT, OFFICEART_TERTIARY_FOPT}:
            properties.update(_simple_properties(child))
        elif child.record_type == OFFICEART_CLIENT_ANCHOR:
            anchor = _excel_client_anchor(child.payload)
    hidden_flags = properties.get(FOPT_GROUP_SHAPE, 0)
    hidden = bool(hidden_flags & F_USE_HIDDEN and hidden_flags & F_HIDDEN)
    pib = properties.get(FOPT_PIB)
    if anchor is None and pib is None:
        return None
    return OfficeArtShape(shape_id=shape_id, anchor=anchor, pib=pib, hidden=hidden)


def extract_excel_shapes(
    data: bytes,
    *,
    charge: Callable[[], None] | None = None,
) -> list[OfficeArtShape]:
    """按 drawing 顺序提取可与 Excel OBJ 一一绑定的形状。"""

    shapes: list[OfficeArtShape] = []
    for record in iter_descendants(data, charge=charge):
        if record.record_type != OFFICEART_SP_CONTAINER:
            continue
        shape = _shape_from_container(record, charge=charge)
        if shape is not None:
            shapes.append(shape)
    return shapes


def _bitmap_payload(body: bytes, instance: int) -> bytes | None:
    """跳过 BLIP UID 和 tag，返回位图原始载荷。"""

    doubled = instance in {0x46B, 0x6E3, 0x6E1, 0x7A9}
    start = (32 if doubled else 16) + 1
    return body[start:] if start < len(body) else None


def decode_blip(record: OfficeArtRecord) -> OfficeImagePayload | None:
    """解码常见位图和 EMF/WMF BLIP，并限制矢量解压输出。"""

    instance = record.instance
    body = record.payload
    if record.record_type in {0xF01D, 0xF01E, 0xF01F, 0xF029}:
        data = _bitmap_payload(body, instance)
        if data is None:
            return None
        if len(data) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError("bitmap BLIP exceeds max_entry_bytes")
        if record.record_type == 0xF01D:
            return OfficeImagePayload(data=data, extension="jpg", content_type="image/jpeg")
        if record.record_type == 0xF01E:
            return OfficeImagePayload(data=data, extension="png", content_type="image/png")
        if record.record_type == 0xF029:
            return OfficeImagePayload(data=data, extension="tiff", content_type="image/tiff")
        return OfficeImagePayload(data=ensure_bmp_header(data), extension="bmp", content_type="image/bmp")

    if record.record_type not in {0xF01A, 0xF01B}:
        return None
    doubled = instance in ({0x3D5} if record.record_type == 0xF01A else {0x217})
    header_offset = 32 if doubled else 16
    if header_offset + 34 > len(body):
        return None
    declared_size = int(struct.unpack_from("<I", body, header_offset)[0])
    compressed_size = int(struct.unpack_from("<I", body, header_offset + 28)[0])
    compression = body[header_offset + 32]
    payload_start = header_offset + 34
    payload = body[payload_start : payload_start + compressed_size]
    if declared_size > MAX_ENTRY_BYTES:
        raise LegacyOfficeResourceLimitError("metafile BLIP exceeds max_entry_bytes")
    if compression == 0:
        data = b""
        reached_eof = False
        for window_bits in (-zlib.MAX_WBITS, zlib.MAX_WBITS):
            try:
                inflater = zlib.decompressobj(window_bits)
                candidate = inflater.decompress(payload, MAX_ENTRY_BYTES + 1)
                candidate += inflater.flush(MAX_ENTRY_BYTES + 1 - len(candidate))
            except zlib.error:
                continue
            if inflater.eof:
                data = candidate
                reached_eof = True
                break
        if not reached_eof:
            return None
        if len(data) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                "metafile BLIP decompression exceeded its limit"
            )
    else:
        data = payload
    if record.record_type == 0xF01A:
        return OfficeImagePayload(data=data, extension="emf", content_type="image/emf")
    if data.startswith(b"\xd7\xcd\xc6\x9a") and len(data) >= 22:
        data = data[22:]
    return OfficeImagePayload(data=data, extension="wmf", content_type="image/wmf")


def first_blip(
    data: bytes,
    *,
    charge: Callable[[], None] | None = None,
) -> OfficeImagePayload | None:
    """深度优先返回一段 OfficeArt 数据中的首个可支持 BLIP。"""

    for record in iter_descendants(data, charge=charge):
        if record.record_type == OFFICEART_BSE:
            decoded = _decode_bse_body(record.payload, charge=charge)
            if decoded is not None:
                return decoded
        decoded = decode_blip(record)
        if decoded is not None:
            return decoded
    return None


def extract_word_shapes(
    data: bytes,
    *,
    charge: Callable[[], None] | None = None,
) -> list[OfficeArtShape]:
    """按 Word drawing 顺序提取 shape id、pib 和隐藏状态。"""

    shapes: list[OfficeArtShape] = []
    for record in iter_descendants(data, charge=charge):
        if record.record_type != OFFICEART_SP_CONTAINER:
            continue
        shape = _shape_from_container(record, charge=charge)
        if shape is not None:
            shapes.append(shape)
    return shapes


def _decode_bse_body(
    body: bytes,
    *,
    charge: Callable[[], None] | None = None,
    delay_stream: bytes | None = None,
) -> OfficeImagePayload | None:
    """从 FBSE body 的内嵌或延迟 BLIP 中恢复图片。"""

    if len(body) < 36:
        return None
    inner_offset = 36 + int(body[33])
    inner = record_at(body, inner_offset, charge=charge) if inner_offset < len(body) else None
    if inner is None and delay_stream is not None:
        delayed_size = int(struct.unpack_from("<I", body, 20)[0])
        reference_count = int(struct.unpack_from("<I", body, 24)[0])
        delayed_offset = int(struct.unpack_from("<I", body, 28)[0])
        if reference_count and delayed_offset != 0xFFFF_FFFF:
            delayed_end = delayed_offset + delayed_size
            if delayed_end >= delayed_offset and delayed_end <= len(delay_stream):
                inner = record_at(delay_stream, delayed_offset, end=delayed_end, charge=charge)
    return decode_blip(inner) if inner is not None else None


def decode_bstore(
    data: bytes,
    *,
    charge: Callable[[], None] | None = None,
    delay_stream: bytes | None = None,
) -> dict[int, OfficeImagePayload]:
    """按一基 BSE 序号解码 drawing group 中的图片资源。"""

    bse_records = [
        record
        for record in iter_descendants(data, charge=charge)
        if record.record_type == OFFICEART_BSE
    ]
    result: dict[int, OfficeImagePayload] = {}
    asset_total = 0
    for index, bse in enumerate(bse_records[:MAX_PICTURE_RECORDS], start=1):
        decoded = _decode_bse_body(
            bse.payload,
            charge=charge,
            delay_stream=delay_stream,
        )
        if decoded is None:
            continue
        asset_total += len(decoded.data)
        if asset_total > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"embedded assets exceed max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )
        result[index] = decoded
    if len(bse_records) > MAX_PICTURE_RECORDS:
        logger.warning(
            "LEGACY_OFFICE_PICTURE_LIMIT: ignored BSE records after {}",
            MAX_PICTURE_RECORDS,
        )
    return result
