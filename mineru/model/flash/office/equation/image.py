# Copyright (c) Opendatalab. All rights reserved.

"""从 WMF/GIF 图片 comment 中安全恢复 MathType MTEF 公式。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import PurePosixPath
import struct

from loguru import logger

from ..errors import (
    LegacyOfficeResourceLimitError,
)
from ..limits import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
    MAX_PICTURE_RECORDS,
)
from .mtef import decode_mtef

_PLACEABLE_WMF_MAGIC = b"\xd7\xcd\xc6\x9a"
_META_ESCAPE = 0x0626
_MFCOMMENT = 0x000F
_APPS_MFCC_ID = b"AppsMFCC"
_GIF_HEADERS = {b"GIF87a", b"GIF89a"}
_HISTORICAL_APPS_SIGNATURES = {
    "design science",
    "design science, inc.",
    "wiris",
}


class _ImageEquationError(ValueError):
    """图片 comment 结构损坏、冲突或不受支持。"""


@dataclass(frozen=True, slots=True)
class _AppsChunk:
    """一段已验证边界和 signature 的 AppsMFCC 数据。"""

    key: tuple[str, ...]
    total_length: int
    data: bytes


def _charge_picture_record(counter: list[int]) -> None:
    """累计 WMF/GIF record 数并在超限时抛稳定资源错误。"""

    counter[0] += 1
    if counter[0] > MAX_PICTURE_RECORDS:
        raise LegacyOfficeResourceLimitError(
            "image equation records exceed "
            f"max_picture_records={MAX_PICTURE_RECORDS}"
        )


def _signature_components(value: str) -> tuple[str, ...] | None:
    """按 AppsMFCC 转义规则拆分 slash 分隔的 signature。"""

    components: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "/":
            components.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        return None
    components.append("".join(current))
    if any(not component for component in components):
        return None
    return tuple(components)


def _parse_apps_chunk(comment: bytes) -> _AppsChunk | None:
    """解析一个 AppsMFCC comment，并仅接受规范或历史 MTEF signature。"""

    if not comment.startswith(_APPS_MFCC_ID):
        return None
    if len(comment) > 0x7FFE:
        raise _ImageEquationError("AppsMFCC comment exceeds WMF size limit")
    if len(comment) < 19:
        raise _ImageEquationError("AppsMFCC header is truncated")
    version, total_length, data_length = struct.unpack_from("<HII", comment, 8)
    if version != 1:
        raise _ImageEquationError("AppsMFCC version is unsupported")
    if (
        total_length <= 0
        or data_length <= 0
        or data_length > total_length
        or total_length > MAX_ENTRY_BYTES
    ):
        raise _ImageEquationError("AppsMFCC lengths are invalid")

    signature_start = 18
    signature_limit = min(len(comment), signature_start + 4097)
    signature_end = comment.find(b"\x00", signature_start, signature_limit)
    if signature_end < 0:
        raise _ImageEquationError("AppsMFCC signature is not null-terminated")
    try:
        signature = comment[signature_start:signature_end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise _ImageEquationError("AppsMFCC signature is not ASCII") from exc
    components = _signature_components(signature)
    if components is None:
        raise _ImageEquationError("AppsMFCC signature is malformed")
    normalized = tuple(component.strip().casefold() for component in components)
    is_mtef = (
        len(normalized) >= 2 and normalized[1] == "mtef"
    ) or (
        len(normalized) == 1
        and normalized[0] in _HISTORICAL_APPS_SIGNATURES
    )
    if not is_mtef:
        return None

    data_start = signature_end + 1
    data_end = data_start + int(data_length)
    if data_end < data_start or data_end > len(comment):
        raise _ImageEquationError("AppsMFCC chunk is truncated")
    if any(comment[data_end:]):
        raise _ImageEquationError("AppsMFCC comment has non-padding tail bytes")
    key = (str(version), *normalized[:2])
    return _AppsChunk(
        key=key,
        total_length=int(total_length),
        data=comment[data_start:data_end],
    )


def _pre6_mtef_comment(comment: bytes) -> bytes | None:
    """从 MathType 6.0b 前的单 comment 头提取 MTEF。"""

    if not comment.startswith(b"MathType"):
        return None
    if len(comment) > 0x7FFE:
        raise _ImageEquationError("pre-6 MathType comment exceeds WMF size limit")
    if len(comment) < 12:
        raise _ImageEquationError("pre-6 MathType comment is truncated")
    magic, data_length = struct.unpack_from("<HH", comment, 8)
    if magic != 0x5555:
        # type=0 是 baseline comment，其他类型也不是 MTEF。
        return None
    data_start = 12
    data_end = data_start + int(data_length)
    if data_end < data_start or data_end > len(comment):
        raise _ImageEquationError("pre-6 MathType comment length is invalid")
    if any(comment[data_end:]):
        raise _ImageEquationError("pre-6 MathType comment has non-padding tail bytes")
    return comment[data_start:data_end]


def _wmf_comments(image_data: bytes) -> tuple[list[bytes], bool]:
    """有界遍历 WMF META_ESCAPE/MFCOMMENT 并返回 comment payloads。"""

    cursor = 22 if image_data.startswith(_PLACEABLE_WMF_MAGIC) else 0
    if cursor + 18 > len(image_data):
        return [], False
    metafile_type, header_words, _version, file_words = struct.unpack_from(
        "<HHHI",
        image_data,
        cursor,
    )
    if metafile_type not in {1, 2} or header_words != 9:
        return [], False
    declared_end = cursor + int(file_words) * 2
    if declared_end < cursor + 18 or declared_end > len(image_data):
        raise _ImageEquationError("WMF declared file size is invalid")

    comments: list[bytes] = []
    counter = [0]
    record_cursor = cursor + 18
    saw_eof = False
    while record_cursor < declared_end:
        _charge_picture_record(counter)
        if record_cursor + 6 > declared_end:
            raise _ImageEquationError("WMF record header is truncated")
        record_words, record_function = struct.unpack_from(
            "<IH",
            image_data,
            record_cursor,
        )
        if record_words < 3:
            raise _ImageEquationError("WMF record size is invalid")
        record_end = record_cursor + int(record_words) * 2
        if record_end <= record_cursor or record_end > declared_end:
            raise _ImageEquationError("WMF record exceeds declared file size")
        if record_function == 0:
            saw_eof = True
            record_cursor = record_end
            break
        if record_function == _META_ESCAPE:
            if record_cursor + 10 > record_end:
                raise _ImageEquationError("WMF META_ESCAPE record is truncated")
            escape_function, byte_count = struct.unpack_from(
                "<HH",
                image_data,
                record_cursor + 6,
            )
            data_start = record_cursor + 10
            data_end = data_start + int(byte_count)
            if data_end < data_start or data_end > record_end:
                raise _ImageEquationError("WMF META_ESCAPE byte count is invalid")
            if escape_function == _MFCOMMENT:
                comments.append(image_data[data_start:data_end])
        record_cursor = record_end

    if not saw_eof or record_cursor != declared_end:
        raise _ImageEquationError("WMF EOF record is missing or misplaced")
    if any(image_data[declared_end:]):
        raise _ImageEquationError("WMF contains non-padding bytes after declared end")
    return comments, True


def _wmf_mtef_candidates(image_data: bytes) -> tuple[list[bytes], bool]:
    """从 WMF comments 提取 pre-6 与 AppsMFCC MTEF candidates。"""

    comments, is_wmf = _wmf_comments(image_data)
    if not is_wmf:
        return [], False
    candidates: list[bytes] = []
    recognized = False
    pending: _AppsChunk | None = None
    pending_data = bytearray()

    for comment in comments:
        pre6 = _pre6_mtef_comment(comment)
        if pre6 is not None:
            recognized = True
            candidates.append(pre6)
            continue

        is_apps = comment.startswith(_APPS_MFCC_ID)
        try:
            chunk = _parse_apps_chunk(comment)
        except _ImageEquationError:
            recognized = True
            pending = None
            pending_data.clear()
            continue
        if chunk is None:
            if is_apps and pending is not None:
                recognized = True
                pending = None
                pending_data.clear()
            continue
        recognized = True

        if pending is None:
            if len(chunk.data) == chunk.total_length:
                candidates.append(chunk.data)
            else:
                pending = chunk
                pending_data = bytearray(chunk.data)
            continue

        if (
            chunk.key != pending.key
            or chunk.total_length != pending.total_length
        ):
            pending = None
            pending_data.clear()
            if len(chunk.data) == chunk.total_length:
                candidates.append(chunk.data)
            elif len(chunk.data) < chunk.total_length:
                pending = chunk
                pending_data = bytearray(chunk.data)
            continue
        pending_data.extend(chunk.data)
        if len(pending_data) == pending.total_length:
            candidates.append(bytes(pending_data))
            pending = None
            pending_data.clear()
        elif len(pending_data) > pending.total_length:
            pending = None
            pending_data.clear()

    if pending is not None:
        recognized = True
    return candidates, recognized


def _gif_subblocks(
    image_data: bytes,
    cursor: int,
    counter: list[int],
) -> tuple[bytes, int]:
    """读取以零长度块终止的 GIF sub-block 序列。"""

    chunks: list[bytes] = []
    total = 0
    while True:
        _charge_picture_record(counter)
        if cursor >= len(image_data):
            raise _ImageEquationError("GIF sub-block length is truncated")
        size = image_data[cursor]
        cursor += 1
        if size == 0:
            return b"".join(chunks), cursor
        end = cursor + size
        if end < cursor or end > len(image_data):
            raise _ImageEquationError("GIF sub-block data is truncated")
        total += size
        if total > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"GIF extension exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        chunks.append(image_data[cursor:end])
        cursor = end


def _gif_mtef_candidates(image_data: bytes) -> tuple[list[bytes], bool]:
    """完整遍历 GIF blocks 并提取 MathType/001 Application Extension。"""

    if len(image_data) < 13 or image_data[:6] not in _GIF_HEADERS:
        return [], False
    packed = image_data[10]
    cursor = 13
    if packed & 0x80:
        cursor += 3 * (1 << ((packed & 0x07) + 1))
    if cursor > len(image_data):
        raise _ImageEquationError("GIF global color table is truncated")

    candidates: list[bytes] = []
    recognized = False
    counter = [0]
    saw_trailer = False
    while cursor < len(image_data):
        _charge_picture_record(counter)
        marker = image_data[cursor]
        cursor += 1
        if marker == 0x3B:
            saw_trailer = True
            break
        if marker == 0x2C:
            if cursor + 9 > len(image_data):
                raise _ImageEquationError("GIF image descriptor is truncated")
            image_packed = image_data[cursor + 8]
            cursor += 9
            if image_packed & 0x80:
                cursor += 3 * (1 << ((image_packed & 0x07) + 1))
            if cursor >= len(image_data):
                raise _ImageEquationError("GIF image data is truncated")
            cursor += 1  # LZW minimum code size
            _image_payload, cursor = _gif_subblocks(image_data, cursor, counter)
            continue
        if marker != 0x21:
            raise _ImageEquationError("GIF block marker is invalid")
        if cursor >= len(image_data):
            raise _ImageEquationError("GIF extension label is truncated")
        extension_label = image_data[cursor]
        cursor += 1
        if extension_label != 0xFF:
            _payload, cursor = _gif_subblocks(image_data, cursor, counter)
            continue
        if cursor >= len(image_data):
            raise _ImageEquationError("GIF application block size is truncated")
        application_size = image_data[cursor]
        cursor += 1
        application_end = cursor + application_size
        if application_end > len(image_data):
            raise _ImageEquationError("GIF application identifier is truncated")
        application = image_data[cursor:application_end]
        cursor = application_end
        payload, cursor = _gif_subblocks(image_data, cursor, counter)
        if application_size != 11:
            continue
        application_id = application[:8]
        authentication = application[8:11]
        if application_id == b"MathType" and authentication == b"001":
            recognized = True
            candidates.append(payload)
        # MathType/002 是 baseline，必须明确忽略。

    if not saw_trailer:
        raise _ImageEquationError("GIF trailer is missing")
    if any(image_data[cursor:]):
        raise _ImageEquationError("GIF contains non-padding bytes after trailer")
    return candidates, recognized


def _select_candidate_latex(candidates: list[bytes]) -> str | None:
    """解码全部 candidates；仅返回唯一且完整一致的 LaTeX。"""

    decoded = {
        latex
        for candidate in candidates
        if (latex := decode_mtef(candidate)) is not None
    }
    return next(iter(decoded)) if len(decoded) == 1 else None


def _format_hint(
    image_data: bytes,
    part_name: object | None,
    content_type: str | None,
) -> str | None:
    """结合 magic、扩展名和内容类型确定 WMF/GIF decoder。"""

    if image_data[:6] in _GIF_HEADERS:
        return "gif"
    if image_data.startswith(_PLACEABLE_WMF_MAGIC):
        return "wmf"
    if (
        len(image_data) >= 4
        and image_data[:2] in {b"\x01\x00", b"\x02\x00"}
        and image_data[2:4] == b"\x09\x00"
    ):
        return "wmf"
    suffix = PurePosixPath(str(part_name or "")).suffix.casefold()
    normalized_content_type = (content_type or "").split(";", 1)[0].strip().casefold()
    if suffix == ".gif" or normalized_content_type == "image/gif":
        return "gif"
    if suffix == ".wmf" or normalized_content_type in {
        "image/wmf",
        "image/x-wmf",
        "application/x-msmetafile",
    }:
        return "wmf"
    return None


@dataclass(slots=True)
class OfficeImageEquationDecoder:
    """按共享资源上限缓存并解码 WMF/GIF 图片中的 MTEF。"""

    total_bytes: int = 0
    _cache: dict[tuple[str, bytes], str | None] = field(default_factory=dict)
    _warned: set[tuple[str, bytes]] = field(default_factory=set)

    def decode(
        self,
        image_data: object | None,
        *,
        part_name: object | None = None,
        content_type: str | None = None,
    ) -> str | None:
        """识别图片格式、执行有界 comment 解包并返回完整 LaTeX。"""

        if not isinstance(image_data, bytes):
            return None
        image_format = _format_hint(image_data, part_name, content_type)
        if image_format is None:
            return None
        if len(image_data) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                "image equation payload exceeds "
                f"max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        digest = hashlib.sha256(image_data).digest()
        cache_key = (image_format, digest)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.total_bytes + len(image_data) > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                "image equation payloads exceed "
                f"max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )
        self.total_bytes += len(image_data)

        recognized = False
        try:
            if image_format == "wmf":
                candidates, recognized = _wmf_mtef_candidates(image_data)
            else:
                candidates, recognized = _gif_mtef_candidates(image_data)
            latex = _select_candidate_latex(candidates)
        except LegacyOfficeResourceLimitError:
            raise
        except (_ImageEquationError, ArithmeticError, IndexError, struct.error):
            latex = None
            recognized = (
                b"MathType" in image_data
                or _APPS_MFCC_ID in image_data
            )
        self._cache[cache_key] = latex
        if recognized and latex is None and cache_key not in self._warned:
            self._warned.add(cache_key)
            logger.warning(
                "OFFICE_IMAGE_MTEF_FALLBACK: format={}, payload_sha256={} is malformed, conflicting, or unsupported",
                image_format,
                digest.hex()[:16],
            )
        return latex


def decode_image_embedded_equation(
    image_data: bytes,
    *,
    part_name: object | None = None,
    content_type: str | None = None,
) -> str | None:
    """使用一次性有界 decoder 从单张 WMF/GIF 中恢复 MTEF。"""

    return OfficeImageEquationDecoder().decode(
        image_data,
        part_name=part_name,
        content_type=content_type,
    )
