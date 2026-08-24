"""构造携带 MathType MTEF comment 的确定性 WMF/GIF 图片。"""

from __future__ import annotations

import base64
import struct

_TINY_GIF = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)


def _wmf_record(function: int, payload: bytes = b"") -> bytes:
    """构造按 WORD 对齐并声明正确 record size 的 WMF record。"""

    padded = payload + b"\x00" * (len(payload) % 2)
    return struct.pack("<IH", (6 + len(padded)) // 2, function) + padded


def _wmf_comment_record(comment: bytes) -> bytes:
    """把 comment 包装为 META_ESCAPE/MFCOMMENT record。"""

    payload = struct.pack("<HH", 0x000F, len(comment)) + comment
    return _wmf_record(0x0626, payload)


def build_wmf(
    comments: list[bytes],
    *,
    placeable: bool = False,
) -> bytes:
    """构造仅含 comment records 与 EOF 的标准 WMF。"""

    records = [_wmf_comment_record(comment) for comment in comments]
    records.append(_wmf_record(0))
    file_words = (18 + sum(len(record) for record in records)) // 2
    max_record_words = max(len(record) // 2 for record in records)
    header = struct.pack(
        "<HHHIHIH",
        1,
        9,
        0x0300,
        file_words,
        0,
        max_record_words,
        0,
    )
    metafile = header + b"".join(records)
    if not placeable:
        return metafile
    placeable_header = struct.pack(
        "<IHhhhhHIH",
        0x9AC6CDD7,
        0,
        0,
        0,
        100,
        100,
        1440,
        0,
        0,
    )
    return placeable_header + metafile


def pre6_wmf_comment(mtef: bytes) -> bytes:
    """构造 MathType 6.0b 前的单 comment MTEF 头。"""

    if len(mtef) > 0xFFFF:
        raise ValueError("pre-6 WMF fixture MTEF is too large")
    return b"MathType" + struct.pack("<HH", 0x5555, len(mtef)) + mtef


def baseline_wmf_comment(delta: int = 0) -> bytes:
    """构造必须被忽略的 MathType baseline comment。"""

    return b"MathType" + struct.pack("<HH", 0, delta & 0xFFFF)


def apps_mfcc_comment(
    chunk: bytes,
    *,
    total_length: int,
    signature: str = "Design Science, Inc./MTEF",
) -> bytes:
    """构造一个 AppsMFCC v1 chunk。"""

    return (
        b"AppsMFCC"
        + struct.pack("<HII", 1, total_length, len(chunk))
        + signature.encode("ascii")
        + b"\x00"
        + chunk
    )


def apps_mfcc_comments(
    mtef: bytes,
    *,
    chunk_size: int,
    signature: str = "Design Science, Inc./MTEF",
) -> list[bytes]:
    """把 MTEF 切成多个连续 AppsMFCC comments。"""

    if chunk_size <= 0:
        raise ValueError("AppsMFCC fixture chunk_size must be positive")
    return [
        apps_mfcc_comment(
            mtef[start : start + chunk_size],
            total_length=len(mtef),
            signature=signature,
        )
        for start in range(0, len(mtef), chunk_size)
    ]


def _gif_application_extension(
    payload: bytes,
    *,
    authentication: bytes,
    chunk_size: int,
) -> bytes:
    """构造 MathType GIF Application Extension 与 sub-blocks。"""

    if len(authentication) != 3 or not 0 < chunk_size <= 255:
        raise ValueError("GIF application fixture parameters are invalid")
    subblocks = b"".join(
        bytes([len(payload[start : start + chunk_size])])
        + payload[start : start + chunk_size]
        for start in range(0, len(payload), chunk_size)
    )
    return b"\x21\xff\x0bMathType" + authentication + subblocks + b"\x00"


def build_gif_with_extensions(extensions: list[bytes]) -> bytes:
    """把 extensions 插入有效 1×1 GIF 的首个图像块之前。"""

    image_separator = _TINY_GIF.find(b"\x2c")
    if image_separator < 0:
        raise ValueError("tiny GIF fixture has no image descriptor")
    return (
        _TINY_GIF[:image_separator]
        + b"".join(extensions)
        + _TINY_GIF[image_separator:]
    )


def gif_mtef_extension(
    mtef: bytes,
    *,
    chunk_size: int = 255,
) -> bytes:
    """构造可组合到同一 GIF 中的 MathType/001 extension。"""

    return _gif_application_extension(
        mtef,
        authentication=b"001",
        chunk_size=chunk_size,
    )


def build_gif_with_mtef(
    mtef: bytes,
    *,
    chunk_size: int = 255,
    include_baseline: bool = False,
) -> bytes:
    """构造带 MathType/001 MTEF 及可选 002 baseline 的有效 GIF。"""

    extensions = [
        gif_mtef_extension(mtef, chunk_size=chunk_size)
    ]
    if include_baseline:
        extensions.insert(
            0,
            _gif_application_extension(
                b"baseline",
                authentication=b"002",
                chunk_size=255,
            ),
        )
    return build_gif_with_extensions(extensions)


def build_baseline_only_gif() -> bytes:
    """构造只有 MathType/002 baseline、没有公式的有效 GIF。"""

    return build_gif_with_extensions(
        [
            _gif_application_extension(
                b"baseline",
                authentication=b"002",
                chunk_size=255,
            )
        ]
    )
