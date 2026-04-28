# Copyright (c) Opendatalab. All rights reserved.
from typing import BinaryIO


def rewind_stream(file_stream: BinaryIO) -> bool:
    """将可复位的二进制流移动到起点；不可复位时返回 False。"""
    try:
        file_stream.seek(0)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def read_stream_bytes_from_start(file_stream: BinaryIO) -> bytes:
    """从流起点读取完整字节；不可复位的流则从当前位置读取剩余字节。"""
    rewind_stream(file_stream)
    return file_stream.read()
