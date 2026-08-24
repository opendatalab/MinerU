# Copyright (c) Opendatalab. All rights reserved.

"""基于 olefile 的有界 OLE2/CFB 只读包装。"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import olefile  # type: ignore[reportMissingModuleSource]

from .errors import (
    LegacyOfficeMalformedError,
    LegacyOfficeMissingPartError,
    LegacyOfficeResourceLimitError,
)
from .limits import MAX_ENTRY_BYTES, MAX_TOTAL_BYTES


class BoundedOleReader:
    """限制输入、单流及累计读取量，并提供大小写无关的 stream 访问。"""

    def __init__(self, file_bytes: bytes) -> None:
        """校验输入大小并打开内存中的 OLE2 容器。"""

        if not isinstance(file_bytes, bytes):
            raise TypeError("legacy Office input must be bytes")
        if len(file_bytes) > MAX_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"input exceeds max_total_bytes={MAX_TOTAL_BYTES}"
            )
        try:
            self._ole: Any = olefile.OleFileIO(BytesIO(file_bytes), raise_defects=olefile.DEFECT_FATAL)
        except Exception as exc:
            raise LegacyOfficeMalformedError(f"not a readable OLE2 compound file: {exc}") from exc
        self._total_read = 0
        self._stream_names = self._build_stream_name_map()

    def _build_stream_name_map(self) -> dict[str, tuple[str, ...]]:
        """建立大小写无关的完整 stream 名称索引。"""

        names: dict[str, tuple[str, ...]] = {}
        try:
            for parts in self._ole.listdir(streams=True, storages=False):
                normalized = tuple(str(part) for part in parts)
                names["/".join(normalized).casefold()] = normalized
        except Exception as exc:
            self.close()
            raise LegacyOfficeMalformedError(f"cannot enumerate OLE streams: {exc}") from exc
        return names

    def has_stream(self, name: str) -> bool:
        """返回容器是否含有指定 stream。"""

        return name.casefold() in self._stream_names

    def read_stream(self, name: str, *, required: bool = True) -> bytes:
        """有界读取指定 stream；可选 stream 不存在时返回空字节。"""

        parts = self._stream_names.get(name.casefold())
        if parts is None:
            if required:
                raise LegacyOfficeMissingPartError(f"missing required OLE stream: {name}")
            return b""
        try:
            size = int(self._ole.get_size(list(parts)))
        except Exception as exc:
            raise LegacyOfficeMalformedError(f"cannot read OLE stream size: {name}: {exc}") from exc
        if size > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"stream {name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        if self._total_read + size > MAX_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"OLE streams exceed max_total_bytes={MAX_TOTAL_BYTES}"
            )
        try:
            with self._ole.openstream(list(parts)) as stream:
                payload = stream.read(MAX_ENTRY_BYTES + 1)
        except Exception as exc:
            raise LegacyOfficeMalformedError(f"cannot read OLE stream {name!r}: {exc}") from exc
        if len(payload) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"stream {name!r} exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        self._total_read += len(payload)
        return payload

    def metadata(self) -> Any | None:
        """尽力读取 SummaryInformation，失败时不影响正文解析。"""

        try:
            return self._ole.get_metadata()
        except Exception:
            return None

    def close(self) -> None:
        """关闭底层 olefile 句柄。"""

        ole = getattr(self, "_ole", None)
        if ole is not None:
            ole.close()
            self._ole = None

    def __enter__(self) -> BoundedOleReader:
        """返回当前有界读取器。"""

        return self

    def __exit__(self, *_args: object) -> None:
        """离开上下文时关闭底层句柄。"""

        self.close()
