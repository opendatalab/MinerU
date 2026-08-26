# Copyright (c) Opendatalab. All rights reserved.

"""现代 Office OOXML 包中的 MathType/Equation OLE 公式解码适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from ..errors import LegacyOfficeResourceLimitError
from ..limits import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
)
from .mtef import decode_equation_object

CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
EQUATION_PROG_ID = "equation"
EQUATION_PROG_ID_PREFIX = "equation."


def is_mathtype_equation_prog_id(prog_id: object | None) -> bool:
    """判断 OLE ProgID 是否为 Equation 或带非空版本后缀的 Equation.*。"""

    if not isinstance(prog_id, str):
        return False
    normalized = prog_id.strip().casefold()
    return normalized == EQUATION_PROG_ID or (
        normalized.startswith(EQUATION_PROG_ID_PREFIX)
        and len(normalized) > len(EQUATION_PROG_ID_PREFIX)
    )


@dataclass(slots=True)
class OoxmlEquationDecoder:
    """按共享资源上限缓存并解码 OOXML 中的公式 OLE 对象。"""

    total_bytes: int = 0
    _cache: dict[bytes, str | None] = field(default_factory=dict)

    def decode(
        self,
        blob: bytes | None,
        *,
        prog_id: object | None,
        show_as_icon: bool = False,
    ) -> str | None:
        """校验公式 ProgID、图标模式、CFB 头和资源预算后返回 LaTeX。"""

        if show_as_icon or not is_mathtype_equation_prog_id(prog_id) or blob is None:
            return None
        if not isinstance(blob, bytes):
            return None
        if len(blob) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"OOXML equation object exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        if not blob.startswith(CFB_MAGIC):
            return None

        digest = hashlib.sha256(blob).digest()
        if digest in self._cache:
            return self._cache[digest]
        if self.total_bytes + len(blob) > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                "OOXML equation objects exceed "
                f"max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )

        self.total_bytes += len(blob)
        latex = decode_equation_object(blob)
        self._cache[digest] = latex
        return latex
