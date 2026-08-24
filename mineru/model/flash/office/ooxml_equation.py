# Copyright (c) Opendatalab. All rights reserved.

"""现代 Office OOXML 包中的 Equation.3 OLE 公式解码适配器。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from mineru.model.flash.legacy_office.errors import LegacyOfficeResourceLimitError
from mineru.model.flash.legacy_office.limits import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
)
from mineru.model.flash.legacy_office.mtef import decode_equation_object

CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
EQUATION_3_PROG_ID = "equation.3"


def is_equation_3_prog_id(prog_id: object | None) -> bool:
    """判断 OOXML OLE ProgID 是否精确指向 Equation.3。"""

    return isinstance(prog_id, str) and prog_id.strip().casefold() == EQUATION_3_PROG_ID


@dataclass(slots=True)
class OoxmlEquationDecoder:
    """按共享资源上限缓存并解码 OOXML 中的 Equation.3 对象。"""

    total_bytes: int = 0
    _cache: dict[bytes, str | None] = field(default_factory=dict)

    def decode(
        self,
        blob: bytes | None,
        *,
        prog_id: object | None,
        show_as_icon: bool = False,
    ) -> str | None:
        """校验 ProgID、图标模式、CFB 头和资源预算后返回 LaTeX。"""

        if show_as_icon or not is_equation_3_prog_id(prog_id) or blob is None:
            return None
        if not isinstance(blob, bytes):
            return None
        if len(blob) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"OOXML Equation.3 object exceeds max_entry_bytes={MAX_ENTRY_BYTES}"
            )
        if not blob.startswith(CFB_MAGIC):
            return None

        digest = hashlib.sha256(blob).digest()
        if digest in self._cache:
            return self._cache[digest]
        if self.total_bytes + len(blob) > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                "OOXML Equation.3 objects exceed "
                f"max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )

        self.total_bytes += len(blob)
        latex = decode_equation_object(blob)
        self._cache[digest] = latex
        return latex
