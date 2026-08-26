# Copyright (c) Opendatalab. All rights reserved.

"""读取 Word 97–2003 WordDocument stream 中的变长 FIB。"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ..errors import LegacyOfficeMalformedError

FIB_IDENT = 0xA5EC
MIN_WORD97_NFIB = 0x00C1

FCLCB_STSHF = 1
FCLCB_FOOTNOTE_REF = 2
FCLCB_FOOTNOTE_TEXT = 3
FCLCB_SECTION = 6
FCLCB_HEADER = 11
FCLCB_BTE_CHPX = 12
FCLCB_BTE_PAPX = 13
FCLCB_FIELD_MAIN = 16
FCLCB_FIELD_HEADER = 17
FCLCB_FIELD_FOOTNOTE = 18
FCLCB_BOOKMARK_NAMES = 21
FCLCB_BOOKMARK_START = 22
FCLCB_BOOKMARK_END = 23
FCLCB_DOP = 31
FCLCB_CLX = 33
FCLCB_SHAPE_MAIN = 40
FCLCB_SHAPE_HEADER = 41
FCLCB_ENDNOTE_REF = 46
FCLCB_ENDNOTE_TEXT = 47
FCLCB_FIELD_ENDNOTE = 48
FCLCB_DGG_INFO = 50
FCLCB_TEXTBOX_TEXT = 56
FCLCB_FIELD_TEXTBOX = 57
FCLCB_HEADER_TEXTBOX_TEXT = 58
FCLCB_FIELD_HEADER_TEXTBOX = 59
FCLCB_LISTS = 73
FCLCB_LIST_OVERRIDES = 74
FCLCB_TEXTBOX_BREAK = 75
FCLCB_HEADER_TEXTBOX_BREAK = 76


@dataclass(frozen=True, slots=True)
class FcLcb:
    """FIB 中一对 stream 偏移和字节长度。"""

    fc: int = 0
    lcb: int = 0


@dataclass(frozen=True, slots=True)
class FibBase:
    """FIB 固定头中与解析相关的字段。"""

    n_fib: int
    lid: int
    flags: int
    fc_min: int
    fc_mac: int

    @property
    def complex(self) -> bool:
        """返回文档是否使用 complex/fast-save piece table。"""

        return bool(self.flags & 0x0004)

    @property
    def encrypted(self) -> bool:
        """返回文档是否设置加密标志。"""

        return bool(self.flags & 0x0100)

    @property
    def uses_1table(self) -> bool:
        """返回 FIB 指定的首选 Table stream。"""

        return bool(self.flags & 0x0200)

    @property
    def far_east(self) -> bool:
        """返回文档是否优先使用远东语言标识。"""

        return bool(self.flags & 0x4000)

    @property
    def obfuscated(self) -> bool:
        """返回文档是否设置 XOR 混淆标志。"""

        return bool(self.flags & 0x8000)


@dataclass(frozen=True, slots=True)
class FileInformationBlock:
    """完成边界校验的 Word 97+ FIB。"""

    base: FibBase
    rgw: tuple[int, ...]
    rglw: tuple[int, ...]
    pairs: tuple[FcLcb, ...]
    csw_new: tuple[int, ...]
    size: int

    @property
    def n_fib(self) -> int:
        """返回版本扩展中的有效 nFib。"""

        return self.csw_new[0] if self.csw_new else self.base.n_fib

    def pair(self, index: int) -> FcLcb:
        """读取可选 fc/lcb 对，不存在时返回零值。"""

        return self.pairs[index] if 0 <= index < len(self.pairs) else FcLcb()

    def story_count(self, index: int) -> int:
        """读取 FibRgLw97 中一个 story 的 UTF-16 CP 数。"""

        return int(self.rglw[index]) if 0 <= index < len(self.rglw) else 0

    @property
    def ccp_text(self) -> int:
        """返回主文档 story 的 CP 数。"""

        return self.story_count(3)

    @property
    def ccp_footnote(self) -> int:
        """返回脚注 story 的 CP 数。"""

        return self.story_count(4)

    @property
    def ccp_header(self) -> int:
        """返回页眉页脚 story 的 CP 数。"""

        return self.story_count(5)

    @property
    def ccp_macro(self) -> int:
        """返回宏 story 的 CP 数。"""

        return self.story_count(6)

    @property
    def ccp_annotation(self) -> int:
        """返回批注 story 的 CP 数。"""

        return self.story_count(7)

    @property
    def ccp_endnote(self) -> int:
        """返回尾注 story 的 CP 数。"""

        return self.story_count(8)

    @property
    def ccp_textbox(self) -> int:
        """返回正文文本框 story 的 CP 数。"""

        return self.story_count(9)

    @property
    def ccp_header_textbox(self) -> int:
        """返回页眉文本框 story 的 CP 数。"""

        return self.story_count(10)

    @property
    def total_story_cp(self) -> int:
        """返回全部已知 story 的累计 CP 数。"""

        return sum(self.story_count(index) for index in range(3, 11))

    @property
    def story_bases(self) -> dict[str, int]:
        """返回各 story 在全局 CP 空间中的起点。"""

        counts = [self.story_count(index) for index in range(3, 11)]
        names = [
            "main",
            "footnote",
            "header",
            "macro",
            "annotation",
            "endnote",
            "textbox",
            "header_textbox",
        ]
        bases: dict[str, int] = {}
        cursor = 0
        for name, count in zip(names, counts, strict=True):
            bases[name] = cursor
            cursor += count
        return bases


def _read_values(data: bytes, offset: int, count: int, width: int, label: str) -> tuple[tuple[int, ...], int]:
    """按指定宽度读取一组无符号小端整数。"""

    if count < 0 or width not in {2, 4}:
        raise LegacyOfficeMalformedError(f"invalid {label} count")
    size = count * width
    end = offset + size
    if offset < 0 or end < offset or end > len(data):
        raise LegacyOfficeMalformedError(f"truncated FIB {label}")
    if count == 0:
        return (), end
    code = "H" if width == 2 else "I"
    return tuple(int(value) for value in struct.unpack_from(f"<{count}{code}", data, offset)), end


def parse_fib(word_document: bytes) -> FileInformationBlock:
    """按 MS-DOC 变长布局解析 FIB，并拒绝 Word 95 及更早版本。"""

    if len(word_document) < 34:
        raise LegacyOfficeMalformedError("WordDocument FIB is truncated")
    ident, n_fib = struct.unpack_from("<HH", word_document, 0)
    if ident != FIB_IDENT:
        raise LegacyOfficeMalformedError("WordDocument FIB magic is invalid")
    if n_fib < MIN_WORD97_NFIB:
        raise LegacyOfficeMalformedError(f"Word 95 or earlier nFib is unsupported: 0x{n_fib:04X}")
    lid = int(struct.unpack_from("<H", word_document, 6)[0])
    flags = int(struct.unpack_from("<H", word_document, 10)[0])
    fc_min = int(struct.unpack_from("<I", word_document, 24)[0])
    fc_mac = int(struct.unpack_from("<I", word_document, 28)[0])
    base = FibBase(n_fib=n_fib, lid=lid, flags=flags, fc_min=fc_min, fc_mac=fc_mac)

    cursor = 32
    csw = int(struct.unpack_from("<H", word_document, cursor)[0])
    cursor += 2
    rgw, cursor = _read_values(word_document, cursor, csw, 2, "FibRgW")
    if cursor + 2 > len(word_document):
        raise LegacyOfficeMalformedError("truncated FIB cslw")
    cslw = int(struct.unpack_from("<H", word_document, cursor)[0])
    cursor += 2
    rglw, cursor = _read_values(word_document, cursor, cslw, 4, "FibRgLw")
    if cursor + 2 > len(word_document):
        raise LegacyOfficeMalformedError("truncated FIB cbRgFcLcb")
    pair_count = int(struct.unpack_from("<H", word_document, cursor)[0])
    cursor += 2
    raw_pairs, cursor = _read_values(word_document, cursor, pair_count * 2, 4, "FibRgFcLcb")
    pairs = tuple(FcLcb(raw_pairs[index], raw_pairs[index + 1]) for index in range(0, len(raw_pairs), 2))

    csw_new: tuple[int, ...] = ()
    if cursor + 2 <= len(word_document):
        count = int(struct.unpack_from("<H", word_document, cursor)[0])
        cursor += 2
        csw_new, cursor = _read_values(word_document, cursor, count, 2, "FibRgCswNew")
    if len(rglw) <= 3:
        # 确定性最小 fixture 可能省略变长计数，但仍保留 Word 97 固定槽位；
        # 仅在标准布局不可用时按这些公开槽位做恢复读取。
        if len(word_document) < 0x6C:
            raise LegacyOfficeMalformedError("FIB does not contain ccpText")
        rglw = tuple(
            int(struct.unpack_from("<I", word_document, 0x40 + index * 4)[0])
            for index in range(11)
        )
        if not pairs and len(word_document) >= 0x382:
            pairs = tuple(
                FcLcb(*struct.unpack_from("<II", word_document, 0x9A + index * 8))
                for index in range(93)
            )
        cursor = max(cursor, 0x382)
    fib = FileInformationBlock(base=base, rgw=rgw, rglw=rglw, pairs=pairs, csw_new=csw_new, size=cursor)
    if fib.total_story_cp < fib.ccp_text:
        raise LegacyOfficeMalformedError("FIB story CP count overflow")
    return fib
