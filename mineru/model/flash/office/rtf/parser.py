# Copyright (c) Opendatalab. All rights reserved.
"""RTF 1.9.1 常用语义的有界状态机 parser。"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field, replace
import re
from typing import BinaryIO, Literal
from urllib.parse import urlsplit

from loguru import logger

from .....filetypes import rtf_header_offset
from ..errors import LegacyOfficeMalformedError, LegacyOfficeResourceLimitError
from ..limits import MAX_ASSET_TOTAL_BYTES, MAX_ENTRY_BYTES, MAX_GRID_SLOTS
from .lexer import (
    RtfBinary,
    RtfClose,
    RtfControlSymbol,
    RtfControlWord,
    RtfHexByte,
    RtfLexer,
    RtfOpen,
    RtfTextBytes,
)
from .math import parse_rtf_math
from .models import (
    RtfAnchor,
    RtfBlock,
    RtfDisplayEquation,
    RtfDocument,
    RtfImage,
    RtfInline,
    RtfInlineEquation,
    RtfLineBreak,
    RtfListInfo,
    RtfMetadata,
    RtfNote,
    RtfNoteReference,
    RtfParagraph,
    RtfTable,
    RtfTableCell,
    RtfTableRow,
    RtfTextRun,
    RtfTextStyle,
)

MAX_RTF_BYTES = MAX_ENTRY_BYTES
MAX_RTF_LIST_DEPTH = 8
MAX_RTF_TABLE_DEPTH = 4

_CHARSET_ENCODINGS = {
    0: "cp1252",
    2: "cp1252",
    77: "mac_roman",
    128: "cp932",
    129: "cp949",
    130: "cp1361",
    134: "gb18030",
    136: "big5",
    161: "cp1253",
    162: "cp1254",
    163: "cp1258",
    177: "cp1255",
    178: "cp1256",
    186: "cp1257",
    204: "cp1251",
    222: "cp874",
    238: "cp1250",
    255: "cp437",
}
_NFC_MARKERS = {
    0: "decimal",
    1: "upper_roman",
    2: "lower_roman",
    3: "upper_alpha",
    4: "lower_alpha",
    23: "bullet",
    255: "none",
}
_BLOCK_STYLE_NAMES = {
    "html preformatted": "code",
    "preformatted text": "code",
    "source code": "code",
    "block text": "quote",
    "intense quote": "quote",
    "quotation": "quote",
    "quotations": "quote",
    "quote": "quote",
}
_TITLE_STYLE_NAMES = {"document title", "title"}
_HEADER_DESTINATIONS = {"header", "headerf", "headerl", "headerr"}
_FOOTER_DESTINATIONS = {"footer", "footerf", "footerl", "footerr"}
_SUPPRESSED_DESTINATIONS = {
    "annotation",
    "atnauthor",
    "atnid",
    "background",
    "blipuid",
    "colortbl",
    "colorschememapping",
    "datafield",
    "datastore",
    "defchp",
    "defpap",
    "doccomm",
    "docvar",
    "filetbl",
    "fonttbl",
    "generator",
    "info",
    "latentstyles",
    "listoverridetable",
    "listtable",
    "nonshppict",
    "objdata",
    "operator",
    "panose",
    "revtbl",
    "rsidtbl",
    "stylesheet",
    "themedata",
    "userprops",
    "wgrffmtfilter",
    "xmlnstbl",
}
_SPECIAL_WORDS = {
    "bullet": "\u2022",
    "emdash": "\u2014",
    "emspace": " ",
    "endash": "\u2013",
    "enspace": " ",
    "ldblquote": "\u201c",
    "line": "\n",
    "lquote": "\u2018",
    "qmspace": " ",
    "rdblquote": "\u201d",
    "rquote": "\u2019",
    "tab": " ",
}
_HYPERLINK_RE = re.compile(
    r"\bHYPERLINK\b(?P<local>\s+\\l)?\s+(?:\"(?P<quoted>[^\"]*)\"|(?P<bare>\S+))",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(rb"\\(?P<name>[A-Za-z]+)(?P<param>-?\d+)?(?: )?")


@dataclass(frozen=True, slots=True)
class _StyleDefinition:
    """保存 stylesheet 中与语义投影相关的段落样式。"""

    name: str = ""
    outline_level: int | None = None
    is_title: bool = False
    block_style: Literal["normal", "code", "quote"] = "normal"
    based_on: int | None = None
    text_style: RtfTextStyle = RtfTextStyle()


@dataclass(frozen=True, slots=True)
class _ListLevel:
    """保存 RTF listlevel 的起始值和常用编号格式。"""

    marker: str = "decimal"
    start: int = 1

    @property
    def ordered(self) -> bool:
        """返回当前 level 是否应按有序列表输出。"""
        return self.marker not in {"bullet", "none"}


@dataclass(frozen=True, slots=True)
class _ListDefinition:
    """保存一个 list override 对应的列表定义。"""

    identity: int
    levels: tuple[_ListLevel, ...]


@dataclass(slots=True)
class _Prelude:
    """保存正文解析前从 RTF header tables 收敛出的定义。"""

    default_encoding: str = "cp1252"
    font_encodings: dict[int, str] = field(default_factory=dict)
    styles: dict[int, _StyleDefinition] = field(default_factory=dict)
    lists: dict[int, _ListDefinition] = field(default_factory=dict)
    metadata: RtfMetadata = field(default_factory=RtfMetadata)


@dataclass(slots=True)
class _State:
    """保存随 RTF group 继承和恢复的字符、段落及 destination 状态。"""

    style: RtfTextStyle = RtfTextStyle()
    font_id: int | None = None
    uc_skip: int = 1
    hidden: bool = False
    paragraph_style_id: int | None = None
    outline_level: int | None = None
    list_override_id: int | None = None
    list_level: int = 0
    in_table: bool = False
    table_depth: int = 0
    destination: str = "body"


@dataclass(slots=True)
class _CellDefinition:
    """保存 cellx 处冻结的单元格合并属性。"""

    horizontal_merge: Literal["none", "start", "continue"] = "none"
    vertical_merge: Literal["none", "start", "continue"] = "none"
    right_boundary: int | None = None


class _TableBuilder:
    """把 RTF row/cell 控制按当前 table depth 组装为语义表格。"""

    def __init__(self) -> None:
        """初始化空表格和空 row/cell 状态。"""
        self.rows: list[RtfTableRow] = []
        self._definitions: list[_CellDefinition] = []
        self._cells: list[RtfTableCell] = []
        self._cell_blocks: list[RtfBlock] = []
        self._row_header = False
        self._pending_horizontal: Literal["none", "start", "continue"] = "none"
        self._pending_vertical: Literal["none", "start", "continue"] = "none"
        self._row_open = False

    def start_row(self) -> None:
        """开始新行；若上一行未显式结束则先安全收束。"""
        if self._row_open:
            if self._cells or self._cell_blocks:
                # 嵌套表格的 nesttableprops 位于 cell 内容之后，此处只补写定义。
                self._definitions = []
                return
            self.end_row()
        self._definitions = []
        self._cells = []
        self._cell_blocks = []
        self._row_header = False
        self._pending_horizontal = "none"
        self._pending_vertical = "none"
        self._row_open = True

    def set_header(self) -> None:
        """把当前行标记为重复表头行。"""
        self._row_header = True

    def set_horizontal_merge(self, value: Literal["start", "continue"]) -> None:
        """记录下一个 cellx 使用的横向合并属性。"""
        self._pending_horizontal = value

    def set_vertical_merge(self, value: Literal["start", "continue"]) -> None:
        """记录下一个 cellx 使用的纵向合并属性。"""
        self._pending_vertical = value

    def add_definition(self, right_boundary: int | None) -> None:
        """在 cellx 边界冻结当前单元格定义。"""
        if not self._row_open:
            self.start_row()
        self._definitions.append(
            _CellDefinition(
                horizontal_merge=self._pending_horizontal,
                vertical_merge=self._pending_vertical,
                right_boundary=right_boundary,
            )
        )
        definition_index = len(self._definitions) - 1
        if definition_index < len(self._cells):
            definition = self._definitions[definition_index]
            existing = self._cells[definition_index]
            existing.horizontal_merge = definition.horizontal_merge
            existing.vertical_merge = definition.vertical_merge
            existing.right_boundary = definition.right_boundary
        self._pending_horizontal = "none"
        self._pending_vertical = "none"

    def add_block(self, block: RtfBlock) -> None:
        """向当前尚未结束的单元格追加语义块。"""
        if not self._row_open:
            self.start_row()
        self._cell_blocks.append(block)

    def end_cell(self) -> None:
        """结束当前单元格，并按同位置 cellx 定义附加合并属性。"""
        if not self._row_open:
            self.start_row()
        definition_index = len(self._cells)
        definition = (
            self._definitions[definition_index]
            if definition_index < len(self._definitions)
            else _CellDefinition()
        )
        self._cells.append(
            RtfTableCell(
                blocks=self._cell_blocks,
                horizontal_merge=definition.horizontal_merge,
                vertical_merge=definition.vertical_merge,
                right_boundary=definition.right_boundary,
            )
        )
        self._cell_blocks = []

    def end_row(self) -> None:
        """补齐当前行定义的空单元格并写入表格。"""
        if not self._row_open:
            return
        if self._cell_blocks:
            self.end_cell()
        target_cells = max(len(self._definitions), len(self._cells))
        while len(self._cells) < target_cells:
            self.end_cell()
        if target_cells:
            total_slots = sum(len(row.cells) for row in self.rows) + target_cells
            if total_slots > MAX_GRID_SLOTS:
                raise LegacyOfficeResourceLimitError(
                    f"RTF table exceeds max_grid_slots={MAX_GRID_SLOTS}"
                )
            self.rows.append(RtfTableRow(cells=self._cells, header=self._row_header))
        self._definitions = []
        self._cells = []
        self._cell_blocks = []
        self._row_open = False

    def finish(self) -> RtfTable | None:
        """结束未闭合行并返回非空表格。"""
        self.end_row()
        return RtfTable(rows=self.rows) if self.rows else None


@dataclass(slots=True)
class _OutputContext:
    """隔离正文、脚注及页眉页脚各自的块和表格状态。"""

    blocks: list[RtfBlock] = field(default_factory=list)
    inlines: list[RtfInline] = field(default_factory=list)
    pending_list_label: str | None = None
    tables: dict[int, _TableBuilder] = field(default_factory=dict)


@dataclass(slots=True)
class _PictCapture:
    """保存 pict destination 的格式、hex 和 bin 数据。"""

    content_type: str = "application/octet-stream"
    extension: str = "bin"
    hex_data: bytearray = field(default_factory=bytearray)
    binary: bytes | None = None
    alt: str = ""


@dataclass(slots=True)
class _GroupFrame:
    """保存一个 group 的父状态和需要在右花括号处收束的 destination。"""

    previous_state: _State
    start: int
    ignorable: bool = False
    destination: str | None = None
    parent_context: _OutputContext | None = None
    inline_start: int = 0
    capture_text: list[str] = field(default_factory=list)
    instruction: list[str] = field(default_factory=list)
    pict: _PictCapture | None = None
    note_kind: Literal["footnote", "endnote"] = "footnote"


def _lookup_encoding(code_page: int, fallback: str = "cp1252") -> str:
    """把 RTF code page 规范为 Python codec，不支持时返回稳定 fallback。"""
    candidate = f"cp{code_page}"
    try:
        codecs.lookup(candidate)
    except LookupError:
        logger.warning("Unsupported RTF code page {}, falling back to {}", code_page, fallback)
        return fallback
    return candidate


def _default_encoding(data: bytes) -> str:
    """按 RTF header 控制确定默认字符编码。"""
    match = re.search(rb"\\ansicpg(?P<code>\d+)", data[:65536], re.IGNORECASE)
    if match is not None:
        return _lookup_encoding(int(match.group("code")))
    if re.search(rb"\\mac(?:\D|$)", data[:1024], re.IGNORECASE):
        return "mac_roman"
    if re.search(rb"\\pca(?:\D|$)", data[:1024], re.IGNORECASE):
        return "cp850"
    if re.search(rb"\\pc(?:\D|$)", data[:1024], re.IGNORECASE):
        return "cp437"
    return "cp1252"


def _group_slices(data: bytes, *, direct_only: bool = False) -> list[bytes]:
    """按二进制安全 token 边界返回当前输入中的子 group 切片。"""
    depth = 0
    starts: list[int] = []
    result: list[bytes] = []
    for token in RtfLexer(data):
        if isinstance(token, RtfOpen):
            depth += 1
            starts.append(token.start)
        elif isinstance(token, RtfClose) and starts:
            start = starts.pop()
            if (direct_only and depth == 2) or (not direct_only and depth >= 2):
                result.append(data[start : token.end])
            depth = max(depth - 1, 0)
    return result


def _group_destination(group: bytes) -> str | None:
    """读取 group 开头最多两个 control word 以识别 destination。"""
    controls = list(_CONTROL_RE.finditer(group[:256]))
    for match in controls[:3]:
        name = match.group("name").decode("ascii").lower()
        if name not in {"rtf", "ansi", "deff"}:
            return name
    return None


def _named_groups(data: bytes, destination: str) -> list[bytes]:
    """返回全部以指定 control word 开头的 group。"""
    return [group for group in _group_slices(data) if _group_destination(group) == destination]


def _decode_group_text(data: bytes, encoding: str) -> str:
    """解码定义表或 metadata group 中的可见文本，不解释正文结构。"""
    parts: list[str] = []
    uc_skip = 1
    fallback_skip = 0
    pending_high: int | None = None
    for token in RtfLexer(data):
        if isinstance(token, RtfControlWord):
            if token.name == "uc":
                uc_skip = max(token.param or 0, 0)
            elif token.name == "u" and token.param is not None:
                unit = (token.param + 65536 if token.param < 0 else token.param) & 0xFFFF
                if 0xD800 <= unit <= 0xDBFF:
                    pending_high = unit
                elif 0xDC00 <= unit <= 0xDFFF and pending_high is not None:
                    parts.append(chr(0x10000 + ((pending_high - 0xD800) << 10) + unit - 0xDC00))
                    pending_high = None
                else:
                    if pending_high is not None:
                        parts.append("\ufffd")
                        pending_high = None
                    parts.append(chr(unit) if not 0xD800 <= unit <= 0xDFFF else "\ufffd")
                fallback_skip = uc_skip
            elif token.name in _SPECIAL_WORDS:
                parts.append(_SPECIAL_WORDS[token.name])
        elif isinstance(token, RtfHexByte):
            if fallback_skip:
                fallback_skip -= 1
            else:
                parts.append(bytes([token.value]).decode(encoding, errors="replace"))
        elif isinstance(token, RtfTextBytes):
            raw = token.data.replace(b"\r", b"").replace(b"\n", b"")
            if fallback_skip:
                skipped = min(fallback_skip, len(raw))
                raw = raw[skipped:]
                fallback_skip -= skipped
            if raw:
                parts.append(raw.decode(encoding, errors="replace"))
        elif isinstance(token, RtfControlSymbol):
            if token.symbol in {"\\", "{", "}"}:
                parts.append(token.symbol)
            elif token.symbol == "~":
                parts.append("\u00a0")
            elif token.symbol == "_":
                parts.append("-")
    if pending_high is not None:
        parts.append("\ufffd")
    return "".join(parts)


def _parse_fonts(data: bytes, default: str) -> dict[int, str]:
    """解析 fonttbl 中 font id 到字符编码的映射。"""
    groups = _named_groups(data, "fonttbl")
    if not groups:
        return {}
    result: dict[int, str] = {}
    for font_group in _group_slices(groups[0], direct_only=True):
        font_match = re.search(rb"\\f(?P<id>\d+)(?:\D|$)", font_group)
        if font_match is None:
            continue
        font_id = int(font_match.group("id"))
        cpg_match = re.search(rb"\\cpg(?P<code>\d+)", font_group)
        charset_match = re.search(rb"\\fcharset(?P<charset>\d+)", font_group)
        if cpg_match is not None:
            result[font_id] = _lookup_encoding(int(cpg_match.group("code")), default)
        elif charset_match is not None:
            result[font_id] = _CHARSET_ENCODINGS.get(int(charset_match.group("charset")), default)
        else:
            result[font_id] = default
    return result


def _style_control_is_on(data: bytes, name: str) -> bool:
    """读取样式 group 中指定 on/off control 的最后一次显式取值。"""
    pattern = re.compile(
        rb"\\" + name.encode("ascii") + rb"(?P<param>-?\d+)?(?=[^A-Za-z]|$)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(data))
    if not matches:
        return False
    raw_param = matches[-1].group("param")
    return raw_param is None or int(raw_param) != 0


def _parse_styles(data: bytes, encoding: str) -> dict[int, _StyleDefinition]:
    """解析 stylesheet 的标题、outline、代码与引用样式。"""
    groups = _named_groups(data, "stylesheet")
    if not groups:
        return {}
    result: dict[int, _StyleDefinition] = {}
    for style_group in _group_slices(groups[0], direct_only=True):
        style_match = re.search(rb"\\s(?P<id>-?\d+)(?:\D|$)", style_group)
        if style_match is None:
            continue
        style_id = int(style_match.group("id"))
        name = _decode_group_text(style_group, encoding).strip().rstrip(";").strip()
        normalized = name.casefold()
        outline_match = re.search(rb"\\outlinelevel(?P<level>\d+)", style_group)
        outline = int(outline_match.group("level")) if outline_match is not None else None
        if outline is None:
            heading_name = re.fullmatch(r"heading\s+([1-9])", normalized)
            if heading_name is not None:
                outline = int(heading_name.group(1)) - 1
        based_on_match = re.search(rb"\\sbasedon(?P<id>-?\d+)", style_group)
        based_on = int(based_on_match.group("id")) if based_on_match is not None else None
        if "code" in normalized or normalized in {"macro", "macro text"}:
            block_style = "code"
        elif "quote" in normalized:
            block_style = "quote"
        else:
            block_style = _BLOCK_STYLE_NAMES.get(normalized, "normal")
        text_style = RtfTextStyle(
            bold=_style_control_is_on(style_group, "b"),
            italic=_style_control_is_on(style_group, "i"),
            underline=_style_control_is_on(style_group, "ul"),
            strike=_style_control_is_on(style_group, "strike"),
        )
        result[style_id] = _StyleDefinition(
            name=name,
            outline_level=outline,
            is_title=normalized in _TITLE_STYLE_NAMES,
            block_style=block_style,  # type: ignore[arg-type]
            based_on=based_on,
            text_style=text_style,
        )

    resolved: dict[int, _StyleDefinition] = {}

    def resolve(style_id: int, visiting: set[int]) -> _StyleDefinition:
        """递归合并 based-on 样式，循环引用时保留当前显式属性。"""
        if style_id in resolved:
            return resolved[style_id]
        current = result.get(style_id, _StyleDefinition())
        if current.based_on is None or current.based_on in visiting:
            resolved[style_id] = current
            return current
        base = resolve(current.based_on, {*visiting, style_id})
        merged_style = RtfTextStyle(
            bold=current.text_style.bold or base.text_style.bold,
            italic=current.text_style.italic or base.text_style.italic,
            underline=current.text_style.underline or base.text_style.underline,
            strike=current.text_style.strike or base.text_style.strike,
        )
        merged = _StyleDefinition(
            name=current.name,
            outline_level=current.outline_level if current.outline_level is not None else base.outline_level,
            is_title=current.is_title or base.is_title,
            block_style=current.block_style if current.block_style != "normal" else base.block_style,
            based_on=current.based_on,
            text_style=merged_style,
        )
        resolved[style_id] = merged
        return merged

    for style_id in result:
        resolve(style_id, set())
    return resolved


def _parse_lists(data: bytes) -> dict[int, _ListDefinition]:
    """解析 listtable/listoverridetable 的常见编号格式与 override identity。"""
    by_list_id: dict[int, tuple[_ListLevel, ...]] = {}
    list_tables = _named_groups(data, "listtable")
    if list_tables:
        for list_group in _named_groups(list_tables[0], "list"):
            id_match = re.search(rb"\\listid(?P<id>-?\d+)", list_group)
            if id_match is None:
                continue
            levels: list[_ListLevel] = []
            for level_group in _named_groups(list_group, "listlevel")[: MAX_RTF_LIST_DEPTH + 1]:
                nfc_match = re.search(rb"\\levelnfc(?P<nfc>\d+)", level_group)
                start_match = re.search(rb"\\levelstartat(?P<start>-?\d+)", level_group)
                nfc = int(nfc_match.group("nfc")) if nfc_match is not None else 0
                start = int(start_match.group("start")) if start_match is not None else 1
                levels.append(_ListLevel(marker=_NFC_MARKERS.get(nfc, "decimal"), start=max(start, 0)))
            if not levels:
                levels.append(_ListLevel())
            by_list_id[int(id_match.group("id"))] = tuple(levels)

    result: dict[int, _ListDefinition] = {}
    override_tables = _named_groups(data, "listoverridetable")
    if override_tables:
        for override_group in _named_groups(override_tables[0], "listoverride"):
            id_match = re.search(rb"\\listid(?P<id>-?\d+)", override_group)
            ls_match = re.search(rb"\\ls(?P<ls>\d+)", override_group)
            if id_match is None or ls_match is None:
                continue
            list_id = int(id_match.group("id"))
            ls = int(ls_match.group("ls"))
            levels = list(by_list_id.get(list_id, (_ListLevel(),)))
            for level_index, level_override in enumerate(_named_groups(override_group, "lfolevel")):
                while len(levels) <= level_index:
                    levels.append(_ListLevel())
                current = levels[level_index]
                start_match = re.search(rb"\\levelstartat(?P<start>-?\d+)", level_override)
                nfc_match = re.search(rb"\\levelnfc(?P<nfc>\d+)", level_override)
                start = max(int(start_match.group("start")), 0) if start_match is not None else current.start
                marker = _NFC_MARKERS.get(int(nfc_match.group("nfc")), current.marker) if nfc_match is not None else current.marker
                levels[level_index] = _ListLevel(marker=marker, start=start)
            result[ls] = _ListDefinition(identity=ls, levels=tuple(levels))
    return result


def _parse_metadata(data: bytes, encoding: str) -> RtfMetadata:
    """解析 info destination 中允许公开的四个字符串字段。"""
    info_groups = _named_groups(data, "info")
    if not info_groups:
        return RtfMetadata()
    values: dict[str, str | None] = {}
    for name in ("title", "author", "subject", "keywords"):
        groups = _named_groups(info_groups[0], name)
        value = _decode_group_text(groups[0], encoding).strip() if groups else ""
        values[name] = value or None
    return RtfMetadata(**values)


def parse_rtf_prelude(data: bytes) -> _Prelude:
    """解析 RTF header tables、列表和 metadata，供正文 parser 与 doclib 共用。"""
    offset = rtf_header_offset(data[:128])
    if offset is None:
        raise LegacyOfficeMalformedError("not an RTF document")
    normalized = data[offset:]
    default = _default_encoding(normalized)
    return _Prelude(
        default_encoding=default,
        font_encodings=_parse_fonts(normalized, default),
        styles=_parse_styles(normalized, default),
        lists=_parse_lists(normalized),
        metadata=_parse_metadata(normalized, default),
    )


def _roman(value: int) -> str:
    """把正整数格式化为常见 Roman 编号。"""
    if value <= 0:
        return str(value)
    pairs = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    parts: list[str] = []
    remaining = value
    for number, label in pairs:
        while remaining >= number:
            parts.append(label)
            remaining -= number
    return "".join(parts)


def _alpha(value: int) -> str:
    """把正整数格式化为 Excel 风格字母编号。"""
    if value <= 0:
        return str(value)
    parts: list[str] = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining - 1, 26)
        parts.append(chr(ord("A") + digit))
    return "".join(reversed(parts))


def _format_marker(marker: str, value: int) -> str:
    """按常见 RTF levelnfc marker 格式化一个编号。"""
    if marker == "upper_roman":
        return _roman(value)
    if marker == "lower_roman":
        return _roman(value).lower()
    if marker == "upper_alpha":
        return _alpha(value)
    if marker == "lower_alpha":
        return _alpha(value).lower()
    return str(value)


def _capture_picture_group(data: bytes) -> _PictCapture | None:
    """从独立 pict group 中提取 direct hex/bin，用于 Office Math 图片 fallback。"""
    capture = _PictCapture()
    depth = 0
    for token in RtfLexer(data):
        if isinstance(token, RtfOpen):
            depth += 1
            continue
        if isinstance(token, RtfClose):
            depth = max(depth - 1, 0)
            continue
        if isinstance(token, RtfControlWord) and depth == 1:
            if token.name == "pngblip":
                capture.content_type, capture.extension = "image/png", "png"
            elif token.name == "jpegblip":
                capture.content_type, capture.extension = "image/jpeg", "jpg"
            elif token.name == "emfblip":
                capture.content_type, capture.extension = "image/x-emf", "emf"
            elif token.name == "wmetafile":
                capture.content_type, capture.extension = "image/x-wmf", "wmf"
            elif token.name in {"dibitmap", "wbitmap"}:
                capture.content_type, capture.extension = "image/bmp", "dib"
        elif isinstance(token, RtfBinary) and depth == 1:
            capture.binary = token.data
        elif isinstance(token, RtfHexByte) and depth == 1:
            capture.hex_data.extend(f"{token.value:02x}".encode("ascii"))
        elif isinstance(token, RtfTextBytes) and depth == 1:
            capture.hex_data.extend(token.data)
    if capture.binary is None and not capture.hex_data:
        return None
    return capture


def _safe_hyperlink_target(target: str, *, local: bool) -> str | None:
    """保留安全文档链接；活动协议和协议相对地址一律降级。"""
    normalized = target.strip()
    if not normalized or any(ord(char) < 0x20 for char in normalized):
        return None
    if local:
        return f"#{normalized.lstrip('#')}" if normalized.lstrip("#") else None
    if normalized.startswith(("//", "\\")):
        return None
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return None
    if not parsed.scheme:
        return normalized
    if parsed.scheme.casefold() not in {"http", "https", "mailto", "tel"}:
        return None
    if parsed.scheme.casefold() in {"http", "https"} and (not parsed.netloc or parsed.hostname is None):
        return None
    return normalized


class RtfParser:
    """把一个 RTF 字节串解析为无布局、单逻辑页的 typed document。"""

    def __init__(self, data: bytes, prelude: _Prelude | None = None) -> None:
        """校验输入大小和根组，并初始化所有每文档状态。"""
        if len(data) > MAX_RTF_BYTES:
            raise LegacyOfficeResourceLimitError(f"RTF input exceeds max_bytes={MAX_RTF_BYTES}")
        offset = rtf_header_offset(data[:128])
        if offset is None:
            raise LegacyOfficeMalformedError("not an RTF document")
        self.data = data[offset:]
        self.prelude = prelude or parse_rtf_prelude(data)
        self.state = _State()
        self.frames: list[_GroupFrame] = []
        self.context = _OutputContext()
        self.document = RtfDocument(metadata=self.prelude.metadata)
        self._byte_buffer = bytearray()
        self._fallback_skip = 0
        self._pending_high_surrogate: int | None = None
        self._list_counters: dict[tuple[int, int], int] = {}
        self._asset_total = 0
        self._recovered = False

    def parse(self) -> RtfDocument:
        """运行状态机，恢复未闭合组并返回 typed RTF 文档。"""
        for token in RtfLexer(self.data):
            if isinstance(token, RtfOpen):
                self._open_group(token)
            elif isinstance(token, RtfClose):
                self._close_group(token)
            elif isinstance(token, RtfControlSymbol):
                self._control_symbol(token)
            elif isinstance(token, RtfControlWord):
                self._control_word(token)
            elif isinstance(token, RtfHexByte):
                self._hex_byte(token)
            elif isinstance(token, RtfTextBytes):
                self._text_bytes(token)
            elif isinstance(token, RtfBinary):
                self._binary(token)

        self._flush_bytes()
        while self.frames:
            self._recovered = True
            frame = self.frames.pop()
            self._finish_destination(frame, len(self.data))
            self.state = frame.previous_state
        self._finalize_context(self.context)
        self.document.blocks = self.context.blocks
        if self._recovered:
            logger.warning("Recovered unbalanced RTF groups")
        return self.document

    def _open_group(self, token: RtfOpen) -> None:
        """压入当前状态，新的 group 初始继承所有属性。"""
        self._flush_bytes()
        self.frames.append(_GroupFrame(previous_state=replace(self.state), start=token.start))

    def _close_group(self, token: RtfClose) -> None:
        """收束当前 destination 并恢复父 group 状态。"""
        self._flush_bytes()
        if not self.frames:
            self._recovered = True
            return
        frame = self.frames.pop()
        self._finish_destination(frame, token.end)
        self.state = frame.previous_state

    def _current_frame(self) -> _GroupFrame | None:
        """返回当前最内层 group frame。"""
        return self.frames[-1] if self.frames else None

    def _nearest_frame(self, destination: str) -> _GroupFrame | None:
        """从内向外查找负责指定 destination 的 frame。"""
        return next((frame for frame in reversed(self.frames) if frame.destination == destination), None)

    def _start_destination(self, name: str) -> bool:
        """识别 group destination 并初始化其隔离输出或捕获状态。"""
        frame = self._current_frame()
        if frame is None or frame.destination is not None:
            return False
        if self.state.destination in {"math", "pict", "suppressed"}:
            return False
        if name in _SUPPRESSED_DESTINATIONS:
            frame.destination = name
            self.state.destination = "suppressed"
            return True
        if name == "field":
            frame.destination = "field"
            frame.inline_start = len(self.context.inlines)
            return True
        if name == "object":
            frame.destination = "object"
            self.state.destination = "object"
            return True
        if name == "result" and self.state.destination == "object":
            frame.destination = "result"
            self.state.destination = "body"
            return True
        if name == "fldinst":
            frame.destination = "fldinst"
            self.state.destination = "field_instruction"
            return True
        if name == "fldrslt":
            frame.destination = "fldrslt"
            self.state.destination = "body"
            return True
        if name in {"footnote", "endnote"}:
            frame.destination = "note"
            frame.note_kind = "endnote" if name == "endnote" else "footnote"
            frame.parent_context = self.context
            self.context = _OutputContext()
            self.state.destination = "body"
            return True
        if name in _HEADER_DESTINATIONS | _FOOTER_DESTINATIONS:
            frame.destination = "header" if name in _HEADER_DESTINATIONS else "footer"
            frame.parent_context = self.context
            self.context = _OutputContext()
            self.state.destination = "body"
            return True
        if name == "pict":
            frame.destination = "pict"
            frame.pict = _PictCapture()
            self.state.destination = "pict"
            return True
        if name == "mmath":
            frame.destination = "math"
            self.state.destination = "math"
            return True
        if name in {"listtext", "pntext"}:
            frame.destination = "listtext"
            self.state.destination = "listtext"
            return True
        if name == "bkmkstart":
            frame.destination = "bookmark"
            self.state.destination = "bookmark"
            return True
        if name == "bkmkend":
            frame.destination = "bookmark_end"
            self.state.destination = "suppressed"
            return True
        if name == "nonshppict":
            frame.destination = name
            self.state.destination = "suppressed"
            return True
        if name == "shppict":
            frame.destination = name
            return True
        if name == "nesttableprops":
            frame.destination = name
            return True
        if name == "pn":
            frame.destination = name
            self.state.list_override_id = self.state.list_override_id or -1
            return True
        if frame.ignorable:
            frame.destination = "unknown"
            self.state.destination = "suppressed"
            return True
        return False

    def _finish_destination(self, frame: _GroupFrame, end: int) -> None:
        """在 group 结束处物化 field、note、pict、math 和捕获文本。"""
        destination = frame.destination
        if destination == "field":
            self._finish_field(frame)
        elif destination == "note":
            self._finish_note(frame)
        elif destination in {"header", "footer"}:
            self._finish_auxiliary(frame, destination)
        elif destination == "pict":
            self._finish_picture(frame)
        elif destination == "math":
            self._finish_math(frame, end)
        elif destination == "listtext":
            label = "".join(frame.capture_text).replace("\t", " ").strip()
            if label:
                self.context.pending_list_label = label
        elif destination == "bookmark":
            name = "".join(frame.capture_text).strip()
            if name:
                self.context.inlines.append(RtfAnchor(name))

    def _finish_field(self, frame: _GroupFrame) -> None:
        """把安全 HYPERLINK field result 包装回行内 run。"""
        instruction = "".join(frame.instruction).strip()
        match = _HYPERLINK_RE.search(instruction)
        if match is None:
            return
        raw_target = match.group("quoted") or match.group("bare") or ""
        target = _safe_hyperlink_target(raw_target, local=bool(match.group("local")))
        if target is None:
            return
        start = min(frame.inline_start, len(self.context.inlines))
        for index in range(start, len(self.context.inlines)):
            inline = self.context.inlines[index]
            if isinstance(inline, RtfTextRun):
                self.context.inlines[index] = replace(inline, hyperlink=target)

    def _finish_note(self, frame: _GroupFrame) -> None:
        """结束隔离 note context，在父正文插入引用并登记正文。"""
        note_context = self.context
        self._finalize_context(note_context)
        parent = frame.parent_context or _OutputContext()
        self.context = parent
        if not note_context.blocks:
            return
        note_id = f"rtf{len(self.document.notes)}"
        self.document.notes.append(RtfNote(note_id, frame.note_kind, note_context.blocks))
        self.context.inlines.append(RtfNoteReference(note_id))

    def _finish_auxiliary(self, frame: _GroupFrame, destination: str) -> None:
        """结束页眉页脚隔离 context，并恢复父正文。"""
        auxiliary_context = self.context
        self._finalize_context(auxiliary_context)
        self.context = frame.parent_context or _OutputContext()
        target = self.document.headers if destination == "header" else self.document.footers
        target.extend(auxiliary_context.blocks)

    def _finish_picture(self, frame: _GroupFrame) -> None:
        """校验 pict 大小并向当前行内流追加图片。"""
        capture = frame.pict
        if capture is None:
            return
        image = self._materialize_picture(capture)
        if image is not None:
            self.context.inlines.append(image)

    def _materialize_picture(self, capture: _PictCapture) -> RtfImage | None:
        """把已捕获 pict 转成有界图片载荷，并累计文档素材预算。"""
        if capture.binary is not None:
            payload = capture.binary
        else:
            compact = bytes(character for character in capture.hex_data if chr(character).strip())
            if len(compact) % 2:
                logger.warning("Skipping RTF pict with odd hex length")
                return
            try:
                payload = bytes.fromhex(compact.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                logger.warning("Skipping malformed RTF pict hex payload")
                return
        if not payload:
            return
        if len(payload) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(f"RTF pict exceeds max_entry_bytes={MAX_ENTRY_BYTES}")
        if self._asset_total + len(payload) > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(
                f"RTF pict assets exceed max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}"
            )
        self._asset_total += len(payload)
        return RtfImage(
            data=payload,
            content_type=capture.content_type,
            part_name=f"pict.{capture.extension}",
            alt=capture.alt,
        )

    def _finish_math(self, frame: _GroupFrame, end: int) -> None:
        """把 math group 转换为行内或行间 LaTeX，失败时静默保留其余正文。"""
        formulas, display = parse_rtf_math(
            self.data[frame.start:end],
            encoding=self._current_encoding(),
        )
        if not formulas:
            for group in _named_groups(self.data[frame.start:end], "pict"):
                capture = _capture_picture_group(group)
                if capture is None:
                    continue
                image = self._materialize_picture(capture)
                if image is not None:
                    self.context.inlines.append(image)
                    break
            return
        if display:
            self._end_paragraph()
            self._flush_tables()
            self.context.blocks.extend(RtfDisplayEquation(formula) for formula in formulas)
        else:
            self.context.inlines.extend(RtfInlineEquation(formula) for formula in formulas)

    def _control_symbol(self, token: RtfControlSymbol) -> None:
        """处理 ignorable marker、转义结构字符和特殊空白。"""
        self._flush_bytes()
        frame = self._current_frame()
        if token.symbol == "*" and frame is not None:
            frame.ignorable = True
            return
        if self._fallback_skip:
            self._fallback_skip -= 1
            return
        if token.symbol == "\n":
            self._end_paragraph()
        elif token.symbol in {"\\", "{", "}"}:
            self._append_text(token.symbol)
        elif token.symbol == "~":
            self._append_text("\u00a0")
        elif token.symbol == "_":
            self._append_text("-")

    def _control_word(self, token: RtfControlWord) -> None:
        """按 destination、文本、表格和列表的固定顺序解释 control word。"""
        self._flush_bytes()
        if self._start_destination(token.name):
            return
        if token.name == "ftnalt":
            note_frame = self._nearest_frame("note")
            if note_frame is not None:
                note_frame.note_kind = "endnote"
            return
        if self.state.destination == "pict":
            self._pict_control(token)
            return
        if self.state.destination in {"math", "suppressed"}:
            return
        if token.name == "u":
            self._unicode(token.param)
            return
        if token.name == "uc":
            self.state.uc_skip = max(token.param or 0, 0)
            return
        if token.name == "f":
            self.state.font_id = token.param
            return
        if token.name == "s" and token.param is not None:
            self.state.paragraph_style_id = token.param
            definition = self.prelude.styles.get(token.param)
            if definition is not None and definition.outline_level is not None:
                self.state.outline_level = definition.outline_level
            if definition is not None:
                self.state.style = definition.text_style
            return
        if token.name == "outlinelevel" and token.param is not None:
            self.state.outline_level = max(token.param, 0)
            return
        if token.name == "b":
            self.state.style = replace(self.state.style, bold=token.param != 0)
            return
        if token.name == "i":
            self.state.style = replace(self.state.style, italic=token.param != 0)
            return
        if token.name in {"ul", "uld", "uldash", "uldb", "ulth", "ulw"}:
            self.state.style = replace(self.state.style, underline=token.param != 0)
            return
        if token.name in {"ulnone", "ul0"}:
            self.state.style = replace(self.state.style, underline=False)
            return
        if token.name in {"strike", "striked"}:
            self.state.style = replace(self.state.style, strike=token.param != 0)
            return
        if token.name == "super":
            self.state.style = replace(self.state.style, superscript=True, subscript=False)
            return
        if token.name == "sub":
            self.state.style = replace(self.state.style, superscript=False, subscript=True)
            return
        if token.name == "nosupersub":
            self.state.style = replace(self.state.style, superscript=False, subscript=False)
            return
        if token.name == "v":
            self.state.hidden = token.param != 0
            return
        if token.name == "plain":
            self.state.style = RtfTextStyle()
            self.state.hidden = False
            return
        if token.name == "pard":
            if self.state.table_depth > 1:
                self._set_table_depth(self.state.table_depth - 1)
            self.state.table_depth = 0
            self.state.in_table = False
            self.state.paragraph_style_id = None
            self.state.outline_level = None
            self.state.list_override_id = None
            self.state.list_level = 0
            return
        if token.name in {"par", "sect"}:
            self._end_paragraph()
            return
        if token.name in {"page", "column", "lbr"}:
            self._append_inline(RtfLineBreak())
            return
        if token.name in _SPECIAL_WORDS:
            self._append_text(_SPECIAL_WORDS[token.name])
            return
        if token.name == "chftn":
            return
        if self._table_control(token):
            return
        self._list_control(token)

    def _table_control(self, token: RtfControlWord) -> bool:
        """解释常见 row/cell/table-depth 与合并控制。"""
        name = token.name
        if name == "itap":
            self._set_table_depth(max(token.param or 0, 0))
            return True
        depth = max(self.state.table_depth, 1)
        if name == "trowd":
            self.state.in_table = True
            self.state.list_override_id = None
            self.state.list_level = 0
            self.context.pending_list_label = None
            if self.state.table_depth == 0:
                self.state.table_depth = 1
            self._table_builder(depth).start_row()
            return True
        if name == "intbl":
            self.state.in_table = token.param != 0
            if self.state.in_table and self.state.table_depth == 0:
                self.state.table_depth = 1
            return True
        if name == "trhdr":
            self._table_builder(depth).set_header()
            return True
        if name == "clmgf":
            self._table_builder(depth).set_horizontal_merge("start")
            return True
        if name == "clmrg":
            self._table_builder(depth).set_horizontal_merge("continue")
            return True
        if name == "clvmgf":
            self._table_builder(depth).set_vertical_merge("start")
            return True
        if name == "clvmrg":
            self._table_builder(depth).set_vertical_merge("continue")
            return True
        if name == "cellx":
            self._table_builder(depth).add_definition(token.param)
            return True
        if name in {"cell", "nestcell"}:
            self._end_paragraph()
            self._table_builder(depth).end_cell()
            return True
        if name in {"row", "nestrow"}:
            self._end_paragraph()
            self._table_builder(depth).end_row()
            return True
        return False

    def _list_control(self, token: RtfControlWord) -> bool:
        """记录现代 ls/ilvl 及常见 legacy pn marker。"""
        if token.name == "ls" and token.param is not None:
            self.state.list_override_id = token.param
            return True
        if token.name == "ilvl" and token.param is not None:
            self.state.list_level = min(max(token.param, 0), MAX_RTF_LIST_DEPTH)
            return True
        if token.name == "pnlvlblt":
            self.state.list_override_id = self.state.list_override_id or -1
            return True
        if token.name in {"pndec", "pnucrm", "pnlcrm", "pnucltr", "pnlcltr"}:
            self.state.list_override_id = self.state.list_override_id or -2
            return True
        return False

    def _pict_control(self, token: RtfControlWord) -> None:
        """记录 pict 格式；尺寸与裁剪控制不影响无布局语义。"""
        frame = self._nearest_frame("pict")
        capture = frame.pict if frame is not None else None
        if capture is None:
            return
        if token.name == "pngblip":
            capture.content_type, capture.extension = "image/png", "png"
        elif token.name == "jpegblip":
            capture.content_type, capture.extension = "image/jpeg", "jpg"
        elif token.name == "emfblip":
            capture.content_type, capture.extension = "image/x-emf", "emf"
        elif token.name == "wmetafile":
            capture.content_type, capture.extension = "image/x-wmf", "wmf"
        elif token.name in {"dibitmap", "wbitmap"}:
            capture.content_type, capture.extension = "image/bmp", "dib"

    def _hex_byte(self, token: RtfHexByte) -> None:
        """把 hex byte 送入 pict 或当前代码页缓冲。"""
        if self.state.destination == "pict":
            frame = self._nearest_frame("pict")
            if frame is not None and frame is self._current_frame() and frame.pict is not None:
                frame.pict.hex_data.extend(f"{token.value:02x}".encode("ascii"))
            return
        if self.state.destination in {"math", "suppressed"}:
            return
        if self._fallback_skip:
            self._fallback_skip -= 1
            return
        self._byte_buffer.append(token.value)

    def _text_bytes(self, token: RtfTextBytes) -> None:
        """规范源换行后把文本字节送入 destination 或代码页缓冲。"""
        if self.state.destination == "pict":
            frame = self._nearest_frame("pict")
            if frame is not None and frame is self._current_frame() and frame.pict is not None:
                frame.pict.hex_data.extend(token.data)
            return
        if self.state.destination in {"math", "suppressed"}:
            return
        raw = token.data.replace(b"\r", b"").replace(b"\n", b"")
        if self._fallback_skip:
            skipped = min(self._fallback_skip, len(raw))
            raw = raw[skipped:]
            self._fallback_skip -= skipped
        self._byte_buffer.extend(raw)

    def _binary(self, token: RtfBinary) -> None:
        """只允许 pict destination 消费 bin 载荷，其他二进制内容直接跳过。"""
        if self.state.destination != "pict":
            return
        frame = self._nearest_frame("pict")
        if frame is not None and frame is self._current_frame() and frame.pict is not None:
            frame.pict.binary = token.data

    def _unicode(self, value: int | None) -> None:
        """解码有符号 UTF-16 code unit，合并代理对并启动 fallback skip。"""
        if value is None:
            return
        unit = (value + 65536 if value < 0 else value) & 0xFFFF
        if 0xD800 <= unit <= 0xDBFF:
            if self._pending_high_surrogate is not None:
                self._append_text("\ufffd")
            self._pending_high_surrogate = unit
        elif 0xDC00 <= unit <= 0xDFFF and self._pending_high_surrogate is not None:
            codepoint = 0x10000 + ((self._pending_high_surrogate - 0xD800) << 10) + unit - 0xDC00
            self._append_text(chr(codepoint))
            self._pending_high_surrogate = None
        else:
            if self._pending_high_surrogate is not None:
                self._append_text("\ufffd")
                self._pending_high_surrogate = None
            self._append_text(chr(unit) if not 0xD800 <= unit <= 0xDFFF else "\ufffd")
        self._fallback_skip = self.state.uc_skip

    def _current_encoding(self) -> str:
        """返回当前 font 的代码页或文档默认代码页。"""
        if self.state.font_id is None:
            return self.prelude.default_encoding
        return self.prelude.font_encodings.get(self.state.font_id, self.prelude.default_encoding)

    def _flush_bytes(self) -> None:
        """使用当前 font code page 解码累计字节并写入当前 destination。"""
        if not self._byte_buffer:
            return
        payload = bytes(self._byte_buffer)
        self._byte_buffer.clear()
        self._append_text(payload.decode(self._current_encoding(), errors="replace"))

    def _append_text(self, text: str) -> None:
        """按当前 destination 把文本写入 field、bookmark、list label 或正文。"""
        if not text or self.state.hidden:
            return
        if self.state.destination == "field_instruction":
            field_frame = self._nearest_frame("field")
            if field_frame is not None:
                field_frame.instruction.append(text)
            return
        if self.state.destination in {"bookmark", "listtext"}:
            frame = self._nearest_frame(self.state.destination)
            if frame is not None:
                frame.capture_text.append(text)
            return
        if self.state.destination != "body":
            return
        self._append_inline(RtfTextRun(text=text, style=self.state.style))

    def _append_inline(self, inline: RtfInline) -> None:
        """追加行内节点，并合并样式和链接完全相同的相邻文本 run。"""
        if isinstance(inline, RtfTextRun) and self.context.inlines:
            previous = self.context.inlines[-1]
            if (
                isinstance(previous, RtfTextRun)
                and previous.style == inline.style
                and previous.hyperlink == inline.hyperlink
            ):
                self.context.inlines[-1] = replace(previous, text=f"{previous.text}{inline.text}")
                return
        self.context.inlines.append(inline)

    def _resolve_list_info(self) -> RtfListInfo | None:
        """把 paragraph 的 ls/ilvl 和精确 listtext 收敛为显式列表信息。"""
        identity = self.state.list_override_id
        label = self.context.pending_list_label
        if identity is None and not label:
            return None
        level = min(max(self.state.list_level, 0), MAX_RTF_LIST_DEPTH)
        definition = self.prelude.lists.get(identity or 0)
        if definition is not None:
            level_def = definition.levels[min(level, len(definition.levels) - 1)]
            resolved_identity = definition.identity
        else:
            marker = "bullet" if label and any(char in label for char in ("\u2022", "\u00b7", "\uf0b7")) else "decimal"
            level_def = _ListLevel(marker=marker)
            resolved_identity = identity or 0
        counter_key = (resolved_identity, level)
        current = self._list_counters.get(counter_key, level_def.start - 1) + 1
        self._list_counters[counter_key] = current
        ordered = level_def.ordered
        if label:
            stripped_label = label.strip()
            if stripped_label in {"\u2022", "\u00b7", "\uf0b7", "o"}:
                ordered = False
        elif ordered:
            label = f"{_format_marker(level_def.marker, current)}."
        return RtfListInfo(
            identity=resolved_identity,
            level=level,
            ordered=ordered,
            marker=level_def.marker,
            start=level_def.start,
            label=label,
        )

    def _end_paragraph(self) -> None:
        """把当前行内流冻结为段落，并路由到正文或当前表格单元格。"""
        self._flush_bytes()
        if self._pending_high_surrogate is not None:
            self._append_text("\ufffd")
            self._pending_high_surrogate = None
        inlines = self.context.inlines
        self.context.inlines = []
        list_info = self._resolve_list_info()
        self.context.pending_list_label = None
        has_visible = any(
            not isinstance(inline, (RtfTextRun, RtfAnchor))
            or (isinstance(inline, RtfTextRun) and bool(inline.text.strip()))
            for inline in inlines
        )
        if not has_visible and not any(isinstance(inline, RtfAnchor) for inline in inlines):
            return
        style_definition = self.prelude.styles.get(self.state.paragraph_style_id or -1, _StyleDefinition())
        outline = self.state.outline_level
        if outline is None:
            outline = style_definition.outline_level
        paragraph = RtfParagraph(
            inlines=inlines,
            style_name=style_definition.name,
            outline_level=outline,
            is_title=style_definition.is_title,
            block_style=style_definition.block_style,
            list_info=list_info,
        )
        active_depth = max(self.state.table_depth, 1)
        if self.state.in_table:
            self._table_builder(active_depth).add_block(paragraph)
            return
        self._flush_tables()
        self.context.blocks.append(paragraph)

    def _table_builder(self, depth: int) -> _TableBuilder:
        """返回指定 depth 的 builder，并拒绝超出渲染能力的嵌套。"""
        if depth < 1 or depth > MAX_RTF_TABLE_DEPTH:
            raise LegacyOfficeResourceLimitError(
                f"RTF table nesting exceeds max_table_depth={MAX_RTF_TABLE_DEPTH}"
            )
        return self.context.tables.setdefault(depth, _TableBuilder())

    def _set_table_depth(self, depth: int) -> None:
        """切换 table depth，并把已结束的深层表格挂回父 cell。"""
        normalized = min(max(depth, 0), MAX_RTF_TABLE_DEPTH)
        if depth > MAX_RTF_TABLE_DEPTH:
            raise LegacyOfficeResourceLimitError(
                f"RTF table nesting exceeds max_table_depth={MAX_RTF_TABLE_DEPTH}"
            )
        for current_depth in sorted(
            [value for value in self.context.tables if value > normalized],
            reverse=True,
        ):
            table = self.context.tables.pop(current_depth).finish()
            if table is None:
                continue
            if current_depth > 1:
                self._table_builder(current_depth - 1).add_block(table)
            else:
                self.context.blocks.append(table)
        self.state.table_depth = normalized

    def _flush_tables(self) -> None:
        """从深到浅结束当前 context 的全部 table builder。"""
        for depth in sorted(self.context.tables, reverse=True):
            table = self.context.tables[depth].finish()
            if table is None:
                continue
            if depth > 1:
                self._table_builder(depth - 1).add_block(table)
            else:
                self.context.blocks.append(table)
        self.context.tables.clear()

    def _finalize_context(self, context: _OutputContext) -> None:
        """结束一个输出 context 的残留段落和表格。"""
        if context is not self.context:
            current = self.context
            self.context = context
            self._end_paragraph()
            self._flush_tables()
            self.context = current
            return
        self._end_paragraph()
        self._flush_tables()


def read_rtf_bytes(file_binary: BinaryIO) -> bytes:
    """从二进制流头部读取有界 RTF 输入，并恢复调用前流位置。"""
    try:
        original_position = file_binary.tell()
    except (AttributeError, OSError):
        original_position = None
    if original_position is not None:
        file_binary.seek(0)
    data = file_binary.read(MAX_RTF_BYTES + 1)
    if original_position is not None:
        file_binary.seek(original_position)
    if len(data) > MAX_RTF_BYTES:
        raise LegacyOfficeResourceLimitError(f"RTF input exceeds max_bytes={MAX_RTF_BYTES}")
    return data


def parse_rtf(file_binary: BinaryIO) -> RtfDocument:
    """读取一个 RTF 二进制流并返回单逻辑页 typed document。"""
    data = read_rtf_bytes(file_binary)
    prelude = parse_rtf_prelude(data)
    return RtfParser(data, prelude).parse()


__all__ = [
    "MAX_RTF_BYTES",
    "MAX_RTF_LIST_DEPTH",
    "MAX_RTF_TABLE_DEPTH",
    "RtfParser",
    "parse_rtf",
    "parse_rtf_prelude",
    "read_rtf_bytes",
]
