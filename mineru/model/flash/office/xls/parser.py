# Copyright (c) Opendatalab. All rights reserved.

"""纯 Python 解析 Excel 97–2003 Workbook BIFF stream。"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from urllib.parse import urlparse
import uuid

from loguru import logger

from ..errors import (
    LegacyOfficeEncryptedError,
    LegacyOfficeMalformedError,
)
from ..legacy.binary import get_f64, get_u16, get_u32
from ..legacy.officeart import (
    OfficeArtShape,
    OfficeImagePayload,
    decode_bstore,
    extract_excel_shapes,
)
from ..image import serialize_office_image
from ..equation.image import (
    OfficeImageEquationDecoder,
)

from .chart import chart_source_axes, chart_source_selection
from .models import (
    XlsCell,
    XlsChart,
    XlsChartSheet,
    XlsEquation,
    XlsFontStyle,
    XlsImage,
    XlsRichRun,
    XlsRichText,
    XlsSheet,
    XlsWorkbook,
)
from .number_format import builtin_number_format, format_number, format_text
from .records import (
    BOF,
    CONTINUE,
    EOF,
    BiffRecord,
    RecordBudget,
    SegmentReader,
    collect_continues,
    iter_records,
    record_at,
)
from .strings import (
    DecodedString,
    clean_text,
    codepage_name,
    read_biff8_string,
    read_byte_string,
    read_txo_text,
    to_rich_text,
)

FILEPASS = 0x002F
CODEPAGE = 0x0042
DATEMODE = 0x0022
BOUNDSHEET = 0x0085
SST = 0x00FC
FORMAT = 0x041E
XF = 0x00E0
FONT = 0x0031
ROW = 0x0208
COLINFO = 0x007D
MERGEDCELLS = 0x00E5
LABELSST = 0x00FD
LABEL = 0x0204
RSTRING = 0x00D6
NUMBER = 0x0203
RK = 0x027E
MULRK = 0x00BD
BOOLERR = 0x0205
FORMULA = 0x0006
STRING = 0x0207
MSODRAWINGGROUP = 0x00EB
MSODRAWING = 0x00EC
OBJ = 0x005D
TXO = 0x01B6
HLINK = 0x01B8
SUPBOOK = 0x01AE
EXTERNSHEET = 0x0017
WINDOW1 = 0x003D

WORKBOOK_GLOBALS_SUBSTREAM = 0x0005
WORKSHEET_SUBSTREAM = 0x0010
CHART_SUBSTREAM = 0x0020
MAX_ROWS = 65_536
MAX_COLS = 256

OBJ_CHART = 0x0005
OBJ_TEXTBOX = 0x0006
OBJ_PICTURE = 0x0008
OBJ_CHECKBOX = 0x000B


@dataclass(frozen=True, slots=True)
class _BoundSheet:
    """BoundSheet8 目录项。"""

    name: str
    offset: int
    visible: bool
    sheet_type: int


@dataclass(frozen=True, slots=True)
class _CellFormat:
    """XF 解析后的字体索引和数值格式代码。"""

    font_index: int
    format_code: str | None


@dataclass(slots=True)
class _Globals:
    """Workbook Globals Substream 中供所有工作表共享的状态。"""

    biff8: bool
    date1904: bool = False
    encoding: str = "cp1252"
    sheets: list[_BoundSheet] = field(default_factory=list)
    strings: list[DecodedString] = field(default_factory=list)
    fonts: list[XlsFontStyle] = field(default_factory=list)
    formats: list[_CellFormat] = field(default_factory=list)
    images: dict[int, OfficeImagePayload] = field(default_factory=dict)
    extern_sheets: list[int | None] = field(default_factory=list)
    active_sheet_index: int | None = None

    def read_string(
        self,
        reader: SegmentReader,
        *,
        short: bool,
        rich: bool = False,
    ) -> DecodedString | None:
        """按当前 BIFF 版本和 codepage 读取字符串。"""

        if self.biff8:
            return read_biff8_string(reader, short=short, rich=rich)
        return read_byte_string(reader, short=short, encoding=self.encoding)

    def cell_format(self, index: int) -> _CellFormat:
        """解析越界 XF 时返回 General 与无字体的稳定默认值。"""

        if 0 <= index < len(self.formats):
            return self.formats[index]
        return _CellFormat(font_index=0, format_code=None)


@dataclass(slots=True)
class _SheetObject:
    """一个 OBJ 记录及其后续 TXO 可见文本。"""

    object_type: int
    object_id: int
    checked: bool | None = None
    embedding_storage: str | None = None
    text: XlsRichText | None = None
    shape: OfficeArtShape | None = None


def _error_literal(code: int) -> str | None:
    """把 BIFF error code 转成 Excel 可见错误文本。"""

    return {
        0x00: "#NULL!",
        0x07: "#DIV/0!",
        0x0F: "#VALUE!",
        0x17: "#REF!",
        0x1D: "#NAME?",
        0x24: "#NUM!",
        0x2A: "#N/A",
        0x2B: "#GETTING_DATA",
    }.get(int(code))


def _rk_number(value: int) -> float:
    """解码 RK 压缩整数或截断双精度数。"""

    if value & 0x02:
        signed = struct.unpack("<i", struct.pack("<I", value))[0]
        number = float(signed >> 2)
    else:
        number = struct.unpack("<d", struct.pack("<Q", (value & 0xFFFF_FFFC) << 32))[0]
    return number / 100.0 if value & 0x01 else number


def _read_font(payload: bytes, *, biff8: bool) -> XlsFontStyle:
    """提取 FONT 中可映射为 MinerU 行内标签的字符属性。"""

    if len(payload) < 11:
        return XlsFontStyle()
    flags = int(get_u16(payload, 2) or 0)
    weight = int(get_u16(payload, 6) or 400)
    escapement = int(get_u16(payload, 8) or 0)
    underline = int(payload[10])
    return XlsFontStyle(
        bold=weight >= 700,
        italic=bool(flags & 0x0002),
        strike=bool(flags & 0x0008),
        underline=underline != 0,
        superscript=escapement == 1,
        subscript=escapement == 2,
    )


def _read_boundsheet(payload: bytes, globals_: _Globals) -> _BoundSheet | None:
    """解析 sheet 偏移、可见性、类型和名称。"""

    if len(payload) < 8:
        return None
    offset = get_u32(payload, 0)
    if offset is None:
        return None
    reader = SegmentReader([payload[6:]])
    decoded = globals_.read_string(reader, short=True)
    if decoded is None:
        return None
    return _BoundSheet(
        name=clean_text(decoded.text) or "Sheet",
        offset=int(offset),
        visible=(payload[4] & 0x03) == 0,
        sheet_type=int(payload[5]),
    )


def _read_sst(segments: list[bytes]) -> list[DecodedString]:
    """读取共享字符串表，并允许损坏尾部保留已完成的 strings。"""

    reader = SegmentReader(segments)
    total = reader.u32()
    unique = reader.u32()
    if total is None or unique is None:
        return []
    strings: list[DecodedString] = []
    while len(strings) < unique:
        decoded = read_biff8_string(reader, short=False, rich=True)
        if decoded is None:
            logger.warning(
                "XLS_SST_TRUNCATED: shared string table stopped at entry {}",
                len(strings),
            )
            break
        strings.append(decoded)
    return strings


def _read_supbook(payload: bytes) -> bool:
    """判断 SUPBOOK 是否表示当前工作簿内部 sheet 集合。"""

    return len(payload) >= 4 and get_u16(payload, 2) == 0x0401


def _read_extern_sheets(
    payload: bytes,
    internal_supbooks: list[bool],
) -> list[int | None]:
    """把 XTI entries 解析为内部工作表索引。"""

    count = min(int(get_u16(payload, 0) or 0), max(0, (len(payload) - 2) // 6))
    result: list[int | None] = []
    for index in range(count):
        offset = 2 + index * 6
        supbook, first_sheet, last_sheet = struct.unpack_from("<3H", payload, offset)
        if (
            supbook < len(internal_supbooks)
            and internal_supbooks[supbook]
            and first_sheet == last_sheet
            and first_sheet < 0xFFFE
        ):
            result.append(int(first_sheet))
        else:
            result.append(None)
    return result


def _read_globals(data: bytes, budget: RecordBudget) -> _Globals:
    """解析 Workbook Globals Substream 及共享图片资源。"""

    first = record_at(data, 0, budget=budget)
    if first is None or first.record_type != BOF:
        raise LegacyOfficeMalformedError("workbook stream does not start with a BOF record")
    if get_u16(first.payload, 2) not in {WORKBOOK_GLOBALS_SUBSTREAM, None}:
        raise LegacyOfficeMalformedError("first BIFF substream is not workbook globals")
    version = get_u16(first.payload, 0)
    if version not in {0x0500, 0x0600}:
        raise LegacyOfficeMalformedError(f"unsupported BIFF version: {version!r}")
    globals_ = _Globals(biff8=version == 0x0600)
    raw_formats: dict[int, str] = {}
    raw_xfs: list[tuple[int, int]] = []
    drawing_chunks: list[bytes] = []
    internal_supbooks: list[bool] = []
    extern_payloads: list[bytes] = []
    cursor = first.next_offset
    depth = 1
    while cursor < len(data):
        record = record_at(data, cursor, budget=budget)
        if record is None:
            logger.warning("XLS_GLOBALS_TRUNCATED: globals end at byte {}", cursor)
            break
        cursor = record.next_offset
        if record.record_type == BOF:
            depth += 1
            continue
        if record.record_type == EOF:
            depth -= 1
            if depth == 0:
                break
            continue
        if depth != 1:
            continue
        if record.record_type == FILEPASS:
            raise LegacyOfficeEncryptedError("password-protected XLS is not supported")
        if record.record_type == CODEPAGE:
            globals_.encoding = codepage_name(int(get_u16(record.payload, 0) or 1252))
        elif record.record_type == DATEMODE:
            globals_.date1904 = get_u16(record.payload, 0) == 1
        elif record.record_type == FONT:
            globals_.fonts.append(_read_font(record.payload, biff8=globals_.biff8))
        elif record.record_type == BOUNDSHEET:
            sheet = _read_boundsheet(record.payload, globals_)
            if sheet is not None and sheet.sheet_type != 0x06:
                globals_.sheets.append(sheet)
        elif record.record_type == FORMAT:
            format_id = get_u16(record.payload, 0)
            if format_id is not None:
                reader = SegmentReader([record.payload[2:]])
                decoded = globals_.read_string(reader, short=not globals_.biff8)
                if decoded is not None:
                    raw_formats[int(format_id)] = decoded.text
        elif record.record_type == XF:
            font_index = int(get_u16(record.payload, 0) or 0)
            format_id = int(get_u16(record.payload, 2) or 0)
            raw_xfs.append((font_index, format_id))
        elif record.record_type == SST and globals_.biff8:
            segments, cursor = collect_continues(data, record, budget=budget)
            globals_.strings = _read_sst(segments)
        elif record.record_type == MSODRAWINGGROUP:
            segments, cursor = collect_continues(data, record, budget=budget)
            drawing_chunks.extend(segments)
        elif record.record_type == SUPBOOK:
            internal_supbooks.append(_read_supbook(record.payload))
        elif record.record_type == EXTERNSHEET:
            extern_payloads.append(record.payload)
        elif record.record_type == WINDOW1 and len(record.payload) >= 12:
            globals_.active_sheet_index = get_u16(record.payload, 10)

    globals_.formats = [
        _CellFormat(
            font_index=font_index,
            format_code=raw_formats.get(format_id) or builtin_number_format(format_id),
        )
        for font_index, format_id in raw_xfs
    ]
    for payload in extern_payloads:
        globals_.extern_sheets.extend(_read_extern_sheets(payload, internal_supbooks))
    if drawing_chunks:
        globals_.images = decode_bstore(b"".join(drawing_chunks), charge=budget.charge)
    return globals_


def _cell_ref(payload: bytes) -> tuple[int, int, int] | None:
    """读取 cell header，并拒绝超出 BIFF8 网格的列。"""

    if len(payload) < 6:
        return None
    row, col, xf_index = struct.unpack_from("<3H", payload, 0)
    if row >= MAX_ROWS or col >= MAX_COLS:
        return None
    return int(row), int(col), int(xf_index)


def _resolved_rich_text(
    decoded: DecodedString,
    globals_: _Globals,
    xf_index: int,
) -> XlsRichText:
    """应用 text number format，并在原文未变时保留 rich runs。"""

    cell_format = globals_.cell_format(xf_index)
    formatted = format_text(decoded.text, cell_format.format_code)
    if formatted != decoded.text:
        return XlsRichText(formatted)
    return to_rich_text(decoded, globals_.fonts)


def _put_cell(
    sheet: XlsSheet,
    row: int,
    col: int,
    value: XlsRichText,
) -> None:
    """仅保存非空文本，并覆盖同坐标较早的缓存记录。"""

    if not value.text:
        return
    sheet.cells[(row, col)] = XlsCell(row=row, col=col, value=value)


def _append_cell_text(
    sheet: XlsSheet,
    row: int,
    col: int,
    value: XlsRichText,
) -> None:
    """把 drawing/control 文本追加到 anchor 单元格且平移 rich runs。"""

    if not value.text:
        return
    existing = sheet.cells.get((row, col))
    if existing is None:
        _put_cell(sheet, row, col, value)
        return
    separator = "\n" if existing.value.text else ""
    shift = len(existing.value.text) + len(separator)
    shifted = tuple(
        XlsRichRun(
            start=run.start + shift,
            end=run.end + shift,
            style=run.style,
        )
        for run in value.runs
    )
    existing.value = XlsRichText(
        text=existing.value.text + separator + value.text,
        runs=existing.value.runs + shifted,
    )


def _read_label_string(
    segments: list[bytes],
    globals_: _Globals,
    *,
    rich_record: bool,
) -> DecodedString | None:
    """读取 LABEL/RSTRING 的字符串并恢复 RSTRING formatting runs。"""

    if not segments or len(segments[0]) < 6:
        return None
    adjusted = [segments[0][6:], *segments[1:]]
    reader = SegmentReader(adjusted)
    decoded = globals_.read_string(reader, short=False)
    if decoded is None or not rich_record:
        return decoded
    run_count = reader.u16()
    if run_count is None:
        return decoded
    starts: list[tuple[int, int]] = []
    for _ in range(run_count):
        raw = reader.read_across(4)
        if raw is None:
            break
        character_index, font_index = struct.unpack("<HH", raw)
        starts.append((int(character_index), int(font_index)))
    return DecodedString(decoded.text, tuple(starts))


def _pict_embedding_storage(payload: bytes, picture_flags: int | None) -> str | None:
    """从 FtPictFmla 读取嵌入对象的 MBD storage 名称。"""

    if picture_flags is None:
        return None
    # DDE、ActiveX、controls stream 与 camera picture 都不是内嵌公式 OLE 对象。
    if picture_flags & (0x0002 | 0x0010 | 0x0020 | 0x0080):
        return None
    if len(payload) < 10:
        return None
    cb_fmla = int(get_u16(payload, 0) or 0)
    formula_end = 2 + cb_fmla
    if cb_fmla <= 0 or cb_fmla % 2 or formula_end + 4 > len(payload):
        return None
    formula = payload[2:formula_end]
    if len(formula) < 7 or int(get_u16(formula, 0) or 0) & 0x7FFF != 5:
        return None
    # ObjectParsedFormula 的四字节 unused 在部分生产器中省略，因此兼容两个合法落点。
    if not any(offset + 5 <= len(formula) and formula[offset] == 0x02 for offset in (6, 2)):
        return None
    location = get_u32(payload, formula_end)
    return f"MBD{int(location):08X}" if location is not None else None


def _read_obj(payload: bytes) -> _SheetObject | None:
    """解析 OBJ subrecords 中的对象类型、id、状态与嵌入 storage。"""

    cursor = 0
    object_type: int | None = None
    object_id = 0
    checked: bool | None = None
    picture_flags: int | None = None
    picture_formula: bytes | None = None
    while cursor + 4 <= len(payload):
        sub_type, length = struct.unpack_from("<HH", payload, cursor)
        data_start = cursor + 4
        data_end = data_start + int(length)
        if data_end > len(payload):
            break
        body = payload[data_start:data_end]
        if sub_type == 0x0015 and len(body) >= 4:
            object_type, object_id = struct.unpack_from("<HH", body, 0)
        elif sub_type == 0x0012 and len(body) >= 2:
            state = int(get_u16(body, 0) or 0)
            checked = state == 1 if state in {0, 1} else None
        elif sub_type == 0x0008 and len(body) >= 2:
            picture_flags = int(get_u16(body, 0) or 0)
        elif sub_type == 0x0009:
            picture_formula = body
        if sub_type == 0:
            break
        cursor = data_end
    if object_type is None:
        return None
    return _SheetObject(
        int(object_type),
        int(object_id),
        checked=checked,
        embedding_storage=(_pict_embedding_storage(picture_formula, picture_flags) if picture_formula is not None else None),
    )


def _sanitize_hyperlink_target(target: str) -> str | None:
    """保留 XlsxModel 允许的网络、邮件、内部与相对链接。"""

    candidate = target.strip().replace("\x00", "")
    if not candidate:
        return None
    lowered = candidate.casefold()
    if lowered.startswith(("javascript:", "data:", "vbscript:", "file:")):
        return None
    parsed = urlparse(candidate)
    if parsed.scheme and parsed.scheme.casefold() not in {"http", "https", "mailto", "ftp"}:
        return None
    if parsed.scheme.casefold() == "mailto" and not parsed.path:
        return None
    return candidate


def _read_hyperlink_unicode(payload: bytes, cursor: int) -> tuple[str | None, int]:
    """读取 Hyperlink Object 中含末尾 NUL 的 UTF-16 字符串。"""

    if cursor + 4 > len(payload):
        return None, len(payload)
    character_count = int(struct.unpack_from("<I", payload, cursor)[0])
    cursor += 4
    byte_count = character_count * 2
    if byte_count < 0 or cursor + byte_count > len(payload):
        return None, len(payload)
    text = payload[cursor : cursor + byte_count].decode("utf-16le", "replace").rstrip("\x00")
    return clean_text(text), cursor + byte_count


def _read_url_moniker(payload: bytes, cursor: int) -> tuple[str | None, int]:
    """读取 URL Moniker 的 UTF-16 URL，忽略可选尾部元数据。"""

    if cursor + 4 > len(payload):
        return None, len(payload)
    byte_count = int(struct.unpack_from("<I", payload, cursor)[0])
    cursor += 4
    if byte_count < 0 or cursor + byte_count > len(payload):
        return None, len(payload)
    raw = payload[cursor : cursor + byte_count]
    usable = raw[: len(raw) - (len(raw) % 2)]
    text = usable.decode("utf-16le", "replace").split("\x00", 1)[0]
    return clean_text(text), cursor + byte_count


def _read_file_moniker(payload: bytes, cursor: int) -> tuple[str | None, int]:
    """尽力读取 File Moniker 的 ANSI 或 Unicode 路径。"""

    if cursor + 6 > len(payload):
        return None, len(payload)
    anti_count = int(struct.unpack_from("<H", payload, cursor)[0])
    ansi_length = int(struct.unpack_from("<I", payload, cursor + 2)[0])
    cursor += 6
    if cursor + ansi_length > len(payload):
        return None, len(payload)
    ansi = payload[cursor : cursor + ansi_length].split(b"\x00", 1)[0]
    cursor += ansi_length
    path = ("../" * anti_count) + ansi.decode("cp1252", "replace")
    return clean_text(path), cursor


def _read_hyperlink_target(payload: bytes) -> str | None:
    """解析 HLink 中的 Hyperlink Object 并返回经过白名单过滤的目标。"""

    if len(payload) < 32:
        return None
    cursor = 24
    version, flags = struct.unpack_from("<II", payload, cursor)
    cursor += 8
    if version != 2:
        return None
    if flags & 0x10:
        _, cursor = _read_hyperlink_unicode(payload, cursor)
    if flags & 0x80:
        _, cursor = _read_hyperlink_unicode(payload, cursor)
    target: str | None = None
    blocked_local_file = False
    if flags & 0x01:
        if flags & 0x0100:
            target, cursor = _read_hyperlink_unicode(payload, cursor)
        elif cursor + 16 <= len(payload):
            moniker = uuid.UUID(bytes_le=payload[cursor : cursor + 16])
            cursor += 16
            if moniker == uuid.UUID("79eac9e0-baf9-11ce-8c82-00aa004ba90b"):
                target, cursor = _read_url_moniker(payload, cursor)
            elif moniker == uuid.UUID("00000303-0000-0000-c000-000000000046"):
                _, cursor = _read_file_moniker(payload, cursor)
                blocked_local_file = True
    location: str | None = None
    if flags & 0x08:
        location, cursor = _read_hyperlink_unicode(payload, cursor)
    if location:
        target = f"{target}#{location}" if target else f"#{location}"
    if blocked_local_file:
        logger.warning("XLS_HYPERLINK_BLOCKED: local File Moniker")
        return None
    sanitized = _sanitize_hyperlink_target(target or "")
    if target and sanitized is None:
        logger.warning("XLS_HYPERLINK_BLOCKED: target={!r}", target)
    return sanitized


def _apply_hlink(
    sheet: XlsSheet,
    payload: bytes,
    pending: dict[tuple[int, int], str],
) -> None:
    """把 HLink 范围目标暂存到所有覆盖单元格。"""

    if len(payload) < 8:
        return
    row_first, row_last, col_first, col_last = struct.unpack_from("<4H", payload, 0)
    target = _read_hyperlink_target(payload)
    if target is None:
        return
    for row in range(min(row_first, row_last), min(max(row_first, row_last), MAX_ROWS - 1) + 1):
        for col in range(min(col_first, col_last), min(max(col_first, col_last), MAX_COLS - 1) + 1):
            pending[(int(row), int(col))] = target


def _shape_anchor(shape: OfficeArtShape | None) -> tuple[int, int] | None:
    """返回 shape 左上角 cell anchor。"""

    if shape is None or shape.anchor is None:
        return None
    return shape.anchor[0], shape.anchor[1]


def _serialize_payload(payload: OfficeImagePayload) -> str | None:
    """使用共享 Office 图片策略序列化 BLIP。"""

    return serialize_office_image(
        payload.data,
        part_name=f"picture.{payload.extension}",
        content_type=payload.content_type,
    )


def _bind_objects(
    sheet: XlsSheet,
    objects: list[_SheetObject],
    drawing_data: bytes,
    chart_streams: list[list[BiffRecord]],
    *,
    globals_: _Globals,
    sheet_index: int,
    native_equations: dict[str, str],
    image_equation_decoder: OfficeImageEquationDecoder,
    budget: RecordBudget,
) -> None:
    """按 drawing/OBJ 顺序绑定文本框、复选框、图片与嵌入图表。"""

    shapes = extract_excel_shapes(drawing_data, charge=budget.charge) if drawing_data else []
    if len(shapes) != len(objects):
        logger.warning(
            "XLS_DRAWING_OBJECT_MISMATCH: sheet={!r}, shapes={}, objects={}",
            sheet.name,
            len(shapes),
            len(objects),
        )
    for object_, shape in zip(objects, shapes, strict=False):
        object_.shape = shape
    chart_objects = [object_ for object_ in objects if object_.object_type == OBJ_CHART]
    for object_ in objects:
        shape = object_.shape
        if shape is None or shape.hidden:
            continue
        anchor = _shape_anchor(shape)
        if anchor is None:
            continue
        row, col = anchor
        equation = (
            native_equations.get(object_.embedding_storage.casefold())
            if object_.object_type == OBJ_PICTURE and object_.embedding_storage
            else None
        )
        if equation:
            sheet.equations.append(XlsEquation(row=row, col=col, latex=equation))
            continue
        if (
            object_.object_type == OBJ_PICTURE
            and shape.pib is not None
            and (payload := globals_.images.get(int(shape.pib))) is not None
        ):
            image_latex = image_equation_decoder.decode(
                payload.data,
                part_name=f"picture.{payload.extension}",
                content_type=payload.content_type,
            )
            if image_latex:
                sheet.equations.append(XlsEquation(row=row, col=col, latex=image_latex))
                continue
        if object_.object_type == OBJ_TEXTBOX and object_.text is not None:
            _append_cell_text(sheet, row, col, object_.text)
        elif object_.object_type == OBJ_CHECKBOX and object_.checked is not None:
            marker = "[x]" if object_.checked else "[ ]"
            caption = object_.text.text.strip() if object_.text is not None else ""
            _append_cell_text(sheet, row, col, XlsRichText(f"{marker} {caption}".rstrip()))
        if object_.object_type in {OBJ_PICTURE, OBJ_CHART} or shape.pib is not None:
            payload = globals_.images.get(int(shape.pib or 0))
            if payload is not None:
                image_base64 = _serialize_payload(payload)
                if image_base64 and object_.object_type != OBJ_CHART:
                    sheet.images.append(XlsImage(row=row, col=col, image_base64=image_base64))

    for index, object_ in enumerate(chart_objects):
        if object_.shape is None or object_.shape.hidden:
            continue
        anchor = _shape_anchor(object_.shape)
        if anchor is None:
            continue
        axes = (
            chart_source_axes(
                chart_streams[index],
                current_sheet_index=sheet_index,
                extern_sheets=globals_.extern_sheets,
            )
            if index < len(chart_streams)
            else None
        )
        preview: str | None = None
        if object_.shape.pib is not None:
            payload = globals_.images.get(object_.shape.pib)
            preview = _serialize_payload(payload) if payload is not None else None
        if axes is None:
            if preview:
                sheet.charts.append(
                    XlsChart(
                        row=anchor[0],
                        col=anchor[1],
                        source_rows=(),
                        source_cols=(),
                        image_base64=preview,
                    )
                )
            else:
                logger.warning(
                    "XLS_CHART_SOURCE_UNSUPPORTED: sheet={!r}, object_id={}",
                    sheet.name,
                    object_.object_id,
                )
            continue
        rows, cols = axes
        sheet.charts.append(
            XlsChart(
                row=anchor[0],
                col=anchor[1],
                source_rows=tuple(rows),
                source_cols=tuple(cols),
                image_base64=preview,
            )
        )


def _read_sheet(
    data: bytes,
    globals_: _Globals,
    descriptor: _BoundSheet,
    offset: int,
    *,
    sheet_index: int,
    recovered: bool,
    native_equations: dict[str, str],
    image_equation_decoder: OfficeImageEquationDecoder,
    budget: RecordBudget,
) -> XlsSheet | None:
    """解析一个 worksheet substream 并绑定其 drawing/chart 对象。"""

    first = record_at(data, offset, budget=budget)
    if first is None or first.record_type != BOF or get_u16(first.payload, 2) != WORKSHEET_SUBSTREAM:
        return None
    sheet = XlsSheet(
        name=descriptor.name,
        visible=descriptor.visible,
        order=sheet_index,
        recovered=recovered,
    )
    cursor = first.next_offset
    depth = 1
    active_chart: list[BiffRecord] | None = None
    chart_streams: list[list[BiffRecord]] = []
    drawing_chunks: list[bytes] = []
    objects: list[_SheetObject] = []
    pending_formula: tuple[int, int, int] | None = None
    pending_links: dict[tuple[int, int], str] = {}

    while cursor < len(data):
        record = record_at(data, cursor, budget=budget)
        if record is None:
            logger.warning(
                "XLS_SHEET_TRUNCATED: sheet={!r}, byte={}",
                sheet.name,
                cursor,
            )
            break
        cursor = record.next_offset
        if active_chart is not None:
            active_chart.append(record)
            if record.record_type == BOF:
                depth += 1
            elif record.record_type == EOF:
                depth -= 1
                if depth == 1:
                    chart_streams.append(active_chart)
                    active_chart = None
            continue
        if record.record_type == BOF:
            depth += 1
            if depth == 2 and get_u16(record.payload, 2) == CHART_SUBSTREAM:
                active_chart = [record]
            continue
        if record.record_type == EOF:
            depth -= 1
            if depth == 0:
                break
            continue
        if depth != 1:
            continue

        if record.record_type == MERGEDCELLS:
            count = min(int(get_u16(record.payload, 0) or 0), max(0, (len(record.payload) - 2) // 8))
            for index in range(count):
                row_first, row_last, col_first, col_last = struct.unpack_from("<4H", record.payload, 2 + index * 8)
                row_start, row_end = sorted((int(row_first), int(row_last)))
                col_start, col_end = sorted((int(col_first), int(col_last)))
                if col_start >= MAX_COLS or (row_start == row_end and col_start == col_end):
                    continue
                sheet.merges.append(
                    (
                        row_start,
                        col_start,
                        min(row_end, MAX_ROWS - 1),
                        min(col_end, MAX_COLS - 1),
                    )
                )
        elif record.record_type == LABELSST:
            reference = _cell_ref(record.payload)
            string_index = get_u32(record.payload, 6)
            if reference is not None and string_index is not None and string_index < len(globals_.strings):
                row, col, xf_index = reference
                _put_cell(
                    sheet,
                    row,
                    col,
                    _resolved_rich_text(globals_.strings[int(string_index)], globals_, xf_index),
                )
        elif record.record_type in {LABEL, RSTRING}:
            segments, cursor = collect_continues(data, record, budget=budget)
            reference = _cell_ref(record.payload)
            decoded = _read_label_string(
                segments,
                globals_,
                rich_record=record.record_type == RSTRING,
            )
            if reference is not None and decoded is not None:
                row, col, xf_index = reference
                _put_cell(sheet, row, col, _resolved_rich_text(decoded, globals_, xf_index))
        elif record.record_type == NUMBER:
            reference = _cell_ref(record.payload)
            value = get_f64(record.payload, 6)
            if reference is not None and value is not None:
                row, col, xf_index = reference
                cell_format = globals_.cell_format(xf_index)
                _put_cell(
                    sheet,
                    row,
                    col,
                    XlsRichText(
                        format_number(
                            value,
                            cell_format.format_code,
                            date1904=globals_.date1904,
                        )
                    ),
                )
        elif record.record_type == RK:
            reference = _cell_ref(record.payload)
            raw_value = get_u32(record.payload, 6)
            if reference is not None and raw_value is not None:
                row, col, xf_index = reference
                cell_format = globals_.cell_format(xf_index)
                _put_cell(
                    sheet,
                    row,
                    col,
                    XlsRichText(
                        format_number(
                            _rk_number(raw_value),
                            cell_format.format_code,
                            date1904=globals_.date1904,
                        )
                    ),
                )
        elif record.record_type == MULRK and len(record.payload) >= 6:
            row = int(get_u16(record.payload, 0) or 0)
            first_col = int(get_u16(record.payload, 2) or 0)
            pair_count = max(0, (len(record.payload) - 6) // 6)
            for index in range(pair_count):
                xf_index = int(get_u16(record.payload, 4 + index * 6) or 0)
                raw_value = get_u32(record.payload, 6 + index * 6)
                col = first_col + index
                if raw_value is None or col >= MAX_COLS:
                    break
                cell_format = globals_.cell_format(xf_index)
                _put_cell(
                    sheet,
                    row,
                    col,
                    XlsRichText(
                        format_number(
                            _rk_number(raw_value),
                            cell_format.format_code,
                            date1904=globals_.date1904,
                        )
                    ),
                )
        elif record.record_type == BOOLERR:
            reference = _cell_ref(record.payload)
            if reference is not None and len(record.payload) >= 8:
                row, col, _ = reference
                value, is_error = record.payload[6], record.payload[7]
                text = _error_literal(value) if is_error == 1 else ("TRUE" if value else "FALSE")
                if text:
                    _put_cell(sheet, row, col, XlsRichText(text))
        elif record.record_type == FORMULA:
            reference = _cell_ref(record.payload)
            if reference is not None and len(record.payload) >= 14:
                row, col, xf_index = reference
                cached = record.payload[6:14]
                if cached[6:8] == b"\xff\xff":
                    kind = cached[0]
                    if kind == 0x00:
                        pending_formula = (row, col, xf_index)
                    elif kind == 0x01:
                        _put_cell(sheet, row, col, XlsRichText("TRUE" if cached[2] else "FALSE"))
                    elif kind == 0x02:
                        error = _error_literal(cached[2])
                        if error:
                            _put_cell(sheet, row, col, XlsRichText(error))
                else:
                    value = get_f64(record.payload, 6)
                    if value is not None:
                        cell_format = globals_.cell_format(xf_index)
                        _put_cell(
                            sheet,
                            row,
                            col,
                            XlsRichText(
                                format_number(
                                    value,
                                    cell_format.format_code,
                                    date1904=globals_.date1904,
                                )
                            ),
                        )
        elif record.record_type == STRING:
            segments, cursor = collect_continues(data, record, budget=budget)
            if pending_formula is not None:
                reader = SegmentReader(segments)
                decoded = globals_.read_string(reader, short=False)
                if decoded is not None:
                    if decoded.text.lstrip().upper().startswith(("=DISPIMG(", "=_XLFN.DISPIMG(")):
                        # 旧版文件无法携带现代 DISPIMG 计算语义，按 Office 回存结果稳定降级。
                        decoded = DecodedString("#NAME?")
                    row, col, xf_index = pending_formula
                    _put_cell(sheet, row, col, _resolved_rich_text(decoded, globals_, xf_index))
                pending_formula = None
        elif record.record_type == MSODRAWING:
            segments, cursor = collect_continues(data, record, budget=budget)
            drawing_chunks.extend(segments)
        elif record.record_type == OBJ:
            object_ = _read_obj(record.payload)
            if object_ is not None:
                objects.append(object_)
        elif record.record_type == TXO:
            segments, cursor = collect_continues(data, record, budget=budget)
            if objects:
                objects[-1].text = read_txo_text(
                    record.payload,
                    segments[1:],
                    globals_.fonts,
                )
        elif record.record_type == HLINK:
            _apply_hlink(sheet, record.payload, pending_links)
        elif record.record_type in {ROW, COLINFO, CONTINUE}:
            # 用户明确要求隐藏行列中的内容仍参与表格重建。
            pass

    for coordinate, target in pending_links.items():
        cell = sheet.cells.get(coordinate)
        if cell is not None:
            cell.hyperlink = target
    _bind_objects(
        sheet,
        objects,
        b"".join(drawing_chunks),
        chart_streams,
        globals_=globals_,
        sheet_index=sheet_index,
        native_equations=native_equations,
        image_equation_decoder=image_equation_decoder,
        budget=budget,
    )
    return sheet


def _worksheet_bof_offsets(data: bytes) -> list[int]:
    """扫描所有可识别 worksheet BOF 偏移，供坏目录恢复使用。"""

    return [
        record.offset
        for record in iter_records(data)
        if record.record_type == BOF and get_u16(record.payload, 2) == WORKSHEET_SUBSTREAM
    ]


def _is_worksheet_offset(data: bytes, offset: int) -> bool:
    """判断 BoundSheet offset 是否精确指向 worksheet BOF。"""

    record = record_at(data, offset)
    return bool(record is not None and record.record_type == BOF and get_u16(record.payload, 2) == WORKSHEET_SUBSTREAM)


def _parse_chart_sheets(
    data: bytes,
    globals_: _Globals,
    *,
    budget: RecordBudget,
) -> list[XlsChartSheet]:
    """解析独立 chart sheet，并把 BRAI 引用绑定到唯一 worksheet。"""

    worksheet_names = {
        index: descriptor.name for index, descriptor in enumerate(globals_.sheets) if descriptor.sheet_type == 0x00
    }
    chart_sheets: list[XlsChartSheet] = []
    for order, descriptor in enumerate(globals_.sheets):
        if descriptor.sheet_type != 0x02:
            continue
        first = record_at(data, descriptor.offset, budget=budget)
        selection = None
        if first is not None and first.record_type == BOF and get_u16(first.payload, 2) == CHART_SUBSTREAM:
            records = list(
                iter_records(
                    data,
                    start=descriptor.offset,
                    stop_at_eof=True,
                    budget=budget,
                )
            )
            selection = chart_source_selection(
                records,
                current_sheet_index=order,
                extern_sheets=globals_.extern_sheets,
            )
        source_name = worksheet_names.get(selection.sheet_index) if selection is not None else None
        chart_sheets.append(
            XlsChartSheet(
                name=descriptor.name,
                visible=descriptor.visible,
                order=order,
                source_sheet_name=source_name,
                source_rows=(selection.rows if selection is not None and source_name is not None else ()),
                source_cols=(selection.cols if selection is not None and source_name is not None else ()),
            )
        )
    return chart_sheets


def parse_xls_workbook(
    data: bytes,
    *,
    native_equations: dict[str, str] | None = None,
) -> XlsWorkbook:
    """解析 Workbook/Book stream，并按目录顺序恢复 worksheets 与原生公式。"""

    if not data:
        raise LegacyOfficeMalformedError("empty Workbook stream")
    budget = RecordBudget()
    normalized_equations = {storage.casefold(): latex for storage, latex in (native_equations or {}).items()}
    image_equation_decoder = OfficeImageEquationDecoder()
    globals_ = _read_globals(data, budget)
    candidates = _worksheet_bof_offsets(data)
    used_offsets: set[int] = set()
    sheets: list[XlsSheet] = []

    descriptor_entries = [(index, sheet) for index, sheet in enumerate(globals_.sheets) if sheet.sheet_type == 0x00]
    if not descriptor_entries and candidates:
        descriptor_entries = [
            (
                index - 1,
                _BoundSheet(
                    name=f"Recovered Sheet {index}",
                    offset=offset,
                    visible=True,
                    sheet_type=0,
                ),
            )
            for index, offset in enumerate(candidates, start=1)
        ]
    if not descriptor_entries and not candidates:
        raise LegacyOfficeMalformedError("workbook contains no worksheet substream")

    for sheet_index, descriptor in descriptor_entries:
        resolved_offset: int | None = None
        recovered = False
        if _is_worksheet_offset(data, descriptor.offset) and descriptor.offset not in used_offsets:
            resolved_offset = descriptor.offset
        else:
            resolved_offset = next((offset for offset in candidates if offset not in used_offsets), None)
            recovered = resolved_offset is not None
            if recovered:
                logger.warning(
                    "XLS_BOUNDSHEET_RECOVERED: sheet={!r}, old_offset={}, new_offset={}",
                    descriptor.name,
                    descriptor.offset,
                    resolved_offset,
                )
        if resolved_offset is None:
            logger.warning("XLS_SHEET_UNREADABLE: keeping empty sheet {!r}", descriptor.name)
            sheets.append(
                XlsSheet(
                    name=descriptor.name,
                    visible=descriptor.visible,
                    order=sheet_index,
                    recovered=True,
                )
            )
            continue
        used_offsets.add(resolved_offset)
        parsed = _read_sheet(
            data,
            globals_,
            descriptor,
            resolved_offset,
            sheet_index=sheet_index,
            recovered=recovered,
            native_equations=normalized_equations,
            image_equation_decoder=image_equation_decoder,
            budget=budget,
        )
        if parsed is None:
            sheets.append(
                XlsSheet(
                    name=descriptor.name,
                    visible=descriptor.visible,
                    order=sheet_index,
                    recovered=True,
                )
            )
        else:
            sheets.append(parsed)
    return XlsWorkbook(
        sheets=sheets,
        chart_sheets=_parse_chart_sheets(data, globals_, budget=budget),
        active_sheet_index=globals_.active_sheet_index,
    )
