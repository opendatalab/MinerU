# Copyright (c) Opendatalab. All rights reserved.

"""PowerPoint 97–2003 二进制文档的分页语义解析器。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct
from typing import Iterable
import unicodedata
from urllib.parse import urlparse
import zlib

from loguru import logger

from ..legacy import (
    BoundedOleReader,
    LegacyOfficeEncryptedError,
    LegacyOfficeMalformedError,
    LegacyOfficeResourceLimitError,
)
from ..legacy.limits import (
    MAX_ASSET_TOTAL_BYTES,
    MAX_ENTRY_BYTES,
    MAX_GRID_SLOTS,
    MAX_PICTURE_RECORDS,
    MAX_USER_EDIT_CHAIN,
)
from ..legacy.mtef import decode_equation_object
from ..legacy.officeart import (
    OfficeArtRecord,
    OfficeImagePayload,
    decode_blip as decode_officeart_blip,
)
from ..image import serialize_office_image
from ..image_equation import (
    OfficeImageEquationDecoder,
)
from ..xls.embedded_chart import extract_embedded_chart_html_from_storage

from .models import (
    PptChartElement,
    PptEquationElement,
    PptImageElement,
    PptParagraph,
    PptPresentation,
    PptSlide,
    PptTableCell,
    PptTableElement,
    PptTextElement,
    PptTextRun,
)
from .records import (
    CONTAINER_VERSION,
    PptRecord,
    RecordBudget,
    get_i16,
    get_u16,
    get_u32,
    iter_descendants,
    iter_records,
    record_at,
    utf16_text,
)
from .style_text import CharacterRun, MasterLevel, ParagraphRun, StyleRuns, parse_master_style, parse_style_text

# MS-PPT records.
RT_DOCUMENT = 0x03E8
RT_DOCUMENT_ATOM = 0x03E9
RT_SLIDE = 0x03EE
RT_SLIDE_ATOM = 0x03EF
RT_NOTES = 0x03F0
RT_NOTES_ATOM = 0x03F1
RT_SLIDE_PERSIST_ATOM = 0x03F3
RT_MAIN_MASTER = 0x03F8
RT_SLIDE_SHOW_SLIDE_INFO_ATOM = 0x03F9
RT_TEXT_HEADER_ATOM = 0x0F9F
RT_TEXT_CHARS_ATOM = 0x0FA0
RT_STYLE_TEXT_PROP_ATOM = 0x0FA1
RT_TEXT_MASTER_STYLE_ATOM = 0x0FA3
RT_TEXT_BYTES_ATOM = 0x0FA8
RT_CSTRING = 0x0FBA
RT_TEXT_INTERACTIVE_INFO_ATOM = 0x0FDF
RT_EXTERNAL_HYPERLINK = 0x0FD7
RT_EXTERNAL_HYPERLINK_ATOM = 0x0FD3
RT_SLIDE_LIST_WITH_TEXT = 0x0FF0
RT_INTERACTIVE_INFO = 0x0FF2
RT_INTERACTIVE_INFO_ATOM = 0x0FF3
RT_USER_EDIT_ATOM = 0x0FF5
RT_OUTLINE_TEXT_REF_ATOM = 0x0F9E
RT_PERSIST_DIRECTORY_ATOM = 0x1772
RT_CRYPT_SESSION10_CONTAINER = 0x2F14
RT_EXTERNAL_OBJECT_REF_ATOM = 0x0BC1
RT_EXTERNAL_OLE_OBJECT_ATOM = 0x0FC3
RT_EXTERNAL_OLE_OBJECT_STG = 0x1011

# OfficeArt records and properties.
RT_OFFICEART_DGG_CONTAINER = 0xF000
RT_OFFICEART_BSTORE_CONTAINER = 0xF001
RT_OFFICEART_SPGR_CONTAINER = 0xF003
RT_OFFICEART_SP_CONTAINER = 0xF004
RT_OFFICEART_BSE = 0xF007
RT_OFFICEART_FSPGR = 0xF009
RT_OFFICEART_FSP = 0xF00A
RT_OFFICEART_FOPT = 0xF00B
RT_OFFICEART_CLIENT_TEXTBOX = 0xF00D
RT_OFFICEART_CHILD_ANCHOR = 0xF00F
RT_OFFICEART_CLIENT_ANCHOR = 0xF010
RT_OFFICEART_CLIENT_DATA = 0xF011
RT_OFFICEART_TERTIARY_FOPT = 0xF122
RT_OE_PLACEHOLDER_ATOM = 0x0BC3

FOPT_PIB = 0x0104
FOPT_TABLE_PROPERTIES = 0x039F
FOPT_TABLE_ROW_PROPERTIES = 0x03A0

DEFAULT_SLIDE_WIDTH = 5760
DEFAULT_SLIDE_HEIGHT = 4320
_ALLOWED_LINK_SCHEMES = frozenset({"http", "https", "mailto"})
_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZLIB_SYNC_FLUSH_SUFFIX = b"\x00\x00\xff\xff"
_OLE_SUBTYPE_GRAPH = 0x0000_0004
_OLE_SUBTYPE_EQUATION = 0x0000_0006
_OLE_SUBTYPE_EXCEL_CHART = 0x0000_000E


@dataclass(frozen=True, slots=True)
class _TextContent:
    """一个 TextHeaderAtom 对应的完整段落集合。"""

    paragraphs: tuple[PptParagraph, ...]
    text_type: int


@dataclass(frozen=True, slots=True)
class _PersistLayout:
    """最新 UserEdit 解析出的文档与 persist 映射。"""

    document: PptRecord
    persist: dict[int, int]
    recovered: bool = False


@dataclass(frozen=True, slots=True)
class _EmbeddedOleObject:
    """一个已按 exObjId 绑定并解压的 PPT OLE 对象。"""

    subtype: int
    storage: bytes


@dataclass(frozen=True, slots=True)
class _GroupSpace:
    """嵌套 OfficeArt group 到幻灯片坐标的线性映射。"""

    coord_left: int
    coord_top: int
    coord_right: int
    coord_bottom: int
    abs_left: float
    abs_top: float
    abs_right: float
    abs_bottom: float


@dataclass(frozen=True, slots=True)
class _ShapeInfo:
    """带唯一遍历路径、坐标和 z-order 的 OfficeArt shape。"""

    key: tuple[int, ...]
    record: PptRecord
    space: _GroupSpace | None
    bbox: tuple[float, float, float, float]
    order: int
    group_key: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _TableGroup:
    """一个由 fIsTable 标记的 group 及其所有叶子 shape。"""

    key: tuple[int, ...]
    group_shape: PptRecord
    shapes: tuple[_ShapeInfo, ...]
    order: int
    authoritative: bool


_ImagePayload = OfficeImagePayload


@dataclass(frozen=True, slots=True)
class _NumberingStyle:
    """一个 pp9rt 槽位的自动编号覆盖。"""

    enabled: bool | None
    start: int | None


@dataclass(frozen=True, slots=True)
class _ShapeCollection:
    """一次遍历得到的叶子 shape 与表格 group。"""

    shapes: tuple[_ShapeInfo, ...]
    table_groups: tuple[_TableGroup, ...]


def _direct_children(record: PptRecord, budget: RecordBudget) -> list[PptRecord]:
    """返回容器的直接子记录，普通 atom 返回空列表。"""

    if record.version != CONTAINER_VERSION:
        return []
    return list(iter_records(record.payload, budget=budget))


def _find_latest_user_edit_offset(data: bytes, current_user: bytes) -> int | None:
    """从 Current User stream 读取最新 UserEditAtom 偏移。"""

    offset = get_u32(current_user, 16)
    return int(offset) if offset else None


def _merge_persist_directory(
    mapping: dict[int, int],
    payload: bytes,
) -> None:
    """合并一个 PersistDirectoryAtom；调用顺序保证较新记录优先。"""

    cursor = 0
    while cursor < len(payload):
        head = get_u32(payload, cursor)
        if head is None:
            raise LegacyOfficeMalformedError("PowerPoint persist directory is truncated")
        cursor += 4
        first_id = head & 0x000F_FFFF
        count = head >> 20
        if count <= 0 or cursor + count * 4 > len(payload):
            raise LegacyOfficeMalformedError("PowerPoint persist directory entry is invalid")
        for index in range(count):
            offset = get_u32(payload, cursor)
            if offset is None:
                raise LegacyOfficeMalformedError("PowerPoint persist offset is truncated")
            mapping.setdefault(first_id + index, int(offset))
            cursor += 4


def _latest_document_fallback(data: bytes, budget: RecordBudget) -> PptRecord | None:
    """在 persist 链损坏时选择最后一个完整 DocumentContainer。"""

    candidates = [
        record
        for record in iter_records(data, budget=budget)
        if record.record_type == RT_DOCUMENT and record.version == CONTAINER_VERSION
    ]
    return candidates[-1] if candidates else None


def _locate_document(
    data: bytes,
    current_user: bytes,
    budget: RecordBudget,
) -> _PersistLayout:
    """解析最新 UserEdit 链，失败时回退到顶层 DocumentContainer。"""

    edit_offset = _find_latest_user_edit_offset(data, current_user)
    persist: dict[int, int] = {}
    document_persist_id: int | None = None
    seen: set[int] = set()
    try:
        for _ in range(MAX_USER_EDIT_CHAIN):
            if not edit_offset:
                break
            if edit_offset in seen:
                raise LegacyOfficeMalformedError("PowerPoint UserEdit chain is cyclic")
            seen.add(edit_offset)
            edit = record_at(data, edit_offset, strict=True, budget=budget)
            if edit is None or edit.record_type != RT_USER_EDIT_ATOM:
                raise LegacyOfficeMalformedError("PowerPoint UserEditAtom is invalid")
            if document_persist_id is None:
                document_persist_id = get_u32(edit.payload, 16)
            directory_offset = get_u32(edit.payload, 12)
            if directory_offset is None:
                raise LegacyOfficeMalformedError("PowerPoint persist directory offset is missing")
            directory = record_at(data, directory_offset, strict=True, budget=budget)
            if directory is None or directory.record_type != RT_PERSIST_DIRECTORY_ATOM:
                raise LegacyOfficeMalformedError("PowerPoint PersistDirectoryAtom is invalid")
            _merge_persist_directory(persist, directory.payload)
            previous = get_u32(edit.payload, 8)
            if previous == edit_offset:
                raise LegacyOfficeMalformedError("PowerPoint UserEdit chain points to itself")
            edit_offset = int(previous or 0)
        else:
            raise LegacyOfficeResourceLimitError(f"UserEdit chain exceeds max_user_edit_chain={MAX_USER_EDIT_CHAIN}")
        if document_persist_id is not None:
            document_offset = persist.get(int(document_persist_id))
            if document_offset is not None:
                document = record_at(data, document_offset, strict=True, budget=budget)
                if document is not None and document.record_type == RT_DOCUMENT:
                    return _PersistLayout(document=document, persist=persist)
    except (LegacyOfficeMalformedError, LegacyOfficeResourceLimitError) as exc:
        if isinstance(exc, LegacyOfficeResourceLimitError):
            raise
        logger.warning(f"PPT_PERSIST_RECOVERY: {exc}")

    document = _latest_document_fallback(data, budget)
    if document is None:
        raise LegacyOfficeMalformedError("PowerPoint DocumentContainer is missing")
    # 顶层 persist 记录仍可帮助恢复同一保存版本中的 slide。
    recovered_persist: dict[int, int] = {}
    for record in iter_records(data, budget=budget):
        if record.record_type == RT_PERSIST_DIRECTORY_ATOM:
            try:
                _merge_persist_directory(recovered_persist, record.payload)
            except LegacyOfficeMalformedError:
                continue
    return _PersistLayout(document=document, persist=recovered_persist, recovered=True)


def _slide_entries(document: PptRecord, budget: RecordBudget) -> list[tuple[int, int]]:
    """按 SlideListWithText 的保存顺序返回 persist 引用与稳定 slide id。"""

    entries: list[tuple[int, int]] = []
    for container in iter_descendants(document, budget=budget):
        if container.record_type != RT_SLIDE_LIST_WITH_TEXT or container.instance != 0:
            continue
        for record in iter_records(container.payload, budget=budget):
            if record.record_type != RT_SLIDE_PERSIST_ATOM:
                continue
            reference = get_u32(record.payload, 0)
            slide_id = get_u32(record.payload, 12)
            if reference:
                entries.append((int(reference), int(slide_id or 0)))
        break
    return entries


def _master_entries(document: PptRecord, budget: RecordBudget) -> list[tuple[int, int]]:
    """返回 master persist 引用与 master id。"""

    entries: list[tuple[int, int]] = []
    for container in iter_descendants(document, budget=budget):
        if container.record_type != RT_SLIDE_LIST_WITH_TEXT or container.instance != 1:
            continue
        for record in iter_records(container.payload, budget=budget):
            if record.record_type != RT_SLIDE_PERSIST_ATOM:
                continue
            reference = get_u32(record.payload, 0)
            master_id = get_u32(record.payload, 12)
            if reference and master_id:
                entries.append((int(reference), int(master_id)))
        break
    return entries


def _notes_entries(document: PptRecord, budget: RecordBudget) -> list[int]:
    """返回 notes list 中的 persist 引用。"""

    references: list[int] = []
    for container in iter_descendants(document, budget=budget):
        if container.record_type != RT_SLIDE_LIST_WITH_TEXT or container.instance != 2:
            continue
        for record in iter_records(container.payload, budget=budget):
            if record.record_type == RT_SLIDE_PERSIST_ATOM:
                reference = get_u32(record.payload, 0)
                if reference:
                    references.append(int(reference))
        break
    return references


def _slide_master_id(slide: PptRecord, budget: RecordBudget) -> int | None:
    """读取 SlideAtom.masterIdRef。"""

    for child in iter_descendants(slide, budget=budget):
        if child.record_type == RT_SLIDE_ATOM:
            master_id = get_u32(child.payload, 12)
            return int(master_id) if master_id else None
    return None


def _presentation_size(document: PptRecord, budget: RecordBudget) -> tuple[int, int]:
    """从 DocumentAtom 读取页面 master units 尺寸。"""

    for child in iter_descendants(document, budget=budget):
        if child.record_type != RT_DOCUMENT_ATOM or len(child.payload) < 8:
            continue
        width = get_u32(child.payload, 0)
        height = get_u32(child.payload, 4)
        if width and height and width < 100_000 and height < 100_000:
            return int(width), int(height)
    return DEFAULT_SLIDE_WIDTH, DEFAULT_SLIDE_HEIGHT


def _decode_text_atom(record: PptRecord) -> str | None:
    """解码 TextCharsAtom 或 TextBytesAtom。"""

    if record.record_type == RT_TEXT_CHARS_ATOM:
        return utf16_text(record.payload)
    if record.record_type == RT_TEXT_BYTES_ATOM:
        return record.payload.decode("cp1252", "replace").rstrip("\x00")
    return None


def _safe_hyperlink_target(target: str) -> str | None:
    """仅允许公开输出中可安全表达的外部链接 scheme。"""

    normalized = target.strip()
    if not normalized:
        return None
    scheme = urlparse(normalized).scheme.lower()
    return normalized if scheme in _ALLOWED_LINK_SCHEMES else None


def _hyperlink_targets(document: PptRecord, budget: RecordBudget) -> dict[int, str]:
    """建立 ExHyperlinkId 到安全外链目标的映射。"""

    result: dict[int, str] = {}
    for container in iter_descendants(document, budget=budget):
        if container.record_type != RT_EXTERNAL_HYPERLINK or container.version != CONTAINER_VERSION:
            continue
        link_id = None
        strings: list[str] = []
        for child in iter_records(container.payload, budget=budget):
            if child.record_type == RT_EXTERNAL_HYPERLINK_ATOM:
                link_id = get_u32(child.payload, 0)
            elif child.record_type == RT_CSTRING:
                strings.append(utf16_text(child.payload))
        if link_id is None or not strings:
            continue
        safe_target = _safe_hyperlink_target(strings[-1])
        if safe_target is None:
            logger.warning(f"PPT_UNSAFE_HYPERLINK: hyperlink id={link_id} was downgraded")
            continue
        result[int(link_id)] = safe_target
    return result


def _interactive_spans(
    related_records: Iterable[PptRecord],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> list[tuple[int, int, str]]:
    """解析一段文本之后的 InteractiveInfo 与 UTF-16 范围。"""

    flattened: list[PptRecord] = []
    for record in related_records:
        flattened.append(record)
        if record.version == CONTAINER_VERSION:
            flattened.extend(iter_descendants(record, budget=budget))
    spans: list[tuple[int, int, str]] = []
    pending_id: int | None = None
    for record in flattened:
        if record.record_type == RT_INTERACTIVE_INFO_ATOM:
            pending_id = get_u32(record.payload, 4)
            continue
        if record.record_type != RT_TEXT_INTERACTIVE_INFO_ATOM or pending_id is None:
            continue
        start = get_u32(record.payload, 0)
        end = get_u32(record.payload, 4)
        target = hyperlinks.get(int(pending_id))
        if start is not None and end is not None and target and end > start:
            spans.append((int(start), int(end), target))
        pending_id = None
    return spans


def _utf16_width(text: str) -> int:
    """返回字符串占用的 UTF-16 code unit 数。"""

    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _master_level(levels: list[MasterLevel], depth: int) -> MasterLevel:
    """按深度取得母版默认值，超界时回退最后一个可用层级。"""

    if not levels:
        return MasterLevel()
    return levels[min(max(depth, 0), len(levels) - 1)]


def _paragraph_run_at(runs: list[ParagraphRun], offset: int) -> ParagraphRun:
    """返回覆盖指定 UTF-16 偏移的段落 run。"""

    cursor = 0
    for run in runs:
        if cursor <= offset < cursor + run.count:
            return run
        cursor += run.count
    return ParagraphRun(count=0, depth=0)


def _character_run_at(runs: list[CharacterRun], offset: int) -> CharacterRun:
    """返回覆盖指定 UTF-16 偏移的字符 run。"""

    cursor = 0
    for run in runs:
        if cursor <= offset < cursor + run.count:
            return run
        cursor += run.count
    return CharacterRun(count=0)


def _hyperlink_at(spans: list[tuple[int, int, str]], offset: int) -> str | None:
    """返回覆盖当前 UTF-16 偏移的超链接目标。"""

    return next((target for start, end, target in spans if start <= offset < end), None)


def _resolve_run(
    text: str,
    explicit: CharacterRun,
    master: MasterLevel,
    hyperlink: str | None,
) -> PptTextRun:
    """把字符异常属性覆盖到母版默认值上。"""

    return PptTextRun(
        text=text,
        bold=explicit.bold if explicit.bold is not None else bool(master.bold),
        italic=explicit.italic if explicit.italic is not None else bool(master.italic),
        underline=(explicit.underline if explicit.underline is not None else bool(master.underline) or hyperlink is not None),
        strike=bool(explicit.strike),
        baseline=explicit.baseline if explicit.baseline is not None else master.baseline,
        hyperlink=hyperlink,
    )


def _flush_text_run(
    output: list[PptTextRun],
    text: str,
    style: PptTextRun | None,
) -> None:
    """追加非空文本，并合并相邻同样式 run。"""

    if not text or style is None:
        return
    candidate = replace(style, text=text)
    if output and replace(output[-1], text="") == replace(candidate, text=""):
        output[-1] = replace(output[-1], text=f"{output[-1].text}{text}")
        return
    output.append(candidate)


def _build_paragraphs(
    text: str,
    styles: StyleRuns,
    master_levels: list[MasterLevel],
    hyperlinks: list[tuple[int, int, str]],
) -> tuple[PptParagraph, ...]:
    """把 UTF-16 属性范围转换为带样式 run 的段落。"""

    paragraphs: list[PptParagraph] = []
    current_runs: list[PptTextRun] = []
    run_text: list[str] = []
    active_style: PptTextRun | None = None
    utf16_offset = 0
    paragraph_start = 0

    def flush_run() -> None:
        """把当前相同样式字符提交到段落。"""

        nonlocal run_text
        _flush_text_run(current_runs, "".join(run_text), active_style)
        run_text = []

    def flush_paragraph() -> None:
        """完成当前段落并解析列表属性。"""

        flush_run()
        paragraph_style = _paragraph_run_at(styles.paragraphs, paragraph_start)
        character_style = _character_run_at(styles.characters, paragraph_start)
        master = _master_level(master_levels, paragraph_style.depth)
        bullet = paragraph_style.bullet if paragraph_style.bullet is not None else bool(master.bullet)
        visible = any(run.text.strip() for run in current_runs)
        if visible:
            paragraphs.append(
                PptParagraph(
                    runs=tuple(current_runs),
                    depth=max(0, int(paragraph_style.depth)),
                    list_kind="unordered" if bullet else None,
                    pp9rt=character_style.pp9rt,
                )
            )
        current_runs.clear()

    for character in text:
        paragraph_style = _paragraph_run_at(styles.paragraphs, utf16_offset)
        master = _master_level(master_levels, paragraph_style.depth)
        explicit = _character_run_at(styles.characters, utf16_offset)
        hyperlink = _hyperlink_at(hyperlinks, utf16_offset)
        style = _resolve_run("", explicit, master, hyperlink)
        if active_style is None or replace(active_style, text="") != replace(style, text=""):
            flush_run()
            active_style = style
        if character == "\r":
            flush_paragraph()
            paragraph_start = utf16_offset + 1
            active_style = None
        elif character in {"\x0b", "\n"}:
            run_text.append("\n")
        elif not unicodedata.category(character).startswith("C") or character == "\t":
            run_text.append(character)
        utf16_offset += _utf16_width(character)
    if run_text or current_runs:
        flush_paragraph()
    return tuple(paragraphs)


def _parse_text_contents(
    records: list[PptRecord],
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> list[_TextContent]:
    """从一组相邻记录中解析所有文本形状内容。"""

    result: list[_TextContent] = []
    text_type = 4
    for index, record in enumerate(records):
        if record.record_type == RT_TEXT_HEADER_ATOM:
            text_type = int(get_u32(record.payload, 0) or 4)
            continue
        text = _decode_text_atom(record)
        if text is None:
            continue
        tail = records[index + 1 :]
        next_text_index = next(
            (
                position
                for position, candidate in enumerate(tail)
                if candidate.record_type in {RT_TEXT_CHARS_ATOM, RT_TEXT_BYTES_ATOM, RT_SLIDE_PERSIST_ATOM}
            ),
            len(tail),
        )
        related = tail[:next_text_index]
        style_atom = next(
            (candidate for candidate in related if candidate.record_type == RT_STYLE_TEXT_PROP_ATOM),
            None,
        )
        styles = parse_style_text(style_atom.payload, _utf16_width(text)) if style_atom is not None else StyleRuns()
        spans = _interactive_spans(related, hyperlinks, budget)
        paragraphs = _build_paragraphs(
            text,
            styles,
            master_styles.get(text_type, []),
            spans,
        )
        result.append(_TextContent(paragraphs=paragraphs, text_type=text_type))
    return result


def _external_text_records(
    document: PptRecord,
    budget: RecordBudget,
) -> dict[int, list[PptRecord]]:
    """按 slide persist 引用收集 SlideListWithText 中的外置文本记录。"""

    grouped: dict[int, list[PptRecord]] = {}
    for container in iter_descendants(document, budget=budget):
        if container.record_type != RT_SLIDE_LIST_WITH_TEXT or container.instance != 0:
            continue
        current_reference: int | None = None
        for record in iter_records(container.payload, budget=budget):
            if record.record_type == RT_SLIDE_PERSIST_ATOM:
                reference = get_u32(record.payload, 0)
                current_reference = int(reference) if reference else None
                if current_reference is not None:
                    grouped.setdefault(current_reference, [])
                continue
            if current_reference is not None:
                grouped[current_reference].append(record)
        break
    return grouped


def _master_styles(record: PptRecord, budget: RecordBudget) -> dict[int, list[MasterLevel]]:
    """解析一个 master container 的逐文本类型默认样式。"""

    result: dict[int, list[MasterLevel]] = {}
    for child in iter_descendants(record, budget=budget):
        if child.record_type != RT_TEXT_MASTER_STYLE_ATOM:
            continue
        result.setdefault(
            int(child.instance),
            parse_master_style(child.payload, int(child.instance)),
        )
    return result


def _collect_masters(
    layout: _PersistLayout,
    data: bytes,
    budget: RecordBudget,
) -> tuple[dict[int, tuple[PptRecord, dict[int, list[MasterLevel]]]], tuple[PptRecord, dict[int, list[MasterLevel]]] | None]:
    """按 master id 建立容器与样式映射，并返回确定性 fallback。"""

    masters: dict[int, tuple[PptRecord, dict[int, list[MasterLevel]]]] = {}
    for reference, master_id in _master_entries(layout.document, budget):
        offset = layout.persist.get(reference)
        if offset is None:
            continue
        record = record_at(data, offset, budget=budget)
        if record is None or record.record_type not in {RT_MAIN_MASTER, RT_SLIDE}:
            continue
        masters.setdefault(master_id, (record, _master_styles(record, budget)))

    if not masters:
        for root in iter_records(data, budget=budget):
            if root.record_type != RT_MAIN_MASTER:
                continue
            masters.setdefault(0, (root, _master_styles(root, budget)))
    fallback = next(iter(masters.values()), None)
    return masters, fallback


def _fopt_properties(record: PptRecord) -> dict[int, int]:
    """读取 OfficeArt FOPT 的简单属性，重复属性以后者覆盖。"""

    count = int(record.instance)
    properties: dict[int, int] = {}
    for index in range(count):
        offset = index * 6
        if offset + 6 > len(record.payload):
            break
        opid, value = struct.unpack_from("<HI", record.payload, offset)
        properties[opid & 0x3FFF] = int(value)
    return properties


def _fopt_complex_properties(record: PptRecord) -> dict[int, bytes]:
    """读取 OfficeArt FOPT 中紧随属性数组的复杂载荷。"""

    count = int(record.instance)
    cursor = count * 6
    result: dict[int, bytes] = {}
    entries: list[tuple[int, int, bool]] = []
    for index in range(count):
        offset = index * 6
        if offset + 6 > len(record.payload):
            break
        opid, value = struct.unpack_from("<HI", record.payload, offset)
        entries.append((opid & 0x3FFF, int(value), bool(opid & 0x8000)))
    for property_id, size, is_complex in entries:
        if not is_complex:
            continue
        end = cursor + size
        if size < 0 or end < cursor or end > len(record.payload):
            break
        result[property_id] = record.payload[cursor:end]
        cursor = end
    return result


def _shape_properties(shape: PptRecord, budget: RecordBudget) -> dict[int, int]:
    """合并 shape 的 primary 与 tertiary FOPT 属性。"""

    properties: dict[int, int] = {}
    for child in _direct_children(shape, budget):
        if child.record_type in {RT_OFFICEART_FOPT, RT_OFFICEART_TERTIARY_FOPT}:
            properties.update(_fopt_properties(child))
    return properties


def _shape_external_object_id(shape: PptRecord, budget: RecordBudget) -> int | None:
    """从 OfficeArtClientData 读取 ExObjRefAtom 的外部对象 id。"""

    for child in _direct_children(shape, budget):
        if child.record_type != RT_OFFICEART_CLIENT_DATA:
            continue
        candidates: Iterable[PptRecord]
        if child.version == CONTAINER_VERSION:
            candidates = iter_descendants(child, budget=budget)
        else:
            candidates = iter_records(child.payload, budget=budget)
        for candidate in candidates:
            if candidate.record_type != RT_EXTERNAL_OBJECT_REF_ATOM:
                continue
            reference = get_u32(candidate.payload, 0)
            return int(reference) if reference else None
    return None


def _embedded_object_references(
    document: PptRecord,
    budget: RecordBudget,
) -> dict[int, tuple[int, int]]:
    """收集支持的嵌入 OLE subtype、exObjId 与 persistIdRef。"""

    references: dict[int, tuple[int, int]] = {}
    for atom in iter_descendants(document, budget=budget):
        if atom.record_type != RT_EXTERNAL_OLE_OBJECT_ATOM or len(atom.payload) < 24:
            continue
        object_type = get_u32(atom.payload, 4)
        object_id = get_u32(atom.payload, 8)
        object_subtype = get_u32(atom.payload, 12)
        persist_id = get_u32(atom.payload, 16)
        if (
            object_type == 0
            and object_subtype
            in {
                _OLE_SUBTYPE_GRAPH,
                _OLE_SUBTYPE_EQUATION,
                _OLE_SUBTYPE_EXCEL_CHART,
            }
            and object_id
            and persist_id
        ):
            references.setdefault(
                int(object_id),
                (int(object_subtype), int(persist_id)),
            )
    return references


def _decompress_ole_storage(record: PptRecord) -> bytes | None:
    """按 ExOleObjStg instance 有界恢复独立 CFB 字节。"""

    if record.record_type != RT_EXTERNAL_OLE_OBJECT_STG:
        return None
    if record.instance == 0:
        if len(record.payload) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(f"OLE object exceeds max_entry_bytes={MAX_ENTRY_BYTES}")
        storage = record.payload
        if not storage.startswith(_CFB_MAGIC):
            return None
        try:
            with BoundedOleReader(storage):
                pass
        except ValueError:
            return None
        return storage
    if record.instance != 1 or len(record.payload) < 4:
        return None
    declared_size = int(get_u32(record.payload, 0) or 0)
    if declared_size <= 0:
        return None
    if declared_size > MAX_ENTRY_BYTES:
        raise LegacyOfficeResourceLimitError(f"OLE object exceeds max_entry_bytes={MAX_ENTRY_BYTES}")
    try:
        inflater = zlib.decompressobj(zlib.MAX_WBITS)
        storage = inflater.decompress(record.payload[4:], MAX_ENTRY_BYTES + 1)
        if len(storage) > MAX_ENTRY_BYTES:
            raise LegacyOfficeResourceLimitError(f"OLE object exceeds max_entry_bytes={MAX_ENTRY_BYTES}")
        storage += inflater.flush(MAX_ENTRY_BYTES + 1 - len(storage))
    except (ValueError, zlib.error):
        return None
    if len(storage) > MAX_ENTRY_BYTES:
        raise LegacyOfficeResourceLimitError(f"OLE object exceeds max_entry_bytes={MAX_ENTRY_BYTES}")
    if len(storage) != declared_size or inflater.unconsumed_tail:
        return None
    if not inflater.eof:
        if not record.payload[4:].endswith(_ZLIB_SYNC_FLUSH_SUFFIX):
            return None
    elif inflater.unused_data:
        return None
    if not storage.startswith(_CFB_MAGIC):
        return None
    try:
        with BoundedOleReader(storage):
            pass
    except ValueError:
        return None
    return storage


def _embedded_object_map(
    layout: _PersistLayout,
    data: bytes,
    budget: RecordBudget,
) -> dict[int, _EmbeddedOleObject]:
    """统一解压受支持的 PPT persist OLE storages 并共享资源预算。"""

    objects: dict[int, _EmbeddedOleObject] = {}
    asset_total = 0
    for object_id, (subtype, persist_id) in _embedded_object_references(
        layout.document,
        budget,
    ).items():
        offset = layout.persist.get(persist_id)
        record = record_at(data, offset, budget=budget) if offset is not None else None
        if record is None:
            logger.warning(
                "PPT_OLE_FALLBACK: exObjId={} persistIdRef={} is missing",
                object_id,
                persist_id,
            )
            continue
        storage = _decompress_ole_storage(record)
        if storage is None:
            logger.warning(
                "PPT_OLE_FALLBACK: exObjId={} has an invalid OLE storage",
                object_id,
            )
            continue
        asset_total += len(storage)
        if asset_total > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(f"embedded assets exceed max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}")
        objects[object_id] = _EmbeddedOleObject(subtype=subtype, storage=storage)
    return objects


def _equation_map(objects: dict[int, _EmbeddedOleObject]) -> dict[int, str]:
    """从共享 OLE 对象集合解析 Equation Native。"""

    equations: dict[int, str] = {}
    for object_id, embedded in objects.items():
        if embedded.subtype != _OLE_SUBTYPE_EQUATION:
            continue
        latex = decode_equation_object(embedded.storage)
        if latex is None:
            logger.warning(
                "PPT_MTEF_FALLBACK: exObjId={} has an invalid or unsupported Equation Native stream",
                object_id,
            )
            continue
        equations[object_id] = latex
    return equations


def _chart_map(objects: dict[int, _EmbeddedOleObject]) -> dict[int, str]:
    """从共享 OLE 对象集合解析 Excel.Chart 与 MSGraph.Chart 数据表。"""

    charts: dict[int, str] = {}
    for object_id, embedded in objects.items():
        if embedded.subtype not in {_OLE_SUBTYPE_GRAPH, _OLE_SUBTYPE_EXCEL_CHART}:
            continue
        content = extract_embedded_chart_html_from_storage(embedded.storage)
        if content is None:
            logger.warning(
                "PPT_CHART_FALLBACK: exObjId={} has no supported chart datasheet",
                object_id,
            )
            continue
        charts[object_id] = content
    return charts


def _shape_complex_properties(shape: PptRecord, budget: RecordBudget) -> dict[int, bytes]:
    """合并 shape 的复杂 FOPT 属性。"""

    properties: dict[int, bytes] = {}
    for child in _direct_children(shape, budget):
        if child.record_type in {RT_OFFICEART_FOPT, RT_OFFICEART_TERTIARY_FOPT}:
            properties.update(_fopt_complex_properties(child))
    return properties


def _shape_type(shape: PptRecord, budget: RecordBudget) -> int | None:
    """返回 OfficeArtFSP header 中的 MSOSPT 类型。"""

    for child in _direct_children(shape, budget):
        if child.record_type == RT_OFFICEART_FSP:
            return int(child.instance)
    return None


def _is_background_shape(shape: PptRecord, budget: RecordBudget) -> bool:
    """判断 FSP 标志是否把 shape 标记为背景。"""

    for child in _direct_children(shape, budget):
        if child.record_type != RT_OFFICEART_FSP or len(child.payload) < 8:
            continue
        return bool(int(get_u32(child.payload, 4) or 0) & 0x0000_0400)
    return False


def _is_placeholder(shape: PptRecord, budget: RecordBudget) -> bool:
    """判断 shape 的 ClientData 是否包含 OEPlaceholderAtom。"""

    for child in _direct_children(shape, budget):
        if child.record_type != RT_OFFICEART_CLIENT_DATA:
            continue
        if child.version == CONTAINER_VERSION:
            candidates: Iterable[PptRecord] = iter_descendants(child, budget=budget)
        else:
            candidates = iter_records(child.payload, budget=budget)
        if any(candidate.record_type == RT_OE_PLACEHOLDER_ATOM for candidate in candidates):
            return True
    return False


def _client_rect(payload: bytes) -> tuple[float, float, float, float] | None:
    """把 OfficeArtClientAnchor 转成 left/top/right/bottom。"""

    if len(payload) < 8:
        return None
    top, left, right, bottom = struct.unpack_from("<4h", payload)
    return float(left), float(top), float(right), float(bottom)


def _child_rect(payload: bytes) -> tuple[float, float, float, float] | None:
    """把 OfficeArtChildAnchor 转成 left/top/right/bottom。"""

    if len(payload) < 16:
        return None
    left, top, right, bottom = struct.unpack_from("<4i", payload)
    return float(left), float(top), float(right), float(bottom)


def _map_group_rect(
    rect: tuple[float, float, float, float],
    space: _GroupSpace,
) -> tuple[float, float, float, float]:
    """把 group 子坐标线性映射到幻灯片坐标。"""

    coord_width = max(1.0, float(space.coord_right - space.coord_left))
    coord_height = max(1.0, float(space.coord_bottom - space.coord_top))
    scale_x = (space.abs_right - space.abs_left) / coord_width
    scale_y = (space.abs_bottom - space.abs_top) / coord_height
    left, top, right, bottom = rect
    return (
        space.abs_left + (left - space.coord_left) * scale_x,
        space.abs_top + (top - space.coord_top) * scale_y,
        space.abs_left + (right - space.coord_left) * scale_x,
        space.abs_top + (bottom - space.coord_top) * scale_y,
    )


def _shape_bbox(
    shape: PptRecord,
    space: _GroupSpace | None,
    budget: RecordBudget,
) -> tuple[float, float, float, float] | None:
    """解析 shape anchor，并应用父 group 坐标映射。"""

    children = _direct_children(shape, budget)
    for child in children:
        if child.record_type == RT_OFFICEART_CLIENT_ANCHOR:
            rect = _client_rect(child.payload)
            if rect is not None:
                return rect
    for child in children:
        if child.record_type != RT_OFFICEART_CHILD_ANCHOR:
            continue
        rect = _child_rect(child.payload)
        if rect is None:
            continue
        if space is not None:
            return _map_group_rect(rect, space)
        if max(abs(value) for value in rect) > 100_000:
            return tuple(value * 576.0 / 914_400.0 for value in rect)  # type: ignore[return-value]
        return rect
    return None


def _group_space(
    group_shape: PptRecord,
    parent: _GroupSpace | None,
    budget: RecordBudget,
) -> _GroupSpace | None:
    """解析 group 自身坐标系与它在父坐标系中的外框。"""

    children = _direct_children(group_shape, budget)
    fspgr = next(
        (child for child in children if child.record_type == RT_OFFICEART_FSPGR and len(child.payload) >= 16),
        None,
    )
    if fspgr is None:
        return parent
    coord_left, coord_top, coord_right, coord_bottom = struct.unpack_from("<4i", fspgr.payload)
    if coord_right <= coord_left or coord_bottom <= coord_top:
        return parent
    raw_rect = None
    for child in children:
        if child.record_type == RT_OFFICEART_CLIENT_ANCHOR:
            raw_rect = _client_rect(child.payload)
            break
        if child.record_type == RT_OFFICEART_CHILD_ANCHOR:
            raw_rect = _child_rect(child.payload)
            break
    if raw_rect is None:
        return parent
    rect = _map_group_rect(raw_rect, parent) if parent is not None else raw_rect
    left, top, right, bottom = rect
    return _GroupSpace(
        coord_left=int(coord_left),
        coord_top=int(coord_top),
        coord_right=int(coord_right),
        coord_bottom=int(coord_bottom),
        abs_left=left,
        abs_top=top,
        abs_right=right,
        abs_bottom=bottom,
    )


def _is_table_group(group_shape: PptRecord, budget: RecordBudget) -> bool:
    """读取 tableProperties.fIsTable 标志。"""

    return bool(_shape_properties(group_shape, budget).get(FOPT_TABLE_PROPERTIES, 0) & 0x1)


def _collect_shapes(slide: PptRecord, budget: RecordBudget) -> _ShapeCollection:
    """单次遍历收集叶子 shape，并保留 table group 的成员边界。"""

    shapes: list[_ShapeInfo] = []
    groups: dict[tuple[int, ...], tuple[PptRecord, list[_ShapeInfo], int, bool]] = {}
    order = 0

    def walk(
        record: PptRecord,
        space: _GroupSpace | None,
        path: tuple[int, ...],
        active_table: tuple[int, ...] | None,
    ) -> None:
        """递归遍历 OfficeArt 容器并传播 group 空间与 table 身份。"""

        nonlocal order
        if record.record_type == RT_OFFICEART_SPGR_CONTAINER and record.version == CONTAINER_VERSION:
            children = _direct_children(record, budget)
            if not children:
                return
            group_shape = children[0]
            group_key = path + (0,)
            nested_space = _group_space(group_shape, space, budget)
            authoritative = _is_table_group(group_shape, budget)
            groups[group_key] = (group_shape, [], order, authoritative)
            table_key = group_key
            for index, child in enumerate(children[1:], start=1):
                walk(child, nested_space, path + (index,), table_key)
            return
        if record.record_type == RT_OFFICEART_SP_CONTAINER:
            bbox = _shape_bbox(record, space, budget)
            if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return
            info = _ShapeInfo(
                key=path,
                record=record,
                space=space,
                bbox=bbox,
                order=order,
                group_key=active_table,
            )
            order += 1
            shapes.append(info)
            if active_table in groups:
                groups[active_table][1].append(info)
            return
        if record.version == CONTAINER_VERSION:
            for index, child in enumerate(_direct_children(record, budget)):
                walk(child, space, path + (index,), active_table)

    walk(slide, None, (), None)
    table_groups = tuple(
        _TableGroup(
            key=key,
            group_shape=group_shape,
            shapes=tuple(members),
            order=group_order,
            authoritative=authoritative,
        )
        for key, (group_shape, members, group_order, authoritative) in groups.items()
    )
    return _ShapeCollection(shapes=tuple(shapes), table_groups=table_groups)


def _shape_text_content(
    shape: _ShapeInfo,
    external_text: list[_TextContent],
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> _TextContent | None:
    """解析 shape 的 ClientTextbox，必要时回退到 OutlineTextRefAtom。"""

    textbox = next(
        (child for child in _direct_children(shape.record, budget) if child.record_type == RT_OFFICEART_CLIENT_TEXTBOX),
        None,
    )
    if textbox is None:
        return None
    textbox_records = list(iter_descendants(textbox, budget=budget))
    contents = _parse_text_contents(textbox_records, master_styles, hyperlinks, budget)
    if contents:
        return _apply_shape_numbering(contents[0], shape.record, budget)
    reference = next(
        (child for child in textbox_records if child.record_type == RT_OUTLINE_TEXT_REF_ATOM and len(child.payload) >= 4),
        None,
    )
    if reference is None:
        return None
    index = int(get_u32(reference.payload, 0) or 0)
    for candidate in (index, index - 1) if index else (0,):
        if 0 <= candidate < len(external_text):
            return _apply_shape_numbering(external_text[candidate], shape.record, budget)
    return None


def _skip_style_text9_cf(payload: bytes, position: int) -> int | None:
    """跳过 TextCFException9；未知扩展位时停止该 atom 的解析。"""

    mask = get_u32(payload, position)
    if mask is None:
        return None
    position += 4
    if mask == 0:
        return position
    # 当前只需要自动编号，复杂 CF9 不影响已解析文本样式，安全终止后续槽位。
    return None


def _skip_style_text9_si(payload: bytes, position: int) -> int | None:
    """跳过 TextSIException 中固定长度的语言与拼写字段。"""

    mask = get_u32(payload, position)
    if mask is None:
        return None
    position += 4
    for bit, size in ((0, 2), (1, 2), (2, 2), (5, 4), (6, 2)):
        if mask & (1 << bit):
            position += size
    if mask & (1 << 9) or position > len(payload):
        return None
    return position


def _parse_style_text9_numbering(payload: bytes) -> dict[int, _NumberingStyle]:
    """解析 StyleTextProp9 数组中与自动编号有关的三个字段。"""

    result: dict[int, _NumberingStyle] = {}
    position = 0
    slot = 0
    while position + 12 <= len(payload) and slot < 16:
        mask = get_u32(payload, position)
        if mask is None or mask & ~0x0380_0000:
            break
        position += 4
        if mask & 0x0080_0000:
            position += 2
        enabled = None
        if mask & 0x0200_0000:
            value = get_i16(payload, position)
            if value is None:
                break
            enabled = value == 1
            position += 2
        start = None
        if mask & 0x0100_0000:
            scheme = get_u16(payload, position)
            start_value = get_u16(payload, position + 2)
            if scheme is None or start_value is None:
                break
            start = max(0, int(start_value))
            position += 4
        position = _skip_style_text9_cf(payload, position) or -1
        if position < 0:
            break
        position = _skip_style_text9_si(payload, position) or -1
        if position < 0:
            break
        result[slot] = _NumberingStyle(enabled=enabled, start=start)
        slot += 1
    return result


def _shape_numbering_styles(
    shape: PptRecord,
    budget: RecordBudget,
) -> dict[int, _NumberingStyle]:
    """从 PP9ShapeBinaryTagExtension 取出 StyleTextProp9 自动编号。"""

    for tag in iter_descendants(shape, budget=budget):
        if tag.record_type != 0x138A or tag.version != CONTAINER_VERSION:
            continue
        children = list(iter_records(tag.payload, budget=budget))
        marker = next(
            (child for child in children if child.record_type == RT_CSTRING and utf16_text(child.payload) == "___PPT9"),
            None,
        )
        blob = next((child for child in children if child.record_type == 0x138B), None)
        if marker is None or blob is None:
            continue
        atom = record_at(blob.payload, 0, budget=budget)
        if atom is not None and atom.record_type == 0x0FAC:
            return _parse_style_text9_numbering(atom.payload)
    return {}


def _apply_shape_numbering(
    content: _TextContent,
    shape: PptRecord,
    budget: RecordBudget,
) -> _TextContent:
    """按 paragraph 起始字符的 pp9rt 槽位覆盖列表类型和起始编号。"""

    numbering = _shape_numbering_styles(shape, budget)
    if not numbering:
        return content
    paragraphs: list[PptParagraph] = []
    for paragraph in content.paragraphs:
        style = numbering.get(paragraph.pp9rt)
        if style is None or style.enabled is None:
            paragraphs.append(paragraph)
            continue
        paragraphs.append(
            replace(
                paragraph,
                list_kind="ordered" if style.enabled else None,
                start=style.start if style.enabled else None,
            )
        )
    return replace(content, paragraphs=tuple(paragraphs))


def _cluster_coordinates(values: list[float], tolerance: float = 8.0) -> list[float]:
    """把生产器舍入误差导致的近邻坐标合并为稳定边界。"""

    if not values:
        return []
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and abs(value - sum(clusters[-1]) / len(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _boundary_index(boundaries: list[float], value: float, tolerance: float = 16.0) -> int | None:
    """返回与坐标最近的边界索引，偏差过大时拒绝映射。"""

    if not boundaries:
        return None
    index = min(range(len(boundaries)), key=lambda candidate: abs(boundaries[candidate] - value))
    return index if abs(boundaries[index] - value) <= tolerance else None


def _table_from_group(
    group: _TableGroup,
    external_text: list[_TextContent],
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> PptTableElement | None:
    """从 fIsTable group 的矩形单元格恢复含合并信息的完整网格。"""

    cell_shapes: list[tuple[_ShapeInfo, _TextContent | None]] = []
    x_values: list[float] = []
    y_values: list[float] = []
    for shape in group.shapes:
        shape_type = _shape_type(shape.record, budget)
        left, top, right, bottom = shape.bbox
        width = right - left
        height = bottom - top
        # 线条只贡献边界，不成为单元格。
        if shape_type in {20, 32, 33, 34, 35, 36, 37, 38, 39, 40} or width <= 2 or height <= 2:
            if width <= 2:
                x_values.extend((left, right))
            if height <= 2:
                y_values.extend((top, bottom))
            continue
        if shape_type != 1:
            continue
        content = _shape_text_content(
            shape,
            external_text,
            master_styles,
            hyperlinks,
            budget,
        )
        cell_shapes.append((shape, content))
        x_values.extend((left, right))
        y_values.extend((top, bottom))
    if len(cell_shapes) < 4:
        return None

    x_boundaries = _cluster_coordinates(x_values)
    y_boundaries = _cluster_coordinates(y_values)
    cols = len(x_boundaries) - 1
    rows = len(y_boundaries) - 1
    if rows < 2 or cols < 2 or rows * cols > MAX_GRID_SLOTS:
        return None

    coverage: dict[tuple[int, int], tuple[int, int]] = {}
    cells: list[PptTableCell] = []
    for shape, content in cell_shapes:
        left, top, right, bottom = shape.bbox
        col_start = _boundary_index(x_boundaries, left)
        col_end = _boundary_index(x_boundaries, right)
        row_start = _boundary_index(y_boundaries, top)
        row_end = _boundary_index(y_boundaries, bottom)
        if (
            col_start is None
            or col_end is None
            or row_start is None
            or row_end is None
            or col_end <= col_start
            or row_end <= row_start
        ):
            return None
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                if (row, col) in coverage:
                    return None
                coverage[(row, col)] = (row_start, col_start)
        cells.append(
            PptTableCell(
                row=row_start,
                col=col_start,
                row_span=row_end - row_start,
                col_span=col_end - col_start,
                paragraphs=content.paragraphs if content is not None else (),
            )
        )
    if len(coverage) != rows * cols:
        return None

    cells.sort(key=lambda cell: (cell.row, cell.col))
    return PptTableElement(
        rows=rows,
        cols=cols,
        cells=tuple(cells),
        bbox=(x_boundaries[0], y_boundaries[0], x_boundaries[-1], y_boundaries[-1]),
        order=group.order,
        shape_offsets=frozenset(shape.key for shape in group.shapes),
    )


def _decode_blip(record: PptRecord) -> _ImagePayload | None:
    """通过 legacy-office 共享层解码一个 OfficeArt BLIP。"""

    return decode_officeart_blip(
        OfficeArtRecord(
            offset=record.offset,
            version=record.version,
            instance=record.instance,
            record_type=record.record_type,
            payload=record.payload,
        )
    )


def _decode_bse_body(body: bytes, budget: RecordBudget) -> _ImagePayload | None:
    """从 FBSE body 的可选内嵌 BLIP 中提取图片。"""

    if len(body) < 36:
        return None
    name_length = body[33]
    inner_offset = 36 + int(name_length)
    inner = record_at(body, inner_offset, budget=budget)
    return _decode_blip(inner) if inner is not None else None


def _picture_map(
    document: PptRecord,
    pictures: bytes,
    budget: RecordBudget,
) -> dict[int, _ImagePayload]:
    """按 BStore 中 1-based BSE 序号建立图片资源映射。"""

    result: dict[int, _ImagePayload] = {}
    asset_total = 0
    bse_records = [record for record in iter_descendants(document, budget=budget) if record.record_type == RT_OFFICEART_BSE]
    for index, bse in enumerate(bse_records[:MAX_PICTURE_RECORDS], start=1):
        decoded = None
        picture_offset = get_u32(bse.payload, 28)
        if picture_offset is not None and picture_offset < len(pictures):
            picture_record = record_at(pictures, int(picture_offset), budget=budget)
            if picture_record is not None:
                if picture_record.record_type == RT_OFFICEART_BSE:
                    decoded = _decode_bse_body(picture_record.payload, budget)
                else:
                    decoded = _decode_blip(picture_record)
        if decoded is None:
            decoded = _decode_bse_body(bse.payload, budget)
        if decoded is None:
            continue
        asset_total += len(decoded.data)
        if asset_total > MAX_ASSET_TOTAL_BYTES:
            raise LegacyOfficeResourceLimitError(f"embedded assets exceed max_asset_total_bytes={MAX_ASSET_TOTAL_BYTES}")
        result[index] = decoded
    if len(bse_records) > MAX_PICTURE_RECORDS:
        logger.warning(f"PPT_PICTURE_LIMIT: ignored BSE records after {MAX_PICTURE_RECORDS}")
    return result


def _image_from_shape(
    shape: _ShapeInfo,
    image_map: dict[int, _ImagePayload],
    equation_decoder: OfficeImageEquationDecoder,
    budget: RecordBudget,
) -> PptImageElement | PptEquationElement | None:
    """把 shape 的 pib 属性解析为图片或 comment 内公式。"""

    reference = _shape_properties(shape.record, budget).get(FOPT_PIB)
    if reference is None:
        return None
    image = image_map.get(int(reference))
    if image is None:
        logger.warning(f"PPT_IMAGE_REFERENCE_MISSING: shape={shape.key}, pib={reference}")
        return None
    latex = equation_decoder.decode(
        image.data,
        part_name=f"picture.{image.extension}",
        content_type=image.content_type,
    )
    if latex:
        return PptEquationElement(
            latex=latex,
            bbox=shape.bbox,
            order=shape.order,
            shape_offset=shape.order,
        )
    data_uri = serialize_office_image(
        image.data,
        part_name=f"picture.{image.extension}",
        content_type=image.content_type,
    )
    if not data_uri:
        logger.warning(f"PPT_IMAGE_UNSUPPORTED: shape={shape.key}, type={image.content_type}")
        return None
    return PptImageElement(
        image_base64=data_uri,
        bbox=shape.bbox,
        order=shape.order,
        shape_offset=shape.order,
    )


def _is_small_picture(
    bbox: tuple[float, float, float, float],
    slide_width: int,
    slide_height: int,
) -> bool:
    """复用现代 PPTX 的尺寸阈值过滤装饰性小图。"""

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0 or slide_width <= 0 or slide_height <= 0:
        return False
    if width < 0.1 * slide_width or height < 0.1 * slide_height:
        return True
    return width * height / float(slide_width * slide_height) < 0.01


def _slide_elements(
    slide: PptRecord,
    external_text: list[_TextContent],
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    image_map: dict[int, _ImagePayload],
    equation_map: dict[int, str],
    chart_map: dict[int, str],
    image_equation_decoder: OfficeImageEquationDecoder,
    slide_width: int,
    slide_height: int,
    budget: RecordBudget,
) -> list[PptTextElement | PptImageElement | PptEquationElement | PptChartElement | PptTableElement]:
    """把一张 slide 的 shapes 转换为文本、表格、chart 和图片元素。"""

    collection = _collect_shapes(slide, budget)
    tables: list[PptTableElement] = []
    consumed_keys: set[tuple[int, ...]] = set()
    for group in sorted(
        collection.table_groups,
        key=lambda candidate: (not candidate.authoritative, candidate.order),
    ):
        if any(shape.key in consumed_keys for shape in group.shapes):
            continue
        table = _table_from_group(
            group,
            external_text,
            master_styles,
            hyperlinks,
            budget,
        )
        if table is None and group.authoritative:
            logger.warning(f"PPT_TABLE_RECOVERY_FAILED: group={group.key}")
            continue
        if table is None:
            continue
        tables.append(table)
        consumed_keys.update(shape.key for shape in group.shapes)

    elements: list[PptTextElement | PptImageElement | PptEquationElement | PptChartElement | PptTableElement] = list(tables)
    for shape in collection.shapes:
        if shape.key in consumed_keys or _is_background_shape(shape.record, budget):
            continue
        external_object_id = _shape_external_object_id(shape.record, budget)
        chart = chart_map.get(external_object_id or 0)
        if chart:
            preview = _image_from_shape(
                shape,
                image_map,
                image_equation_decoder,
                budget,
            )
            elements.append(
                PptChartElement(
                    content=chart,
                    image_base64=(preview.image_base64 if isinstance(preview, PptImageElement) else None),
                    bbox=shape.bbox,
                    order=shape.order,
                    shape_offset=shape.order,
                )
            )
            continue
        equation = equation_map.get(external_object_id or 0)
        if equation:
            elements.append(
                PptEquationElement(
                    latex=equation,
                    bbox=shape.bbox,
                    order=shape.order,
                    shape_offset=shape.order,
                )
            )
            continue
        content = _shape_text_content(
            shape,
            external_text,
            master_styles,
            hyperlinks,
            budget,
        )
        if content is not None and content.paragraphs:
            elements.append(
                PptTextElement(
                    paragraphs=content.paragraphs,
                    text_type=content.text_type,
                    bbox=shape.bbox,
                    order=shape.order,
                    shape_offset=shape.order,
                    is_placeholder=_is_placeholder(shape.record, budget),
                )
            )
        image_or_equation = _image_from_shape(
            shape,
            image_map,
            image_equation_decoder,
            budget,
        )
        if isinstance(image_or_equation, PptEquationElement):
            elements.append(image_or_equation)
        elif image_or_equation is not None and not _is_small_picture(
            image_or_equation.bbox,
            slide_width,
            slide_height,
        ):
            elements.append(image_or_equation)
    if not any(isinstance(element, PptTextElement) for element in elements):
        # Handmade、早期生产器或恢复路径可能把文本直接放在 SlideContainer 中。
        raw_contents = _parse_text_contents(
            list(iter_descendants(slide, budget=budget)),
            master_styles,
            hyperlinks,
            budget,
        )
        for offset, content in enumerate(raw_contents):
            if not content.paragraphs:
                continue
            elements.append(
                PptTextElement(
                    paragraphs=content.paragraphs,
                    text_type=content.text_type,
                    bbox=(
                        288.0,
                        288.0 + offset * 432.0,
                        float(max(slide_width - 288, 576)),
                        576.0 + offset * 432.0,
                    ),
                    order=len(elements) + offset,
                    shape_offset=len(elements) + offset,
                )
            )
    return elements


def _slide_hidden(slide: PptRecord, budget: RecordBudget) -> bool:
    """读取 SlideShowSlideInfoAtom 的隐藏标志。"""

    for child in iter_descendants(slide, budget=budget):
        if child.record_type != RT_SLIDE_SHOW_SLIDE_INFO_ATOM or len(child.payload) < 12:
            continue
        flags = get_u16(child.payload, 10)
        return bool(int(flags or 0) & 0x0004)
    return False


def _plain_paragraph_text(paragraph: PptParagraph) -> str:
    """返回一个内部段落的纯文本。"""

    return "".join(run.text for run in paragraph.runs)


def _note_paragraphs(
    note_record: PptRecord,
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> list[PptParagraph]:
    """提取 notes container 中的正文，排除占位符字段。"""

    paragraphs: list[PptParagraph] = []
    collection = _collect_shapes(note_record, budget)
    for shape in collection.shapes:
        content = _shape_text_content(shape, [], master_styles, hyperlinks, budget)
        if content is None:
            continue
        for paragraph in content.paragraphs:
            text = _plain_paragraph_text(paragraph).strip()
            if text and text != "*":
                paragraphs.append(paragraph)
    if paragraphs:
        return paragraphs

    # 少数生产器不把 notes 文本包在 OfficeArtClientTextbox 中。
    all_records = list(iter_descendants(note_record, budget=budget))
    for content in _parse_text_contents(all_records, master_styles, hyperlinks, budget):
        for paragraph in content.paragraphs:
            text = _plain_paragraph_text(paragraph).strip()
            if text and text != "*":
                paragraphs.append(paragraph)
    return paragraphs


def _notes_by_slide_id(
    layout: _PersistLayout,
    data: bytes,
    fallback_master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> tuple[dict[int, list[PptParagraph]], list[list[PptParagraph]]]:
    """按 NotesAtom.slideIdRef 绑定备注，并保留无主备注的顺序。"""

    bound: dict[int, list[PptParagraph]] = {}
    unbound: list[list[PptParagraph]] = []
    for reference in _notes_entries(layout.document, budget):
        offset = layout.persist.get(reference)
        if offset is None:
            continue
        note = record_at(data, offset, budget=budget)
        if note is None or note.record_type != RT_NOTES:
            continue
        slide_id = None
        for child in iter_descendants(note, budget=budget):
            if child.record_type == RT_NOTES_ATOM:
                slide_id = get_u32(child.payload, 0)
                break
        paragraphs = _note_paragraphs(
            note,
            fallback_master_styles,
            hyperlinks,
            budget,
        )
        if not paragraphs:
            continue
        if slide_id:
            bound[int(slide_id)] = paragraphs
        else:
            unbound.append(paragraphs)
    return bound, unbound


def _root_slide_records(data: bytes, budget: RecordBudget) -> list[PptRecord]:
    """恢复 persist 不可用时直接位于文档 stream 顶层的 slide records。"""

    return [
        record
        for record in iter_records(data, budget=budget)
        if record.record_type == RT_SLIDE and record.version == CONTAINER_VERSION
    ]


def _fallback_slide_from_text(
    data: bytes,
    master_styles: dict[int, list[MasterLevel]],
    hyperlinks: dict[int, str],
    budget: RecordBudget,
) -> PptSlide | None:
    """没有可靠 slide 边界时，把可恢复文本放入单个逻辑页。"""

    contents = _parse_text_contents(
        list(iter_records(data, budget=budget)),
        master_styles,
        hyperlinks,
        budget,
    )
    elements: list[PptTextElement] = []
    for order, content in enumerate(contents):
        if not content.paragraphs:
            continue
        elements.append(
            PptTextElement(
                paragraphs=content.paragraphs,
                text_type=content.text_type,
                bbox=(288.0, 288.0 + order * 432.0, 5472.0, 576.0 + order * 432.0),
                order=order,
                shape_offset=order,
            )
        )
    return PptSlide(slide_id=None, elements=list(elements)) if elements else None


def parse_ppt_document(
    powerpoint_document: bytes,
    *,
    current_user: bytes = b"",
    pictures: bytes = b"",
) -> PptPresentation:
    """把三个核心 PPT streams 解析为分页内部语义模型。"""

    if current_user:
        record_type = get_u16(current_user, 2)
        if record_type not in {None, 0x0FF6}:
            raise LegacyOfficeMalformedError("PowerPoint 95 or earlier Current User stream is unsupported")
        if get_u32(current_user, 12) == 0xF3D1_C4DF:
            raise LegacyOfficeEncryptedError("password-protected PPT is unsupported")

    budget = RecordBudget()
    # 先完整验证可递归记录深度，避免无 Document 的攻击形状被误报为普通坏文件。
    for root_record in iter_records(
        powerpoint_document,
        budget=budget,
        strict_first=True,
    ):
        for _ in iter_descendants(root_record, budget=budget):
            pass
    layout = _locate_document(powerpoint_document, current_user, budget)
    if any(child.record_type == RT_CRYPT_SESSION10_CONTAINER for child in iter_descendants(layout.document, budget=budget)):
        raise LegacyOfficeEncryptedError("encrypted PPT record stream is unsupported")

    width, height = _presentation_size(layout.document, budget)
    hyperlinks = _hyperlink_targets(layout.document, budget)
    image_map = _picture_map(layout.document, pictures, budget)
    image_equation_decoder = OfficeImageEquationDecoder()
    embedded_objects = _embedded_object_map(layout, powerpoint_document, budget)
    native_equations = _equation_map(embedded_objects)
    native_charts = _chart_map(embedded_objects)
    master_map, fallback_master = _collect_masters(
        layout,
        powerpoint_document,
        budget,
    )
    fallback_styles = fallback_master[1] if fallback_master is not None else {}
    external_records = _external_text_records(layout.document, budget)
    bound_notes, unbound_notes = _notes_by_slide_id(
        layout,
        powerpoint_document,
        fallback_styles,
        hyperlinks,
        budget,
    )

    slides: list[PptSlide] = []
    resolved_offsets: set[int] = set()
    resolved_slide_count = 0
    unbound_note_index = 0
    for reference, slide_id in _slide_entries(layout.document, budget):
        offset = layout.persist.get(reference)
        notes = bound_notes.get(slide_id)
        if notes is None and unbound_note_index < len(unbound_notes):
            notes = unbound_notes[unbound_note_index]
            unbound_note_index += 1
        if offset is None or offset in resolved_offsets:
            logger.warning(f"PPT_SLIDE_MISSING: persist_ref={reference}, slide_id={slide_id}")
            slides.append(PptSlide(slide_id=slide_id or None, notes=list(notes or [])))
            continue
        slide = record_at(powerpoint_document, offset, budget=budget)
        if slide is None or slide.record_type != RT_SLIDE:
            logger.warning(f"PPT_SLIDE_MALFORMED: persist_ref={reference}, slide_id={slide_id}")
            slides.append(PptSlide(slide_id=slide_id or None, notes=list(notes or [])))
            continue
        resolved_offsets.add(offset)
        resolved_slide_count += 1
        master = master_map.get(_slide_master_id(slide, budget) or -1, fallback_master)
        master_styles = master[1] if master is not None else {}
        external_text = _parse_text_contents(
            external_records.get(reference, []),
            master_styles,
            hyperlinks,
            budget,
        )
        slides.append(
            PptSlide(
                slide_id=slide_id or None,
                elements=_slide_elements(
                    slide,
                    external_text,
                    master_styles,
                    hyperlinks,
                    image_map,
                    native_equations,
                    native_charts,
                    image_equation_decoder,
                    width,
                    height,
                    budget,
                ),
                notes=list(notes or []),
                hidden=_slide_hidden(slide, budget),
            )
        )

    if resolved_slide_count == 0:
        slides = []
        logger.warning("PPT_SLIDE_RECOVERY: persist mapping did not resolve slide records")
        for slide in _root_slide_records(powerpoint_document, budget):
            master = master_map.get(_slide_master_id(slide, budget) or -1, fallback_master)
            master_styles = master[1] if master is not None else {}
            slides.append(
                PptSlide(
                    slide_id=None,
                    elements=_slide_elements(
                        slide,
                        [],
                        master_styles,
                        hyperlinks,
                        image_map,
                        native_equations,
                        native_charts,
                        image_equation_decoder,
                        width,
                        height,
                        budget,
                    ),
                    hidden=_slide_hidden(slide, budget),
                )
            )
    if not slides:
        fallback_slide = _fallback_slide_from_text(
            powerpoint_document,
            fallback_styles,
            hyperlinks,
            budget,
        )
        if fallback_slide is not None:
            logger.warning("PPT_SINGLE_PAGE_RECOVERY: slide boundaries were not recoverable")
            slides.append(fallback_slide)
    if not slides:
        raise LegacyOfficeMalformedError("PPT contains no recoverable slides or text")
    return PptPresentation(slides=slides, width=width, height=height)
