# Copyright (c) Opendatalab. All rights reserved.

"""把 WordDocument/Table/Data streams 解析为逐 section DOC 语义模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

from mineru.model.flash.legacy_office import LegacyOfficeMalformedError

from .bookmarks import parse_bookmarks
from .fib import (
    FCLCB_BOOKMARK_END,
    FCLCB_BOOKMARK_NAMES,
    FCLCB_BOOKMARK_START,
    FCLCB_BTE_CHPX,
    FCLCB_BTE_PAPX,
    FCLCB_CLX,
    FCLCB_DGG_INFO,
    FCLCB_ENDNOTE_REF,
    FCLCB_ENDNOTE_TEXT,
    FCLCB_FOOTNOTE_REF,
    FCLCB_FOOTNOTE_TEXT,
    FCLCB_HEADER,
    FCLCB_LIST_OVERRIDES,
    FCLCB_LISTS,
    FCLCB_SECTION,
    FCLCB_SHAPE_MAIN,
    FCLCB_STSHF,
    FileInformationBlock,
)
from .fields import apply_field_result, field_keyword, is_toc_field
from .formatting import FormattingRuns, parse_formatting_runs
from .images import ImageStore, floating_pictures, inline_picture
from .lists import ListTables, parse_list_tables
from .models import (
    DocCharStyle,
    DocDocument,
    DocElement,
    DocImage,
    DocParagraph,
    DocSection,
    DocTable,
    DocTableCell,
    DocTableCellFormat,
    DocTableFormat,
    DocTableRow,
    DocTextRun,
)
from .pieces import Piece, TextStream, codec_for_lid, extract_text, legacy_single_piece, parse_clx
from .records import DocBudget, bounded_slice, parse_plc
from .sprm import (
    PapDelta,
    apply_character_sprms,
    apply_paragraph_sprms,
    chpx_picture_location,
    chpx_style_id,
)
from .styles import Stylesheet, parse_stylesheet


@dataclass(slots=True)
class _FieldFrame:
    """一个尚未闭合的 Word 字段。"""

    instruction: str = ""
    in_result: bool = False
    transparent: bool = False
    runs: list[DocTextRun] = field(default_factory=list)


@dataclass(slots=True)
class _RawTableRow:
    """应用全表网格前的一行单元格。"""

    cells: list[DocTableCell]
    table_format: DocTableFormat | None


class _Assembler:
    """把全局文字流、格式 run 和辅助 PLC 组装为段落及表格。"""

    def __init__(
        self,
        *,
        text: TextStream,
        pieces: list[Piece],
        formatting: FormattingRuns,
        stylesheet: Stylesheet,
        lists: ListTables,
        bookmarks: dict[int, list[str]],
        data_stream: bytes,
        image_store: ImageStore,
        budget: DocBudget,
    ) -> None:
        """保存解析上下文；每个 story 开始时会重置字段栈。"""

        self.text = text
        self.pieces = pieces
        self.formatting = formatting
        self.stylesheet = stylesheet
        self.lists = lists
        self.bookmarks = bookmarks
        self.data_stream = data_stream
        self.image_store = image_store
        self.budget = budget
        self._fields: list[_FieldFrame] = []

    def _piece_prm(self, char_index: int) -> bytes:
        """返回字符所在 piece 的 Prm grpprl。"""

        if char_index >= len(self.text.piece_indexes):
            return b""
        piece_index = self.text.piece_indexes[char_index]
        return self.pieces[piece_index].prm if piece_index < len(self.pieces) else b""

    def _character_style(self, fc: int, char_index: int) -> DocCharStyle:
        """按样式链、CHPX、piece Prm 顺序解析字符样式。"""

        paragraph_run = self.formatting.paragraph_at(fc)
        paragraph_style_id = paragraph_run.style_id if paragraph_run is not None else 0
        character_run = self.formatting.character_at(fc)
        grpprl = character_run.grpprl if character_run is not None else b""
        style_id = chpx_style_id(grpprl)
        if style_id is None:
            style_id = paragraph_style_id
        base = self.stylesheet.get(style_id).character
        result = apply_character_sprms(grpprl, base, base, budget=self.budget)
        prm = self._piece_prm(char_index)
        if prm:
            result = apply_character_sprms(prm, result, base, budget=self.budget)
        return result

    def _paragraph_properties(self, fc: int, char_index: int) -> tuple[int, PapDelta]:
        """按样式链、PAPX、piece Prm 顺序解析段落属性。"""

        paragraph_run = self.formatting.paragraph_at(fc)
        style_id = paragraph_run.style_id if paragraph_run is not None else 0
        style = self.stylesheet.get(style_id)
        delta = style.paragraph
        if paragraph_run is not None:
            delta = delta.merge(paragraph_run.delta)
        prm = self._piece_prm(char_index)
        if prm:
            delta = apply_paragraph_sprms(prm, b"", delta, budget=self.budget)
        return style_id, delta

    @staticmethod
    def _append_run(target: list[DocTextRun], run: DocTextRun) -> None:
        """合并相邻同样式同链接 run。"""

        if not run.text:
            return
        if target and target[-1].style == run.style and target[-1].hyperlink == run.hyperlink:
            previous = target[-1]
            target[-1] = DocTextRun(previous.text + run.text, previous.style, previous.hyperlink)
        else:
            target.append(run)

    def _push_visible_run(self, visible: list[DocTextRun], run: DocTextRun) -> None:
        """把一个结果 run 送入当前字段或段落。"""

        if not self._fields:
            self._append_run(visible, run)
            return
        frame = self._fields[-1]
        if not frame.in_result:
            frame.instruction += run.text
        elif frame.transparent:
            self._append_run(visible, run)
        else:
            self._append_run(frame.runs, run)

    def _field_begin(self) -> None:
        """打开一个嵌套字段。"""

        self._fields.append(_FieldFrame())

    def _field_separator(self, paragraph_keywords: set[str]) -> None:
        """结束字段指令并识别跨段 TOC。"""

        if not self._fields:
            return
        frame = self._fields[-1]
        frame.in_result = True
        keyword = field_keyword(frame.instruction)
        if keyword:
            paragraph_keywords.add(keyword)
        frame.transparent = is_toc_field(frame.instruction)

    def _field_end(self, visible: list[DocTextRun], paragraph_keywords: set[str]) -> None:
        """关闭字段，把缓存结果绑定链接后放回父上下文。"""

        if not self._fields:
            return
        frame = self._fields.pop()
        keyword = field_keyword(frame.instruction)
        if keyword:
            paragraph_keywords.add(keyword)
        if frame.transparent:
            return
        for run in apply_field_result(frame.instruction, frame.runs):
            self._push_visible_run(visible, run)

    def _paragraph_anchor(self, cp_start: int, cp_end: int) -> str | None:
        """返回段落范围内优先级最高的书签起点。"""

        candidates = [
            name
            for cp, names in self.bookmarks.items()
            if cp_start <= cp < cp_end
            for name in names
        ]
        if not candidates:
            return None
        return next((name for name in candidates if name.startswith("_Toc")), candidates[0])

    def _finish_paragraph(
        self,
        *,
        cp_start: int,
        cp_end: int,
        char_index: int,
        fc: int,
        terminator: str,
        runs: list[DocTextRun],
        images: list,
        keywords: set[str],
    ) -> DocParagraph | None:
        """把段落终止字符上的 PAPX 解析为完整语义段落。"""

        style_id, pap = self._paragraph_properties(fc, char_index)
        style = self.stylesheet.get(style_id)
        cell_mark = terminator == "\x07" or bool(pap.inner_cell)
        row_mark = bool(pap.row_mark) or bool(pap.inner_row)
        in_table = bool(pap.in_table) or cell_mark or row_mark
        visible = any(run.text for run in runs) or bool(images)
        if not visible and not in_table:
            return None
        list_info = None
        if pap.ilfo is not None:
            list_info = self.lists.paragraph_info(pap.ilfo, pap.ilvl or 0)
        outline = pap.outline_level if isinstance(pap.outline_level, int) and pap.outline_level > 0 else None
        toc_level = style.toc_level
        toc_active = toc_level is not None or any(frame.transparent for frame in self._fields)
        return DocParagraph(
            cp_start=cp_start,
            cp_end=cp_end,
            runs=runs,
            images=images,
            style_name=style.name,
            heading_level=style.heading_level or outline,
            is_title=style.is_title,
            is_toc=toc_active,
            toc_level=toc_level,
            is_caption=any(keyword == "SEQ" for keyword in keywords),
            is_code=style.is_code,
            anchor=self._paragraph_anchor(cp_start, cp_end),
            list_info=list_info,
            in_table=in_table,
            table_depth=max(pap.table_depth or (1 if in_table else 0), 0),
            cell_mark=cell_mark,
            row_mark=row_mark,
            table_format=pap.table,
        )

    def paragraphs(
        self,
        cp_start: int,
        cp_end: int,
        *,
        note_refs: dict[int, str] | None = None,
    ) -> list[DocParagraph]:
        """解析一个 story CP 范围内的全部段落。"""

        self._fields = []
        refs = note_refs or {}
        start_index = self.text.index_of_cp(cp_start)
        end_index = self.text.index_of_cp(cp_end)
        paragraphs: list[DocParagraph] = []
        visible: list[DocTextRun] = []
        images: list = []
        keywords: set[str] = set()
        paragraph_start = cp_start
        for char_index in range(start_index, min(end_index, len(self.text.chars))):
            char = self.text.chars[char_index]
            cp = self.text.cps[char_index]
            fc = self.text.fcs[char_index]
            if char == "\x13":
                self._field_begin()
                continue
            if char == "\x14":
                self._field_separator(keywords)
                continue
            if char == "\x15":
                self._field_end(visible, keywords)
                continue
            if cp in refs:
                self._push_visible_run(
                    visible,
                    DocTextRun(f"[{refs[cp]}]", DocCharStyle(superscript=True)),
                )
                continue
            if char in {"\r", "\x07", "\x0c", "\x0e"}:
                paragraph = self._finish_paragraph(
                    cp_start=paragraph_start,
                    cp_end=cp + 1,
                    char_index=char_index,
                    fc=fc,
                    terminator=char,
                    runs=visible,
                    images=images,
                    keywords=keywords,
                )
                if paragraph is not None:
                    paragraphs.append(paragraph)
                visible = []
                images = []
                keywords = set()
                paragraph_start = cp + 1
                continue
            if char == "\x01":
                character_run = self.formatting.character_at(fc)
                location = chpx_picture_location(character_run.grpprl) if character_run is not None else None
                if location is not None:
                    payload = inline_picture(
                        self.data_stream,
                        offset=location,
                        store=self.image_store,
                        budget=self.budget,
                    )
                    if payload is not None:
                        if all(existing is not payload for existing in images):
                            images.append(payload)
                continue
            if char == "\x0b":
                char = "\n"
            elif char == "\t":
                char = "\t"
            elif char == "\x1e":
                char = "-"
            elif ord(char) < 0x20:
                continue
            style = self._character_style(fc, char_index)
            if self._fields and not self._fields[-1].in_result:
                self._push_visible_run(visible, DocTextRun(char, style))
            elif not style.hidden and not style.deleted:
                self._push_visible_run(visible, DocTextRun(char, style))
        if visible or images:
            last_index = max(start_index, min(end_index, len(self.text.chars)) - 1)
            if last_index < len(self.text.chars):
                paragraph = self._finish_paragraph(
                    cp_start=paragraph_start,
                    cp_end=cp_end,
                    char_index=last_index,
                    fc=self.text.fcs[last_index],
                    terminator="",
                    runs=visible,
                    images=images,
                    keywords=keywords,
                )
                if paragraph is not None:
                    paragraphs.append(paragraph)
        return paragraphs


def _paragraph_visible(paragraph: DocParagraph) -> bool:
    """判断表格标记段落是否含有实际单元格内容。"""

    return any(run.text for run in paragraph.runs) or bool(paragraph.images)


def _cell_formats(row: _RawTableRow) -> tuple[list[int], list[DocTableCellFormat]]:
    """为缺失 TAP 的行构造稳定伪边界和默认 cell 属性。"""

    count = len(row.cells)
    table_format = row.table_format
    if table_format is not None and len(table_format.boundaries) >= count + 1:
        boundaries = list(table_format.boundaries[: count + 1])
        formats = list(table_format.cells[:count])
        while len(formats) < count:
            formats.append(DocTableCellFormat(right=boundaries[len(formats) + 1]))
        return boundaries, formats
    boundaries = list(range(count + 1))
    return boundaries, [DocTableCellFormat(right=index + 1) for index in range(count)]


def _materialize_table_rows(raw_rows: list[_RawTableRow], budget: DocBudget) -> list[DocTableRow]:
    """把各行独立 twip 边界统一为 rowspan/colspan HTML 网格。"""

    all_boundaries: set[int] = set()
    row_formats: list[tuple[list[int], list[DocTableCellFormat]]] = []
    for row in raw_rows:
        boundaries, formats = _cell_formats(row)
        row_formats.append((boundaries, formats))
        all_boundaries.update(boundaries)
    edges = sorted(all_boundaries)
    edge_index = {value: index for index, value in enumerate(edges)}
    vertical_origins: dict[tuple[int, int], DocTableCell] = {}
    rows: list[DocTableRow] = []
    for raw_row, (boundaries, formats) in zip(raw_rows, row_formats, strict=True):
        cells: list[DocTableCell] = []
        keys: list[tuple[int, int]] = []
        index = 0
        while index < len(raw_row.cells):
            source = raw_row.cells[index]
            cell_format = formats[index]
            left = edge_index.get(boundaries[index], index)
            right = edge_index.get(boundaries[index + 1], left + 1)
            col_span = max(right - left, 1)
            if cell_format.horizontal_continue and cells:
                cells[-1].col_span += col_span
                cells[-1].blocks.extend(source.blocks)
                previous_left, _ = keys[-1]
                keys[-1] = (previous_left, right)
                index += 1
                continue
            cell = DocTableCell(blocks=source.blocks, col_span=col_span)
            key = (left, right)
            if cell_format.vertical_continue and key in vertical_origins:
                vertical_origins[key].row_span += 1
            else:
                cells.append(cell)
                keys.append(key)
                if cell_format.vertical_first:
                    vertical_origins[key] = cell
            index += 1
            budget.charge(col_span)
        rows.append(
            DocTableRow(
                cells=cells,
                header=bool(raw_row.table_format and raw_row.table_format.header),
            )
        )
    return rows


def _parse_table(
    paragraphs: list[DocParagraph],
    start: int,
    depth: int,
    budget: DocBudget,
) -> tuple[DocTable, int]:
    """递归解析指定 table depth 的表格和嵌套表格。"""

    index = start
    cp_start = paragraphs[start].cp_start
    cp_end = cp_start
    raw_rows: list[_RawTableRow] = []
    row_cells: list[DocTableCell] = []
    current_blocks: list[DocElement] = []
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        if not paragraph.in_table or paragraph.table_depth < depth:
            break
        if paragraph.table_depth > depth:
            nested, index = _parse_table(paragraphs, index, paragraph.table_depth, budget)
            current_blocks.append(nested)
            cp_end = max(cp_end, nested.cp_end)
            continue
        cp_end = max(cp_end, paragraph.cp_end)
        if paragraph.row_mark:
            if _paragraph_visible(paragraph):
                current_blocks.append(paragraph)
            if current_blocks:
                row_cells.append(DocTableCell(blocks=current_blocks))
                current_blocks = []
            if row_cells:
                raw_rows.append(_RawTableRow(row_cells, paragraph.table_format))
                row_cells = []
        elif paragraph.cell_mark:
            if _paragraph_visible(paragraph):
                current_blocks.append(paragraph)
            row_cells.append(DocTableCell(blocks=current_blocks))
            current_blocks = []
        else:
            current_blocks.append(paragraph)
        index += 1
    if current_blocks:
        row_cells.append(DocTableCell(blocks=current_blocks))
    if row_cells:
        raw_rows.append(_RawTableRow(row_cells, None))
    rows = _materialize_table_rows(raw_rows, budget)
    return DocTable(cp_start=cp_start, cp_end=cp_end, rows=rows), index


def _assemble_main_elements(
    paragraphs: list[DocParagraph],
    floating: list[DocImage],
    budget: DocBudget,
) -> list[DocElement]:
    """把连续 table paragraph 收敛成表格，并按 CP 合入 floating 图片。"""

    elements: list[DocElement] = []
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        if paragraph.in_table and paragraph.table_depth > 0:
            # DOC 可直接以 depth=2 的 inner-cell 开始；从 depth=1 建立隐式外层格。
            table, index = _parse_table(paragraphs, index, 1, budget)
            if table.rows:
                elements.append(table)
            continue
        elements.append(paragraph)
        index += 1
    elements.extend(floating)
    elements.sort(
        key=lambda element: (
            element.cp if isinstance(element, DocImage) else element.cp_start,
            1 if isinstance(element, DocImage) else 0,
        )
    )
    return elements


def _section_ranges(
    table_stream: bytes,
    fib: FileInformationBlock,
    budget: DocBudget,
) -> list[tuple[int, int]]:
    """从 PlcfSed 恢复 section CP 范围，损坏时回退单 section。"""

    pair = fib.pair(FCLCB_SECTION)
    payload = bounded_slice(table_stream, pair.fc, pair.lcb)
    if payload is None:
        return [(0, fib.ccp_text)]
    cps, items = parse_plc(payload, item_size=12, budget=budget)
    if not items or len(cps) != len(items) + 1:
        return [(0, fib.ccp_text)]
    ranges: list[tuple[int, int]] = []
    for start, end in zip(cps, cps[1:]):
        if start > end or start > fib.ccp_text:
            logger.warning("DOC PlcfSed CP values are invalid; using one recovery section")
            return [(0, fib.ccp_text)]
        ranges.append((start, min(end, fib.ccp_text)))
    return ranges or [(0, fib.ccp_text)]


def _distribute_sections(
    ranges: list[tuple[int, int]],
    elements: list[DocElement],
) -> list[DocSection]:
    """按元素起始 CP 将主文档内容绑定到 section。"""

    sections = [DocSection(start, end) for start, end in ranges]
    for element in elements:
        cp = element.cp if isinstance(element, DocImage) else element.cp_start
        target = sections[-1]
        for section in sections:
            if section.cp_start <= cp < section.cp_end or (
                section is sections[-1] and cp == section.cp_end
            ):
                target = section
                break
        target.elements.append(element)
    return sections


def _header_story_ranges(
    table_stream: bytes,
    fib: FileInformationBlock,
    budget: DocBudget,
) -> list[tuple[int, int]]:
    """返回 PlcfHdd 中相对于 header story 的范围。"""

    pair = fib.pair(FCLCB_HEADER)
    payload = bounded_slice(table_stream, pair.fc, pair.lcb)
    if payload is None or len(payload) < 8 or len(payload) % 4:
        return []
    budget.charge(len(payload) // 4)
    cps = [int.from_bytes(payload[index : index + 4], "little") for index in range(0, len(payload), 4)]
    return [(start, end) for start, end in zip(cps, cps[1:]) if start <= end]


def _attach_headers(
    sections: list[DocSection],
    assembler: _Assembler,
    table_stream: bytes,
    fib: FileInformationBlock,
    budget: DocBudget,
) -> None:
    """把每节六种 header/footer story 的非空段落绑定到 section。"""

    ranges = _header_story_ranges(table_stream, fib, budget)
    base = fib.story_bases["header"]
    for section_index, section in enumerate(sections):
        group = 6 + section_index * 6
        if group + 5 >= len(ranges):
            break
        for story_index in (0, 1, 4):
            start, end = ranges[group + story_index]
            section.headers.extend(assembler.paragraphs(base + start, base + end))
        for story_index in (2, 3, 5):
            start, end = ranges[group + story_index]
            section.footers.extend(assembler.paragraphs(base + start, base + end))
    inherited_header: DocParagraph | None = None
    for section in sections:
        if section.headers:
            inherited_header = section.headers[0]
        elif inherited_header is not None:
            section.headers.append(inherited_header)


def _note_ranges(
    table_stream: bytes,
    *,
    ref_offset: int,
    ref_size: int,
    text_offset: int,
    text_size: int,
    story_base: int,
    prefix: str,
    budget: DocBudget,
) -> tuple[dict[int, str], list[tuple[int, int, int, str]]]:
    """解析脚注或尾注 reference CP 与正文 story 范围。"""

    ref_payload = bounded_slice(table_stream, ref_offset, ref_size)
    text_payload = bounded_slice(table_stream, text_offset, text_size)
    if ref_payload is None or text_payload is None:
        return {}, []
    ref_cps, ref_items = parse_plc(ref_payload, item_size=2, budget=budget)
    text_cps, _ = parse_plc(text_payload, item_size=0, budget=budget)
    references: dict[int, str] = {}
    ranges: list[tuple[int, int, int, str]] = []
    for index, _item in enumerate(ref_items):
        if index >= len(ref_cps) or index + 1 >= len(text_cps):
            break
        label = str(index + 1)
        references[ref_cps[index]] = label
        ranges.append(
            (
                ref_cps[index],
                story_base + text_cps[index],
                story_base + text_cps[index + 1],
                f"{prefix}{label}",
            )
        )
    return references, ranges


def _prepend_note_label(paragraphs: list[DocParagraph], label: str) -> None:
    """把脚注或尾注编号写入首个可见段落。"""

    if not paragraphs:
        return
    paragraphs[0].runs.insert(0, DocTextRun(f"[{label.removeprefix('fn').removeprefix('en')}] "))


def _attach_notes(
    sections: list[DocSection],
    assembler: _Assembler,
    footnote_ranges: list[tuple[int, int, int, str]],
    endnote_ranges: list[tuple[int, int, int, str]],
) -> None:
    """脚注按引用 section、尾注按最后 section 追加。"""

    for reference_cp, start, end, label in footnote_ranges:
        paragraphs = assembler.paragraphs(start, end)
        _prepend_note_label(paragraphs, label)
        target = sections[-1]
        for section in sections:
            if section.cp_start <= reference_cp < section.cp_end:
                target = section
                break
        target.footnotes.extend(paragraphs)
    for _reference_cp, start, end, label in endnote_ranges:
        paragraphs = assembler.paragraphs(start, end)
        _prepend_note_label(paragraphs, label)
        sections[-1].footnotes.extend(paragraphs)


def parse_doc_document(
    word_document: bytes,
    table_stream: bytes,
    data_stream: bytes,
    fib: FileInformationBlock,
) -> DocDocument:
    """解析三个核心 streams 并返回逐 section 语义文档。"""

    budget = DocBudget()
    clx = fib.pair(FCLCB_CLX)
    if clx.lcb:
        pieces = parse_clx(table_stream, offset=clx.fc, size=clx.lcb, budget=budget)
    elif not fib.base.complex:
        pieces = legacy_single_piece(
            fc_min=fib.base.fc_min,
            fc_mac=fib.base.fc_mac,
            ccp_text=fib.ccp_text,
        )
    else:
        raise LegacyOfficeMalformedError("complex DOC is missing its CLX piece table")
    if not pieces and fib.total_story_cp:
        raise LegacyOfficeMalformedError("DOC contains no recoverable text pieces")
    lid = fib.base.lid
    if fib.base.far_east:
        lid_fe = int.from_bytes(word_document[0x3C:0x3E], "little") if len(word_document) >= 0x3E else 0
        if lid_fe:
            lid = lid_fe
        elif len(fib.rgw) > 13 and fib.rgw[13]:
            lid = fib.rgw[13]
    text = extract_text(
        word_document,
        pieces,
        total_cp=fib.total_story_cp,
        codec=codec_for_lid(lid),
        budget=budget,
    )
    stsh = fib.pair(FCLCB_STSHF)
    stylesheet = parse_stylesheet(
        table_stream,
        offset=stsh.fc,
        size=stsh.lcb,
        budget=budget,
    )
    chpx = fib.pair(FCLCB_BTE_CHPX)
    papx = fib.pair(FCLCB_BTE_PAPX)
    formatting = parse_formatting_runs(
        word_document,
        table_stream,
        data_stream,
        chpx_offset=chpx.fc,
        chpx_size=chpx.lcb,
        papx_offset=papx.fc,
        papx_size=papx.lcb,
        budget=budget,
    )
    lists_pair = fib.pair(FCLCB_LISTS)
    overrides_pair = fib.pair(FCLCB_LIST_OVERRIDES)
    lists = parse_list_tables(
        table_stream,
        list_offset=lists_pair.fc,
        list_size=lists_pair.lcb,
        override_offset=overrides_pair.fc,
        override_size=overrides_pair.lcb,
        budget=budget,
    )
    names = fib.pair(FCLCB_BOOKMARK_NAMES)
    starts = fib.pair(FCLCB_BOOKMARK_START)
    ends = fib.pair(FCLCB_BOOKMARK_END)
    bookmarks = parse_bookmarks(
        table_stream,
        names_offset=names.fc,
        names_size=names.lcb,
        starts_offset=starts.fc,
        starts_size=starts.lcb,
        ends_offset=ends.fc,
        ends_size=ends.lcb,
        budget=budget,
    )
    foot_ref = fib.pair(FCLCB_FOOTNOTE_REF)
    foot_text = fib.pair(FCLCB_FOOTNOTE_TEXT)
    foot_refs, foot_ranges = _note_ranges(
        table_stream,
        ref_offset=foot_ref.fc,
        ref_size=foot_ref.lcb,
        text_offset=foot_text.fc,
        text_size=foot_text.lcb,
        story_base=fib.story_bases["footnote"],
        prefix="fn",
        budget=budget,
    )
    end_ref = fib.pair(FCLCB_ENDNOTE_REF)
    end_text = fib.pair(FCLCB_ENDNOTE_TEXT)
    end_refs, end_ranges = _note_ranges(
        table_stream,
        ref_offset=end_ref.fc,
        ref_size=end_ref.lcb,
        text_offset=end_text.fc,
        text_size=end_text.lcb,
        story_base=fib.story_bases["endnote"],
        prefix="en",
        budget=budget,
    )
    note_refs = {**foot_refs, **end_refs}
    store = ImageStore()
    assembler = _Assembler(
        text=text,
        pieces=pieces,
        formatting=formatting,
        stylesheet=stylesheet,
        lists=lists,
        bookmarks=bookmarks,
        data_stream=data_stream,
        image_store=store,
        budget=budget,
    )
    main_paragraphs = assembler.paragraphs(0, fib.ccp_text, note_refs=note_refs)
    shape_pair = fib.pair(FCLCB_SHAPE_MAIN)
    drawing_pair = fib.pair(FCLCB_DGG_INFO)
    floating = floating_pictures(
        table_stream,
        word_document=word_document,
        shape_plc_offset=shape_pair.fc,
        shape_plc_size=shape_pair.lcb,
        drawing_offset=drawing_pair.fc,
        drawing_size=drawing_pair.lcb,
        store=store,
        budget=budget,
    )
    elements = _assemble_main_elements(main_paragraphs, floating, budget)
    ranges = _section_ranges(table_stream, fib, budget)
    sections = _distribute_sections(ranges, elements)
    _attach_headers(sections, assembler, table_stream, fib, budget)
    _attach_notes(sections, assembler, foot_ranges, end_ranges)
    return DocDocument(sections=sections)
