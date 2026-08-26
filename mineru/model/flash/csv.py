# Copyright (c) Opendatalab. All rights reserved.
"""将分隔符文本 CSV 转换为 MinerU 单页表格 model-list。"""

from __future__ import annotations

import codecs
import csv as csv_module
import html
import re
from collections import Counter
from io import StringIO
from typing import Any, BinaryIO, Final, Literal, TypeAlias

from ftfy.badness import badness

from ...types import BlockType

MAX_CSV_BYTES: Final = 200 * 1024 * 1024
MAX_CSV_ROWS: Final = 1_048_576
MAX_CSV_COLUMNS: Final = 16_384
MAX_CSV_GRID_SLOTS: Final = 4_000_000

_DELIMITER_CANDIDATES: Final = (",", ";", "\t", "|")
_DELIMITER_SAMPLE_RECORDS: Final = 20
_HEADER_SAMPLE_ROWS: Final = 50
_HEADER_KIND_DOMINANCE_NUM: Final = 9
_HEADER_KIND_DOMINANCE_DEN: Final = 10
_MAX_HEADER_LABEL_CHARS: Final = 64
_SEP_DIRECTIVE_RE = re.compile(r"\Asep=(?P<delimiter>[,;\t|])(?:\r\n|\n|\r|$)", re.IGNORECASE)
_DISALLOWED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
_DATE_RE = re.compile(
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?$"
)
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?$")

CsvValueKind: TypeAlias = Literal["number", "boolean", "date", "text"]


def _text_quality_score(text: str) -> int:
    """计算候选解码文本的异常字符分数，分数越低越可信。"""
    control_penalty = len(_DISALLOWED_CONTROL_RE.findall(text)) * 10
    return badness(text) + control_penalty


def _decode_csv_bytes(file_bytes: bytes) -> str:
    """按 BOM、UTF-8、GB18030、Windows-1252 的固定顺序严格解码 CSV。"""
    if file_bytes.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        raise ValueError("Unsupported CSV encoding: UTF-32")
    if file_bytes.startswith(codecs.BOM_UTF8):
        return file_bytes.decode("utf-8-sig", errors="strict")
    if file_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return file_bytes.decode("utf-16", errors="strict")
    try:
        return file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        pass

    candidates: dict[str, str] = {}
    for encoding in ("gb18030", "cp1252"):
        try:
            candidates[encoding] = file_bytes.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    if not candidates:
        raise ValueError("Unsupported CSV encoding; expected UTF-8, UTF-16, GB18030, or Windows-1252")
    return min(
        candidates.items(),
        key=lambda item: (
            _text_quality_score(item[1]),
            0 if item[0] == "cp1252" else 1,
        ),
    )[1]


def _extract_sep_directive(text: str) -> tuple[str, str | None]:
    """提取 Excel 风格的 sep 指令，并从后续 CSV 数据中移除该物理行。"""
    match = _SEP_DIRECTIVE_RE.match(text)
    if match is None:
        return text, None
    return text[match.end() :], match.group("delimiter")


def _sample_record_widths(text: str, delimiter: str) -> list[int]:
    """用候选分隔符读取有限个完整逻辑记录并返回每条记录的字段数。"""
    reader = csv_module.reader(
        StringIO(text, newline=""),
        delimiter=delimiter,
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
        strict=True,
    )
    widths: list[int] = []
    try:
        for record in reader:
            widths.append(max(1, len(record)))
            if len(widths) >= _DELIMITER_SAMPLE_RECORDS:
                break
    except csv_module.Error:
        return []
    return widths


def _sniff_delimiter(text: str) -> str:
    """按逻辑记录列宽的一致性选择分隔符，并在完全平局时优先逗号。"""
    best_delimiter = ","
    best_score = (0, 0, 0)
    for preference, delimiter in enumerate(_DELIMITER_CANDIDATES):
        widths = _sample_record_widths(text, delimiter)
        if not widths:
            continue
        width_counts = Counter(widths)
        modal_width, frequency = max(width_counts.items(), key=lambda item: (item[1], item[0]))
        if modal_width < 2:
            continue
        score = (frequency, modal_width, -preference)
        if score > best_score:
            best_delimiter = delimiter
            best_score = score
    return best_delimiter


def _read_csv_rows(text: str, delimiter: str) -> list[list[str]]:
    """严格读取全部 CSV 记录，同时执行行数、列数与网格规模限制。"""
    reader = csv_module.reader(
        StringIO(text, newline=""),
        delimiter=delimiter,
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
        strict=True,
    )
    rows: list[list[str]] = []
    max_columns = 0
    try:
        for record in reader:
            row = list(record) or [""]
            rows.append(row)
            if len(rows) > MAX_CSV_ROWS:
                raise ValueError(f"CSV exceeds max_rows={MAX_CSV_ROWS}")
            max_columns = max(max_columns, len(row))
            if max_columns > MAX_CSV_COLUMNS:
                raise ValueError(f"CSV exceeds max_columns={MAX_CSV_COLUMNS}")
    except csv_module.Error as exc:
        raise ValueError(f"Malformed CSV near physical line {reader.line_num}: {exc}") from exc

    grid_slots = len(rows) * max_columns
    if grid_slots > MAX_CSV_GRID_SLOTS:
        raise ValueError(f"CSV exceeds max_grid_slots={MAX_CSV_GRID_SLOTS}")
    return rows


def _classify_value(value: str) -> CsvValueKind | None:
    """把非空字段粗分为数字、布尔、日期或文本，供表头投票使用。"""
    normalized = value.strip()
    if not normalized:
        return None
    numeric = normalized.removesuffix("%")
    compact_numeric = "".join(char for char in numeric if char not in {",", " ", "_", "\u00a0"})
    if any(char.isascii() and char.isdigit() for char in compact_numeric):
        try:
            float(compact_numeric)
        except ValueError:
            pass
        else:
            return "number"
    if normalized.casefold() in {"true", "false", "yes", "no"}:
        return "boolean"
    if _DATE_RE.fullmatch(normalized) or _TIME_RE.fullmatch(normalized):
        return "date"
    return "text"


def _dominant_kind(values: list[str]) -> CsvValueKind | None:
    """返回至少覆盖九成非空主体值的字段类型，没有优势类型时返回空。"""
    kinds = [kind for value in values if (kind := _classify_value(value)) is not None]
    if not kinds:
        return None
    counts = Counter(kinds)
    for kind in ("number", "boolean", "date", "text"):
        if counts[kind] * _HEADER_KIND_DOMINANCE_DEN >= len(kinds) * _HEADER_KIND_DOMINANCE_NUM:
            return kind
    return None


def _fold_header_value(value: str) -> str:
    """生成忽略首尾空白和大小写的表头比较值。"""
    return value.strip().casefold()


def _modal_row_width(rows: list[list[str]]) -> int:
    """返回行宽众数，频次相同时选择更宽的记录。"""
    if not rows:
        return 0
    counts = Counter(len(row) for row in rows)
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _infer_header_row(rows: list[list[str]]) -> bool:
    """根据首行标签形态和主体列类型保守判断 CSV 是否具有一行表头。"""
    if len(rows) < 2:
        return False
    body = rows[1 : _HEADER_SAMPLE_ROWS + 1]
    if len(rows[0]) != _modal_row_width(body):
        return False

    header = rows[0]
    seen_labels: set[str] = set()
    for column, value in enumerate(header):
        folded = _fold_header_value(value)
        if not folded:
            if column == 0:
                continue
            return False
        if "\n" in value or "\r" in value or len(value) > _MAX_HEADER_LABEL_CHARS:
            return False
        if folded in seen_labels:
            return False
        seen_labels.add(folded)

    header_votes = 0
    data_votes = 0
    for column, label in enumerate(header):
        values = [row[column].strip() for row in body if column < len(row) and row[column].strip()]
        if not values:
            continue
        label_kind = _classify_value(label)
        dominant_kind = _dominant_kind(values)
        if dominant_kind is not None and dominant_kind != "text":
            if label_kind == "text" or (column == 0 and not label.strip()):
                header_votes += 1
            else:
                data_votes += 1
            continue
        folded_label = _fold_header_value(label)
        if folded_label and any(_fold_header_value(value) == folded_label for value in values):
            data_votes += 1

    if header_votes or data_votes:
        return header_votes > data_votes
    return True


def _normalize_row_widths(rows: list[list[str]]) -> list[list[str]]:
    """在不改变已有字段内容的前提下，把短记录补齐到最大列宽。"""
    if not rows:
        return []
    max_columns = max(len(row) for row in rows)
    return [row + [""] * (max_columns - len(row)) for row in rows]


def _render_field_html(value: str) -> str:
    """转义一个 CSV 字段，规范换行并替换 HTML 不允许的控制字符。"""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _DISALLOWED_CONTROL_RE.sub("\ufffd", normalized)
    return html.escape(normalized, quote=True).replace("\n", "<br>")


def _rows_to_html(rows: list[list[str]], *, has_header: bool) -> str:
    """把规则矩形 CSV 网格转换为现有 TableBlock 可消费的安全 HTML。"""
    lines = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if has_header and row_index == 0 else "td"
        lines.append("  <tr>")
        for value in row:
            lines.append(f"    <{tag}>{_render_field_html(value)}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def convert_csv(file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
    """读取 CSV 二进制流并返回单逻辑页的表格 model-list。"""
    file_bytes = file_binary.read(MAX_CSV_BYTES + 1)
    if len(file_bytes) > MAX_CSV_BYTES:
        raise ValueError(f"CSV exceeds max_bytes={MAX_CSV_BYTES}")

    text = _decode_csv_bytes(file_bytes)
    text, declared_delimiter = _extract_sep_directive(text)
    delimiter = declared_delimiter or _sniff_delimiter(text)
    rows = _read_csv_rows(text, delimiter)
    if not rows:
        return [[]]
    has_header = _infer_header_row(rows)
    normalized_rows = _normalize_row_widths(rows)
    table_html = _rows_to_html(normalized_rows, has_header=has_header)
    return [[{"type": BlockType.TABLE, "content": table_html}]]


__all__ = ["convert_csv"]
