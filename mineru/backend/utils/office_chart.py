# Copyright (c) Opendatalab. All rights reserved.
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from html import escape
from io import BytesIO
from typing import Any, Final

import pandas as pd
from lxml import etree
from openpyxl import load_workbook
from openpyxl.utils.cell import range_to_tuple


_CHART_NS: Final = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_DRAWING_NS: Final = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS: Final = {"c": _CHART_NS, "a": _DRAWING_NS}
_EXCEL_EPOCH_1900: Final = datetime(1899, 12, 30)
_EXCEL_EPOCH_1904: Final = datetime(1904, 1, 1)
_PLOT_TAGS: Final = (
    "areaChart",
    "area3DChart",
    "barChart",
    "bar3DChart",
    "bubbleChart",
    "doughnutChart",
    "lineChart",
    "line3DChart",
    "ofPieChart",
    "pieChart",
    "pie3DChart",
    "radarChart",
    "scatterChart",
    "stockChart",
    "surfaceChart",
    "surface3DChart",
)


@dataclass
class SeriesSpec:
    name_formula: str | None = None
    literal_name: str | None = None
    cat_formula: str | None = None
    x_formula: str | None = None
    val_formula: str | None = None
    y_formula: str | None = None
    bubble_size_formula: str | None = None
    cached_categories: list[str] = field(default_factory=list)
    cached_x_values: list[str] = field(default_factory=list)
    cached_values: list[str] = field(default_factory=list)
    cached_bubble_sizes: list[str] = field(default_factory=list)


@dataclass
class ChartSpec:
    chart_type: str
    plot_kind: str
    title: str = ""
    category_axis_title: str = ""
    value_axis_title: str = ""
    x_axis_title: str = ""
    has_date_axis: bool = False
    date_1904: bool = False
    series: list[SeriesSpec] = field(default_factory=list)


def minify_html(html: str) -> str:
    """Strip formatting whitespace from HTML while preserving content."""
    if not html:
        return html

    html = re.sub(r'>\s+<', '><', html)
    html = re.sub(r'\n\s*', '', html)
    return html


def html_table_from_excel_bytes(excel_bytes: bytes) -> str:
    """Convert the first non-empty worksheet in an embedded workbook to HTML."""
    if not excel_bytes:
        return ""

    worksheets = pd.read_excel(BytesIO(excel_bytes), sheet_name=None)
    for dataframe in worksheets.values():
        if dataframe is None:
            continue
        if dataframe.empty and len(dataframe.columns) == 0:
            continue
        return minify_html(dataframe.to_html(index=False, header=True))

    return ""


def extract_chart_html_from_ooxml(chart_xml: bytes, workbook_bytes: bytes | None) -> str:
    spec = parse_chart_spec_from_ooxml(chart_xml)
    if spec is None or not spec.series:
        return ""

    if workbook_bytes:
        chart_html = render_chart_html_from_workbook(spec, workbook_bytes)
        if chart_html:
            return chart_html

    return render_chart_html_from_cache(spec)


def parse_chart_spec_from_ooxml(chart_xml: bytes) -> ChartSpec | None:
    try:
        root = etree.fromstring(chart_xml)
    except (etree.XMLSyntaxError, TypeError, ValueError):
        return None

    plot_area = root.find(".//c:plotArea", namespaces=_NS)
    if plot_area is None:
        return None

    plot = None
    chart_type = ""
    for tag_name in _PLOT_TAGS:
        plot = plot_area.find(f"c:{tag_name}", namespaces=_NS)
        if plot is not None:
            chart_type = tag_name
            break

    if plot is None:
        return None

    has_date_axis = plot_area.find("c:dateAx", namespaces=_NS) is not None
    if chart_type == "scatterChart":
        plot_kind = "scatter"
    elif chart_type == "bubbleChart":
        plot_kind = "bubble"
    elif has_date_axis:
        plot_kind = "date"
    else:
        plot_kind = "category"

    category_axis_title = ""
    axis = plot_area.find("c:dateAx", namespaces=_NS)
    if axis is None:
        axis = plot_area.find("c:catAx", namespaces=_NS)
    if axis is not None:
        category_axis_title = _extract_title_text(axis.find("c:title", namespaces=_NS))

    x_axis_title = ""
    value_axis_title = ""
    if plot_kind in {"scatter", "bubble"}:
        for axis in plot_area.findall("c:valAx", namespaces=_NS):
            axis_pos = axis.find("c:axPos", namespaces=_NS)
            axis_position = axis_pos.get("val") if axis_pos is not None else ""
            title = _extract_title_text(axis.find("c:title", namespaces=_NS))
            if axis_position == "b" and not x_axis_title:
                x_axis_title = title
            elif axis_position == "l" and not value_axis_title:
                value_axis_title = title
        if not x_axis_title:
            x_axis_title = category_axis_title
    else:
        axis = plot_area.find("c:valAx", namespaces=_NS)
        if axis is not None:
            value_axis_title = _extract_title_text(axis.find("c:title", namespaces=_NS))

    series_specs = []
    for series_element in plot.findall("c:ser", namespaces=_NS):
        series_specs.append(
            SeriesSpec(
                name_formula=_extract_tx_formula(series_element.find("c:tx", namespaces=_NS)),
                literal_name=_extract_tx_text(series_element.find("c:tx", namespaces=_NS)),
                cat_formula=_extract_reference_formula(series_element.find("c:cat", namespaces=_NS)),
                x_formula=_extract_reference_formula(series_element.find("c:xVal", namespaces=_NS)),
                val_formula=_extract_reference_formula(series_element.find("c:val", namespaces=_NS)),
                y_formula=_extract_reference_formula(series_element.find("c:yVal", namespaces=_NS)),
                bubble_size_formula=_extract_reference_formula(
                    series_element.find("c:bubbleSize", namespaces=_NS)
                ),
                cached_categories=_extract_reference_cache(
                    series_element.find("c:cat", namespaces=_NS),
                    date_hint=has_date_axis,
                    date_1904=_chart_uses_date_1904(root),
                ),
                cached_x_values=_extract_reference_cache(
                    series_element.find("c:xVal", namespaces=_NS)
                ),
                cached_values=_extract_reference_cache(
                    _first_non_none(
                        series_element.find("c:val", namespaces=_NS),
                        series_element.find("c:yVal", namespaces=_NS),
                    )
                ),
                cached_bubble_sizes=_extract_reference_cache(
                    series_element.find("c:bubbleSize", namespaces=_NS)
                ),
            )
        )

    return ChartSpec(
        chart_type=chart_type,
        plot_kind=plot_kind,
        title=_extract_title_text(root.find(".//c:chart/c:title", namespaces=_NS)),
        category_axis_title=category_axis_title,
        value_axis_title=value_axis_title,
        x_axis_title=x_axis_title,
        has_date_axis=has_date_axis,
        date_1904=_chart_uses_date_1904(root),
        series=series_specs,
    )


def render_chart_html_from_workbook(spec: ChartSpec, workbook_bytes: bytes) -> str:
    try:
        workbook = load_workbook(
            filename=BytesIO(workbook_bytes),
            data_only=True,
            read_only=True,
        )
    except Exception:
        return ""

    try:
        if spec.plot_kind in {"category", "date"}:
            return _render_category_like_chart_from_workbook(spec, workbook)
        if spec.plot_kind == "scatter":
            return _render_scatter_like_chart_from_workbook(spec, workbook)
        if spec.plot_kind == "bubble":
            return _render_bubble_chart_from_workbook(spec, workbook)
        return ""
    finally:
        workbook.close()


def render_chart_html_from_cache(spec: ChartSpec) -> str:
    if spec.plot_kind in {"category", "date"}:
        categories = []
        for series in spec.series:
            if series.cached_categories:
                categories = series.cached_categories
                break

        series_names = []
        series_values = []
        for idx, series in enumerate(spec.series, start=1):
            series_names.append(_resolve_series_name(series, idx))
            series_values.append(series.cached_values)

        row_count = max(
            len(categories),
            max((len(values) for values in series_values), default=0),
        )
        if not series_names or row_count == 0:
            return ""

        headers = [spec.category_axis_title or ""] + series_names
        columns = [categories] + series_values
        return _render_html_table(headers, columns, row_count)

    if spec.plot_kind == "scatter":
        return _render_scatter_like_chart_from_cache(spec)

    if spec.plot_kind == "bubble":
        return _render_bubble_chart_from_cache(spec)

    return ""


def _render_category_like_chart_from_workbook(spec: ChartSpec, workbook) -> str:
    categories = []
    data_sheet_name = None

    for series in spec.series:
        if not series.cat_formula:
            continue
        read_result = _read_formula_vector(workbook, series.cat_formula)
        if read_result is None:
            return ""
        sheet_name, values = read_result
        data_sheet_name = sheet_name
        categories = values
        break

    series_names = []
    series_values = []
    for idx, series in enumerate(spec.series, start=1):
        if not series.val_formula:
            return ""
        read_result = _read_formula_vector(
            workbook,
            series.val_formula,
            expected_sheet=data_sheet_name,
        )
        if read_result is None:
            return ""
        sheet_name, values = read_result
        data_sheet_name = data_sheet_name or sheet_name
        series_names.append(_resolve_series_name(series, idx, workbook))
        series_values.append(values)

    row_count = max(
        len(categories),
        max((len(values) for values in series_values), default=0),
    )
    if not series_names or row_count == 0:
        return ""

    headers = [spec.category_axis_title or ""] + series_names
    columns = [
        _stringify_series_values(
            categories,
            date_hint=spec.has_date_axis,
            date_1904=spec.date_1904,
        )
    ]
    columns.extend(_stringify_series_values(values) for values in series_values)
    return _render_html_table(headers, columns, row_count)


def _render_scatter_like_chart_from_workbook(spec: ChartSpec, workbook) -> str:
    x_values, series_names, series_y_values = _read_scatter_axes_from_workbook(
        spec,
        workbook,
    )
    if x_values is None or not series_names:
        return ""

    row_count = max(
        len(x_values),
        max((len(values) for values in series_y_values), default=0),
    )
    if row_count == 0:
        return ""

    headers = [spec.x_axis_title or ""] + series_names
    columns = [_stringify_series_values(x_values)]
    columns.extend(_stringify_series_values(values) for values in series_y_values)
    return _render_html_table(headers, columns, row_count)


def _render_bubble_chart_from_workbook(spec: ChartSpec, workbook) -> str:
    x_values, series_names, series_y_values, series_sizes = _read_bubble_axes_from_workbook(
        spec,
        workbook,
    )
    if x_values is None or not series_names:
        return ""

    row_count = max(
        len(x_values),
        max((len(values) for values in series_y_values), default=0),
        max((len(values) for values in series_sizes), default=0),
    )
    if row_count == 0:
        return ""

    headers = [spec.x_axis_title or ""]
    columns = [_stringify_series_values(x_values)]
    for name, y_values, bubble_sizes in zip(series_names, series_y_values, series_sizes):
        headers.extend((name, f"{name} size"))
        columns.append(_stringify_series_values(y_values))
        columns.append(_stringify_series_values(bubble_sizes))

    return _render_html_table(headers, columns, row_count)


def _render_scatter_like_chart_from_cache(spec: ChartSpec) -> str:
    x_values = _get_shared_axis_values(
        [series.cached_x_values for series in spec.series if series.cached_x_values]
    )
    if x_values is None:
        return ""

    series_names = []
    series_y_values = []
    for idx, series in enumerate(spec.series, start=1):
        if not series.cached_values:
            return ""
        series_names.append(_resolve_series_name(series, idx))
        series_y_values.append(series.cached_values)

    row_count = max(
        len(x_values),
        max((len(values) for values in series_y_values), default=0),
    )
    if not series_names or row_count == 0:
        return ""

    headers = [spec.x_axis_title or ""] + series_names
    columns = [_stringify_series_values(x_values)]
    columns.extend(_stringify_series_values(values) for values in series_y_values)
    return _render_html_table(headers, columns, row_count)


def _render_bubble_chart_from_cache(spec: ChartSpec) -> str:
    x_values = _get_shared_axis_values(
        [series.cached_x_values for series in spec.series if series.cached_x_values]
    )
    if x_values is None:
        return ""

    headers = [spec.x_axis_title or ""]
    columns = [_stringify_series_values(x_values)]
    row_count = len(x_values)
    series_count = 0
    for idx, series in enumerate(spec.series, start=1):
        if not series.cached_values or not series.cached_bubble_sizes:
            return ""
        name = _resolve_series_name(series, idx)
        headers.extend((name, f"{name} size"))
        columns.append(_stringify_series_values(series.cached_values))
        columns.append(_stringify_series_values(series.cached_bubble_sizes))
        row_count = max(row_count, len(series.cached_values), len(series.cached_bubble_sizes))
        series_count += 1

    if series_count == 0 or row_count == 0:
        return ""

    return _render_html_table(headers, columns, row_count)


def _read_scatter_axes_from_workbook(spec: ChartSpec, workbook):
    x_sequences = []
    series_names = []
    series_y_values = []
    data_sheet_name = None

    for idx, series in enumerate(spec.series, start=1):
        if not series.x_formula or not series.y_formula:
            return None, [], []

        x_read = _read_formula_vector(
            workbook,
            series.x_formula,
            expected_sheet=data_sheet_name,
        )
        if x_read is None:
            return None, [], []
        sheet_name, x_values = x_read
        data_sheet_name = data_sheet_name or sheet_name
        x_sequences.append(x_values)

        y_read = _read_formula_vector(
            workbook,
            series.y_formula,
            expected_sheet=data_sheet_name,
        )
        if y_read is None:
            return None, [], []
        _, y_values = y_read
        series_names.append(_resolve_series_name(series, idx, workbook))
        series_y_values.append(y_values)

    shared_x_values = _get_shared_axis_values(x_sequences)
    return shared_x_values, series_names, series_y_values


def _read_bubble_axes_from_workbook(spec: ChartSpec, workbook):
    x_sequences = []
    series_names = []
    series_y_values = []
    series_sizes = []
    data_sheet_name = None

    for idx, series in enumerate(spec.series, start=1):
        if not series.x_formula or not series.y_formula or not series.bubble_size_formula:
            return None, [], [], []

        x_read = _read_formula_vector(
            workbook,
            series.x_formula,
            expected_sheet=data_sheet_name,
        )
        if x_read is None:
            return None, [], [], []
        sheet_name, x_values = x_read
        data_sheet_name = data_sheet_name or sheet_name
        x_sequences.append(x_values)

        y_read = _read_formula_vector(
            workbook,
            series.y_formula,
            expected_sheet=data_sheet_name,
        )
        bubble_size_read = _read_formula_vector(
            workbook,
            series.bubble_size_formula,
            expected_sheet=data_sheet_name,
        )
        if y_read is None or bubble_size_read is None:
            return None, [], [], []

        series_names.append(_resolve_series_name(series, idx, workbook))
        series_y_values.append(y_read[1])
        series_sizes.append(bubble_size_read[1])

    shared_x_values = _get_shared_axis_values(x_sequences)
    return shared_x_values, series_names, series_y_values, series_sizes


def _read_formula_vector(workbook, formula: str, expected_sheet: str | None = None):
    parsed = _parse_formula(formula)
    if parsed is None:
        return None

    sheet_name, min_col, min_row, max_col, max_row = parsed
    if expected_sheet is not None and sheet_name != expected_sheet:
        return None

    try:
        worksheet = workbook[sheet_name]
    except KeyError:
        return None

    if min_col != max_col and min_row != max_row:
        return None

    values = []
    if min_col == max_col:
        for row_idx in range(min_row, max_row + 1):
            values.append(worksheet.cell(row=row_idx, column=min_col).value)
    else:
        for col_idx in range(min_col, max_col + 1):
            values.append(worksheet.cell(row=min_row, column=col_idx).value)

    return sheet_name, values


def _read_formula_scalar(workbook, formula: str) -> str | None:
    read_result = _read_formula_vector(workbook, formula)
    if read_result is None:
        return None

    _, values = read_result
    if not values:
        return None

    value = values[0]
    if value in (None, ""):
        return None
    return _stringify_cell_value(value)


def _parse_formula(formula: str):
    formula = formula.strip()
    if not formula:
        return None
    if formula.startswith("="):
        formula = formula[1:]

    try:
        sheet_name, bounds = range_to_tuple(formula)
    except ValueError:
        return None

    if not all(isinstance(bound, int) for bound in bounds):
        return None

    min_col, min_row, max_col, max_row = bounds
    return sheet_name, min_col, min_row, max_col, max_row


def _extract_reference_formula(container) -> str | None:
    ref_element = _find_reference_element(container)
    if ref_element is None:
        return None
    formula_element = ref_element.find("c:f", namespaces=_NS)
    if formula_element is None or formula_element.text is None:
        return None
    return formula_element.text.strip()


def _extract_reference_cache(
    container,
    *,
    date_hint: bool = False,
    date_1904: bool = False,
) -> list[str]:
    ref_element = _find_reference_element(container)
    if ref_element is None:
        return []

    tag_name = etree.QName(ref_element).localname
    if tag_name == "multiLvlStrRef":
        return _extract_multilevel_string_cache(ref_element)

    cache_element = ref_element.find("c:strCache", namespaces=_NS)
    if cache_element is None:
        cache_element = ref_element.find("c:numCache", namespaces=_NS)
    if cache_element is None:
        return []

    return _extract_cache_points(
        cache_element,
        date_hint=date_hint,
        date_1904=date_1904,
    )


def _extract_cache_points(
    cache_element,
    *,
    date_hint: bool = False,
    date_1904: bool = False,
) -> list[str]:
    points = {}
    for point in cache_element.findall("c:pt", namespaces=_NS):
        raw_index = point.get("idx")
        if raw_index is None:
            continue
        try:
            point_index = int(raw_index)
        except ValueError:
            continue

        value_element = point.find("c:v", namespaces=_NS)
        raw_value = value_element.text if value_element is not None else ""
        points[point_index] = _stringify_cache_value(
            raw_value,
            date_hint=date_hint,
            date_1904=date_1904,
        )

    if not points:
        return []

    return [points.get(index, "") for index in range(max(points.keys()) + 1)]


def _extract_multilevel_string_cache(ref_element) -> list[str]:
    level_maps = []
    max_index = -1
    for level in ref_element.findall("c:multiLvlStrCache/c:lvl", namespaces=_NS):
        values = {}
        for point in level.findall("c:pt", namespaces=_NS):
            raw_index = point.get("idx")
            if raw_index is None:
                continue
            try:
                point_index = int(raw_index)
            except ValueError:
                continue
            value_element = point.find("c:v", namespaces=_NS)
            values[point_index] = value_element.text if value_element is not None else ""
            max_index = max(max_index, point_index)
        level_maps.append(values)

    if max_index < 0:
        return []

    rows = []
    for point_index in range(max_index + 1):
        parts = [
            value_map[point_index]
            for value_map in level_maps
            if value_map.get(point_index)
        ]
        rows.append(" / ".join(parts))
    return rows


def _extract_tx_formula(tx_element) -> str | None:
    if tx_element is None:
        return None
    str_ref = tx_element.find("c:strRef", namespaces=_NS)
    if str_ref is None:
        return None
    formula_element = str_ref.find("c:f", namespaces=_NS)
    if formula_element is None or formula_element.text is None:
        return None
    return formula_element.text.strip()


def _extract_tx_text(tx_element) -> str | None:
    if tx_element is None:
        return None

    str_cache = tx_element.find("c:strRef/c:strCache", namespaces=_NS)
    if str_cache is not None:
        values = _extract_cache_points(str_cache)
        return values[0] if values else None

    value_element = tx_element.find("c:v", namespaces=_NS)
    if value_element is not None and value_element.text:
        return value_element.text.strip()

    return None


def _extract_title_text(title_element) -> str:
    if title_element is None:
        return ""
    texts = title_element.findall(".//a:t", namespaces=_NS)
    return "".join(text.text or "" for text in texts).strip()


def _find_reference_element(container):
    if container is None:
        return None
    for tag_name in ("strRef", "numRef", "multiLvlStrRef"):
        ref_element = container.find(f"c:{tag_name}", namespaces=_NS)
        if ref_element is not None:
            return ref_element
    return None


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _chart_uses_date_1904(root) -> bool:
    date_1904 = root.find("c:date1904", namespaces=_NS)
    if date_1904 is None:
        return False
    return date_1904.get("val") == "1"


def _resolve_series_name(series: SeriesSpec, index: int, workbook=None) -> str:
    if workbook is not None and series.name_formula:
        workbook_name = _read_formula_scalar(workbook, series.name_formula)
        if workbook_name:
            return workbook_name
    if series.literal_name:
        return series.literal_name
    return f"Series{index}"


def _get_shared_axis_values(sequences: list[list[Any]]) -> list[Any] | None:
    if not sequences:
        return None

    normalized = [_normalize_sequence(sequence) for sequence in sequences]
    first = normalized[0]
    if any(sequence != first for sequence in normalized[1:]):
        return None
    return sequences[0]


def _normalize_sequence(sequence: list[Any]) -> list[str]:
    return [_stringify_cell_value(value) for value in sequence]


def _stringify_series_values(
    values: list[Any],
    *,
    date_hint: bool = False,
    date_1904: bool = False,
) -> list[str]:
    return [
        _stringify_cell_value(
            value,
            date_hint=date_hint,
            date_1904=date_1904,
        )
        for value in values
    ]


def _stringify_cache_value(
    value: str | None,
    *,
    date_hint: bool = False,
    date_1904: bool = False,
) -> str:
    if value in (None, ""):
        return ""

    if date_hint:
        try:
            serial = float(value)
        except (TypeError, ValueError):
            return value
        return _excel_serial_to_iso(serial, date_1904=date_1904)

    return value


def _stringify_cell_value(
    value: Any,
    *,
    date_hint: bool = False,
    date_1904: bool = False,
) -> str:
    if value in (None, ""):
        return ""

    if isinstance(value, datetime):
        if date_hint and value.time() == time():
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()

    if date_hint and isinstance(value, (int, float)):
        return _excel_serial_to_iso(float(value), date_1904=date_1904)

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _excel_serial_to_iso(serial: float, *, date_1904: bool = False) -> str:
    epoch = _EXCEL_EPOCH_1904 if date_1904 else _EXCEL_EPOCH_1900
    dt = epoch + timedelta(days=serial)
    if dt.time() == time():
        return dt.date().isoformat()
    return dt.isoformat(sep=" ")


def _render_html_table(headers: list[str], columns: list[list[str]], row_count: int) -> str:
    if row_count <= 0 or len(headers) != len(columns):
        return ""

    html_parts = ["<table><thead><tr>"]
    for header in headers:
        html_parts.append(f"<th>{escape(header)}</th>")
    html_parts.append("</tr></thead><tbody>")

    for row_idx in range(row_count):
        html_parts.append("<tr>")
        for column in columns:
            value = column[row_idx] if row_idx < len(column) else ""
            html_parts.append(f"<td>{escape(value)}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")
    return "".join(html_parts)
