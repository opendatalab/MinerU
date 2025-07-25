
import re
import itertools
import html
from typing import Any, Dict, List
from pydantic import (
    BaseModel,
    computed_field,
    model_validator,
)

class TableCell(BaseModel):
    """TableCell."""
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, Dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                    # "bbox" not in data
                    # or data["bbox"] is None
                    # or isinstance(data["bbox"], BoundingBox)
                    "text"
                    in data
            ):
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field  # type: ignore
    @property
    def grid(
            self,
    ) -> List[List[TableCell]]:
        """grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                    min(cell.start_row_offset_idx, self.num_rows),
                    min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                        min(cell.start_col_offset_idx, self.num_cols),
                        min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data


"""
OTSL
"""
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"


def otsl_extract_tokens_and_text(s: str):
    # Pattern to match anything enclosed by < >
    # (including the angle brackets themselves)
    # pattern = r"(<[^>]+>)"
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, s)
    # Remove any tokens that start with "<loc_"
    tokens = [token for token in tokens]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]

    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # Check and complete the matrix
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)

        # Insert additional <ecel> to tags
        for row_idx, row in enumerate(split_row_tokens):
            while len(row) < max_cols:
                row.append(OTSL_ECEL)

        # Insert additional <ecel> to texts
        new_texts = []
        text_idx = 0

        for row_idx, row in enumerate(split_row_tokens):
            for col_idx, token in enumerate(row):
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if (text_idx < len(texts) and 
                        texts[text_idx] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                        new_texts.append(texts[text_idx])
                        text_idx += 1

            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1

        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel,
            # set properly row_span, col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                OTSL_LCEL,
                OTSL_XCEL,
            ]:
                # we have horisontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [OTSL_LCEL, OTSL_XCEL],
                )
            if next_bottom_cell in [
                OTSL_UCEL,
                OTSL_XCEL,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [OTSL_UCEL, OTSL_XCEL],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
            OTSL_LCEL,
            OTSL_UCEL,
            OTSL_XCEL,
        ]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData):
    nrows = table_data.num_rows
    ncols = table_data.num_cols

    text = ""

    if len(table_data.table_cells) == 0:
        return ""

    body = ""

    grid = table_data.grid
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell: TableCell = grid[i][j]

            rowspan, rowstart = (
                cell.row_span,
                cell.start_row_offset_idx,
            )
            colspan, colstart = (
                cell.col_span,
                cell.start_col_offset_idx,
            )

            if rowstart != i:
                continue
            if colstart != j:
                continue

            content = html.escape(cell.text.strip())
            celltag = "td"
            if cell.column_header:
                celltag = "th"

            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'

            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"

    # dir = get_text_direction(text)
    body = f"<table>{body}</table>"

    return body


def convert_otsl_to_html(otsl_content: str):
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(
            max(len(row) for row in split_row_tokens) if split_row_tokens else 0
        ),
        table_cells=table_cells,
    )

    return export_to_html(table_data)


def block_content_to_html(block_content: str) -> str:
    """
    Converts block content containing OTSL (Open Table Structure Language) tags into HTML.

    This function processes a block of text, splitting it into lines and converting any lines
    containing OTSL table tags (e.g., <fcel>, <ecel>) into HTML tables. Lines without these
    tags are left unchanged.

    Parameters:
        block_content (str): A string containing block content with potential OTSL tags.

    Returns:
        str: The processed block content with OTSL tags converted to HTML tables.
    """
    lines = block_content.split("\n\n")
    new_lines = []
    for line in lines:
        if "<fcel>" in line or "<ecel>" in line:
            line = convert_otsl_to_html(line)
        new_lines.append(line)
    return "\n\n".join(new_lines)
