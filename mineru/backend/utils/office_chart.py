# Copyright (c) Opendatalab. All rights reserved.
import re
from io import BytesIO

import pandas as pd


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
