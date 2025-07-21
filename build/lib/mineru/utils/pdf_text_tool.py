from typing import List
import math

import pypdfium2 as pdfium
from pdftext.pdf.chars import get_chars, deduplicate_chars
from pdftext.pdf.pages import get_spans, get_lines, assign_scripts, get_blocks


def get_page(
    page: pdfium.PdfPage,
    quote_loosebox: bool =True,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> dict:

        textpage = page.get_textpage()
        page_bbox: List[float] = page.get_bbox()
        page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
        page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

        page_rotation = 0
        try:
            page_rotation = page.get_rotation()
        except:
            pass

        chars = deduplicate_chars(get_chars(textpage, page_bbox, page_rotation, quote_loosebox))
        spans = get_spans(chars, superscript_height_threshold=superscript_height_threshold, line_distance_threshold=line_distance_threshold)
        lines = get_lines(spans)
        assign_scripts(lines, height_threshold=superscript_height_threshold, line_distance_threshold=line_distance_threshold)
        blocks = get_blocks(lines)

        page = {
            "bbox": page_bbox,
            "width": page_width,
            "height": page_height,
            "rotation": page_rotation,
            "blocks": blocks
        }
        return page