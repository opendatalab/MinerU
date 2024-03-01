import math
from magic_pdf.para.commons import *


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class LayoutFilterProcessor:
    def __init__(self) -> None:
        pass

    def batch_process_blocks(self, pdf_dict):
        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                if "layout_bboxes" in blocks.keys() and "para_blocks" in blocks.keys():
                    layout_bbox_objs = blocks["layout_bboxes"]
                    if layout_bbox_objs is None:
                        continue
                    layout_bboxes = [bbox_obj["layout_bbox"] for bbox_obj in layout_bbox_objs]

                    # Use math.ceil function to enlarge each value of x0, y0, x1, y1 of each layout_bbox
                    layout_bboxes = [
                        [math.ceil(x0), math.ceil(y0), math.ceil(x1), math.ceil(y1)] for x0, y0, x1, y1 in layout_bboxes
                    ]

                    para_blocks = blocks["para_blocks"]
                    if para_blocks is None:
                        continue

                    for lb_bbox in layout_bboxes:
                        for i, para_block in enumerate(para_blocks):
                            para_bbox = para_block["bbox"]
                            para_blocks[i]["in_layout"] = 0
                            if is_in_bbox(para_bbox, lb_bbox):
                                para_blocks[i]["in_layout"] = 1

                    blocks["para_blocks"] = para_blocks

        return pdf_dict
