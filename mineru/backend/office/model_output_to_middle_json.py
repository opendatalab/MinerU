
import base64
import re
from collections import defaultdict

from loguru import logger

from mineru.backend.office.office_magic_model import MagicModel
from mineru.utils.enum_class import BlockType
from mineru.utils.hash_utils import str_sha256
from mineru.version import __version__


def blocks_to_page_info(page_blocks, image_writer, page_index) -> dict:
    """将blocks转换为页面信息"""

    magic_model = MagicModel(page_blocks)
    image_blocks = magic_model.get_image_blocks()

    # Write embedded images to local storage via image_writer
    if image_writer:
        for img_block in image_blocks:
            for sub_block in img_block.get("blocks", []):
                if sub_block.get("type") != "image_body":
                    continue
                for line in sub_block.get("lines", []):
                    for span in line.get("spans", []):
                        img_b64 = span.get("image_base64", "")
                        if not img_b64:
                            continue
                        # Parse data URI: data:image/{format};base64,{data}
                        m = re.match(r'data:image/(\w+);base64,(.+)', img_b64, re.DOTALL)
                        if not m:
                            logger.warning(f"Unrecognized image_base64 format in page {page_index}, skipping.")
                            continue
                        fmt = m.group(1)
                        ext = "jpg" if fmt == "jpeg" else fmt
                        try:
                            img_bytes = base64.b64decode(m.group(2))
                        except Exception as e:
                            logger.warning(f"Failed to decode image_base64 on page {page_index}: {e}")
                            continue
                        img_path = f"{str_sha256(img_b64)}.{ext}"
                        image_writer.write(img_path, img_bytes)
                        span["image_path"] = img_path
                        del span["image_base64"]

    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    list_blocks = magic_model.get_list_blocks()
    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    page_blocks = []
    page_blocks.extend([
        *image_blocks,
        *table_blocks,
        *title_blocks,
        *text_blocks,
        *interline_equation_blocks,
        *list_blocks,
    ])
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {"para_blocks": page_blocks, "discarded_blocks": discarded_blocks, "page_idx": page_index}
    return page_info


def _extract_section_parts_from_content(content: str, level: int):
    """Try to extract a leading section number (e.g. '1.2.1') from title content.

    Returns a list of ints [n1, n2, ..., nL] when the number of parts equals
    `level`, otherwise None.  Handles formats like:
        '1心肌特异性...'       (no separator)
        '1.2.1建立...'         (Chinese text immediately after number)
        '2.2.1 ALKBH5 ...'    (space separator)
    """
    match = re.match(r'^(\d+(?:\.\d+)*)', content.strip())
    if match:
        parts = [int(p) for p in match.group(1).split('.')]
        if len(parts) == level:
            return parts
    return None


def result_to_middle_json(model_output_blocks_list, image_writer):
    middle_json = {"pdf_info": [], "_backend":"office", "_version_name": __version__}
    for index, page_blocks in enumerate(model_output_blocks_list):
        page_info = blocks_to_page_info(page_blocks, image_writer, index)
        middle_json["pdf_info"].append(page_info)

    section_counters: dict[int, int] = defaultdict(int)
    for page_info in middle_json["pdf_info"]:
        for block in page_info.get("para_blocks", []):
            if block.get("type") != BlockType.TITLE:
                continue
            level = block.get("level", 1)
            if block.get("is_numbered_style", False):
                # Ensure all ancestor levels start at 1 (never 0)
                for ancestor in range(1, level):
                    if section_counters[ancestor] == 0:
                        section_counters[ancestor] = 1
                # Increment current level counter and reset all deeper levels
                section_counters[level] += 1
                for deeper in list(section_counters.keys()):
                    if deeper > level:
                        section_counters[deeper] = 0
                # Build section number string, e.g. "1.2.1."
                section_number = ".".join(
                    str(section_counters[l]) for l in range(1, level + 1)
                ) + "."
                block["section_number"] = section_number
            else:
                # Some documents embed the section number directly in the content
                # (is_numbered_style=False).  Parse it and sync the counters so
                # that subsequent numbered blocks continue from the right base.
                lines = block.get("lines", [])
                content = ""
                if lines and lines[0].get("spans"):
                    content = lines[0]["spans"][0].get("content", "")
                parts = _extract_section_parts_from_content(content, level)
                if parts:
                    for k, v in enumerate(parts, start=1):
                        section_counters[k] = v
                    # Reset all deeper levels
                    for deeper in list(section_counters.keys()):
                        if deeper > level:
                            section_counters[deeper] = 0

    return middle_json