import base64
import os
import re
import time

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from mineru.backend.utils import cross_page_table_merge
from mineru.backend.vlm.vlm_magic_model import MagicModel
from mineru.utils.config_reader import get_table_enable, get_llm_aided_config
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.hash_utils import bytes_md5, str_sha256
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.pdfium_guard import close_pdfium_document, pdfium_guard
from mineru.version import __version__


heading_level_import_success = False
llm_aided_config = get_llm_aided_config()
if llm_aided_config:
    title_aided_config = llm_aided_config.get('title_aided', {})
    if title_aided_config.get('enable', False):
        try:
            from mineru.utils.llm_aided import llm_aided_title
            from mineru.backend.pipeline.model_init import AtomModelSingleton
            heading_level_import_success = True
        except Exception as e:
            logger.warning("The heading level feature cannot be used. If you need to use the heading level feature, "
                            "please execute `pip install mineru[core]` to install the required packages.")


def _save_base64_image(b64_data_uri: str, image_writer, page_index: int):
    """Persist a data-URI image via image_writer and return a relative path."""
    m = re.match(r'data:image/(\w+);base64,(.+)', b64_data_uri, re.DOTALL)
    if not m:
        logger.warning(f"Unrecognized image_base64 format in page {page_index}, skipping.")
        return None

    fmt = m.group(1)
    ext = "jpg" if fmt == "jpeg" else fmt
    try:
        img_bytes = base64.b64decode(m.group(2))
    except Exception as e:
        logger.warning(f"Failed to decode image_base64 on page {page_index}: {e}")
        return None

    img_path = f"{str_sha256(b64_data_uri)}.{ext}"
    image_writer.write(img_path, img_bytes)
    return img_path


def _replace_inline_base64_img_src(markup: str, image_writer, page_index: int) -> str:
    """Replace inline base64 image sources in table HTML with local relative paths."""
    if not markup or "base64," not in markup:
        return markup

    def _replace_src(match, _writer=image_writer, _idx=page_index):
        img_path = _save_base64_image(match.group(1), _writer, _idx)
        if img_path:
            return f'src="{img_path}"'
        return match.group(0)

    return re.sub(
        r'src="(data:image/[^"]+)"',
        _replace_src,
        markup,
    )


def _replace_inline_table_images(table_blocks: list[dict], image_writer, page_index: int) -> None:
    """Persist inline base64 images embedded inside table HTML."""
    if not image_writer:
        return

    for block in table_blocks:
        for sub_block in block.get("blocks", []):
            if sub_block.get("type") != BlockType.TABLE_BODY:
                continue

            for line in sub_block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") != ContentType.TABLE:
                        continue
                    span["html"] = _replace_inline_base64_img_src(
                        span.get("html", ""),
                        image_writer,
                        page_index,
                    )


def blocks_to_page_info(page_blocks, image_dict, page, image_writer, page_index) -> dict:
    """将blocks转换为页面信息"""

    scale = image_dict["scale"]
    # page_pil_img = image_dict["img_pil"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        width, height = map(int, page.get_size())

    magic_model = MagicModel(page_blocks, width, height)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()

    # 如果有标题优化需求，则对title_blocks截图det
    if heading_level_import_success:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang='ch_lite'
        )
        for title_block in title_blocks:
            title_pil_img = get_crop_img(title_block['bbox'], page_pil_img, scale)
            title_np_img = np.array(title_pil_img)
            # 给title_pil_img添加上下左右各50像素白边padding
            title_np_img = cv2.copyMakeBorder(
                title_np_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            title_img = cv2.cvtColor(title_np_img, cv2.COLOR_RGB2BGR)
            ocr_det_res = ocr_model.ocr(title_img, rec=False)[0]
            if len(ocr_det_res) > 0:
                # 计算所有res的平均高度
                avg_height = np.mean([box[2][1] - box[0][1] for box in ocr_det_res])
                title_block['line_avg_height'] = round(avg_height/scale)

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

    _replace_inline_table_images(table_blocks, image_writer, page_index)

    page_blocks = []
    page_blocks.extend([
        *image_blocks,
        *table_blocks,
        *code_blocks,
        *ref_text_blocks,
        *phonetic_blocks,
        *title_blocks,
        *text_blocks,
        *interline_equation_blocks,
        *list_blocks,
    ])
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {"para_blocks": page_blocks, "discarded_blocks": discarded_blocks, "page_size": [width, height], "page_idx": page_index}
    return page_info


def init_middle_json():
    return {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}


def append_page_blocks_to_middle_json(
    middle_json,
    model_output_blocks_list,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    progress_bar=None,
):
    for offset, (page_blocks, image_dict) in enumerate(zip(model_output_blocks_list, images_list)):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index]
        page_info = blocks_to_page_info(page_blocks, image_dict, page, image_writer, page_index)
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def finalize_middle_json(pdf_info_list):
    table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')
    if table_enable:
        cross_page_table_merge(pdf_info_list)

    if heading_level_import_success:
        llm_aided_title_start_time = time.time()
        llm_aided_title(pdf_info_list, title_aided_config)
        logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')


def result_to_middle_json(model_output_blocks_list, images_list, pdf_doc, image_writer):
    middle_json = init_middle_json()
    with tqdm(total=len(model_output_blocks_list), desc="Processing pages") as progress_bar:
        append_page_blocks_to_middle_json(
            middle_json,
            model_output_blocks_list,
            images_list,
            pdf_doc,
            image_writer,
            progress_bar=progress_bar,
        )

    finalize_middle_json(middle_json["pdf_info"])
    close_pdfium_document(pdf_doc)
    return middle_json
