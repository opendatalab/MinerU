
from loguru import logger

from mineru.backend.office.office_magic_model import MagicModel
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.version import __version__



def blocks_to_page_info(page_blocks, image_writer, page_index) -> dict:
    """将blocks转换为页面信息"""

    magic_model = MagicModel(page_blocks)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()
    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)

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


def result_to_middle_json(model_output_blocks_list, images_list, pdf_doc, image_writer):
    middle_json = {"pdf_info": [], "_backend":"vlm", "_version_name": __version__}
    for index, page_blocks in enumerate(model_output_blocks_list):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = blocks_to_page_info(page_blocks, image_dict, page, image_writer, index)
        middle_json["pdf_info"].append(page_info)

    # 关闭pdf文档
    pdf_doc.close()
    return middle_json