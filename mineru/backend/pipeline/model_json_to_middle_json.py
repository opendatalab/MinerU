# Copyright (c) Opendatalab. All rights reserved.
from mineru.utils.pipeline_magic_model import MagicModel
from mineru.version import __version__
from mineru.utils.hash_utils import str_md5


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, lang=None, ocr=False):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = str_md5(image_dict["img_base64"])
    width, height = map(int, page.get_size())
    magic_model = MagicModel(page_model_info, scale)



def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr=False):
    middle_json = {"pdf_info": [], "_backend":"vlm", "_version_name": __version__}
    for page_index, page_model_info in enumerate(model_list):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, lang=lang, ocr=ocr
        )
        middle_json["pdf_info"].append(page_info)
    return middle_json