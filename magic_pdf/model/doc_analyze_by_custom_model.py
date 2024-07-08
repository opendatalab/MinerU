import fitz
import numpy as np
from loguru import logger
from magic_pdf.model.model_list import MODEL
import magic_pdf.model as model_config


def dict_compare(d1, d2):
    return d1.items() == d2.items()


def remove_duplicates_dicts(lst):
    unique_dicts = []
    for dict_item in lst:
        if not any(
                dict_compare(dict_item, existing_dict) for existing_dict in unique_dicts
        ):
            unique_dicts.append(dict_item)
    return unique_dicts


def load_images_from_pdf(pdf_bytes: bytes, dpi=200) -> list:
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed, please install by pip.")
        exit(1)

    images = []
    with fitz.open("pdf", pdf_bytes) as doc:
        for index in range(0, doc.page_count):
            page = doc[index]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 3000 pixels, don't enlarge the image
            if pix.width > 3000 or pix.height > 3000:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            img = np.array(img)
            img_dict = {"img": img, "width": pm.width, "height": pm.height}
            images.append(img_dict)
    return images


def doc_analyze(pdf_bytes: bytes, ocr: bool = False, show_log: bool = False, model=MODEL.Paddle):

    if model_config.__use_inside_model__:
        from magic_pdf.model.pp_structure_v2 import CustomPaddleModel
    else:
        logger.error("use_inside_model is False, not allow to use inside model")
        exit(1)

    images = load_images_from_pdf(pdf_bytes)
    custom_model = None
    if model == MODEL.Paddle:
        custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log)
    else:
        pass
    model_json = []
    for index, img_dict in enumerate(images):
        img = img_dict["img"]
        page_width = img_dict["width"]
        page_height = img_dict["height"]
        result = custom_model(img)
        page_info = {"page_no": index, "height": page_height, "width": page_width}
        page_dict = {"layout_dets": result, "page_info": page_info}

        model_json.append(page_dict)

    # @todo 把公式识别放在后置位置,待整本全部模型结果出来之后再补公式数据

    return model_json
