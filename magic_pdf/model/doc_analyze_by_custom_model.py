import time

import fitz
import numpy as np
from loguru import logger

from magic_pdf.libs.config_reader import get_local_models_dir, get_device, get_table_recog_config
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
            if pm.width > 3000 or pm.height > 3000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            img = np.array(img)
            img_dict = {"img": img, "width": pm.width, "height": pm.height}
            images.append(img_dict)
    return images


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, ocr: bool, show_log: bool):
        key = (ocr, show_log)
        if key not in self._models:
            self._models[key] = custom_model_init(ocr=ocr, show_log=show_log)
        return self._models[key]


def custom_model_init(ocr: bool = False, show_log: bool = False):
    model = None

    if model_config.__model_mode__ == "lite":
        logger.warning("The Lite mode is provided for developers to conduct testing only, and the output quality is "
                       "not guaranteed to be reliable.")
        model = MODEL.Paddle
    elif model_config.__model_mode__ == "full":
        model = MODEL.PEK

    if model_config.__use_inside_model__:
        model_init_start = time.time()
        if model == MODEL.Paddle:
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel
            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log)
        elif model == MODEL.PEK:
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel
            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()
            table_config = get_table_recog_config()
            model_input = {"ocr": ocr,
                           "show_log": show_log,
                           "models_dir": local_models_dir,
                           "device": device,
                           "table_config": table_config}
            custom_model = CustomPEKModel(**model_input)
        else:
            logger.error("Not allow model_name!")
            exit(1)
        model_init_cost = time.time() - model_init_start
        logger.info(f"model init cost: {model_init_cost}")
    else:
        logger.error("use_inside_model is False, not allow to use inside model")
        exit(1)

    return custom_model


def doc_analyze(pdf_bytes: bytes, ocr: bool = False, show_log: bool = False):

    model_manager = ModelSingleton()
    custom_model = model_manager.get_model(ocr, show_log)

    images = load_images_from_pdf(pdf_bytes)

    model_json = []
    doc_analyze_start = time.time()
    for index, img_dict in enumerate(images):
        img = img_dict["img"]
        page_width = img_dict["width"]
        page_height = img_dict["height"]
        result = custom_model(img)
        page_info = {"page_no": index, "height": page_height, "width": page_width}
        page_dict = {"layout_dets": result, "page_info": page_info}
        model_json.append(page_dict)
    doc_analyze_cost = time.time() - doc_analyze_start
    logger.info(f"doc analyze cost: {doc_analyze_cost}")

    return model_json
