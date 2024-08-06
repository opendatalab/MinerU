import time

import fitz
import numpy as np
from loguru import logger

from magic_pdf.libs.config_reader import get_local_models_dir, get_device, get_table_recog_config
from magic_pdf.libs.config_reader import get_local_models_dir, get_device, get_table_recog_config
from magic_pdf.model.model_list import MODEL
import magic_pdf.model as model_config

import cv2
import math

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
            # try: #sometimes Hough doesn't work if there is too little content on image
            #     img = align_image(img)
            # except:
            #     pass
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
        model = MODEL.Paddle
        logger.info("USING PADDLE MODEL")
    elif model_config.__model_mode__ == "full":
        model = MODEL.PEK
        logger.info("USING PEK MODEL")
    elif model_config.__model_mode__ == "torrone":
        model = MODEL.Torrone
        logger.info("USING TORRONE MODEL")
    elif model_config.__model_mode__ == "tesseract":
        model = MODEL.Tesseract
        logger.info("USING TESSERACT MODEL")


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
        elif model == MODEL.Torrone:
            from magic_pdf.model.torrone_custom import CustomTorroneModel
            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()
            table_config = get_table_recog_config()
            model_input = {"ocr": ocr,
                           "show_log": show_log,
                           "models_dir": local_models_dir,
                           "device": device,
                           "table_config": table_config}
            custom_model = CustomTorroneModel(**model_input)
        elif model == MODEL.Tesseract:
            from magic_pdf.model.custom_tesseract import CustomTesseractModel
            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()
            table_config = get_table_recog_config()
            model_input = {"ocr": ocr,
                           "show_log": show_log,
                           "models_dir": local_models_dir,
                           "device": device,
                           "table_config": table_config}
            custom_model = CustomTesseractModel(**model_input)
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

def get_angle(x1, y1, x2, y2) -> float:
    """Get the angle of this line with the horizontal axis."""
    deltaX = x2 - x1
    deltaY = y2 - y1
    angleInDegrees = np.arctan2(deltaY , deltaX) * 180 / math.pi
    
    return angleInDegrees

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255) )
    return result


def align_image(img):

    # binarization with OTSU threshold finder. 0 and 255 are ignored
    threshValue, binaryImage = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    edges = cv2.Canny(binaryImage, 80, 120)

    # Detect and draw lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=20, maxLineGap=15)
    # sort lines from widest to shortest
    lines = sorted(lines,key = (lambda l: abs(l[0][0]-l[0][2])) , reverse = True)

    # if there exist any line, compare it by horizontal line
    # and rotate the image if the angle difference is more than 0.25
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (abs(x2-x1) / edges.shape[1])>0.25 :
                angle = get_angle(x1, y1, x2, y2)
                if abs(angle) > 1.0 :
                    logger.info(f"rotated image {angle} degrees")
                    img = rotate_image(img,angle)
        #exit after comparing widest line
        break

    return img
