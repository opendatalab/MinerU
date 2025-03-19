# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path

import yaml
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.data.utils import load_images_from_pdf
from magic_pdf.libs.config_reader import get_local_models_dir, get_device
from magic_pdf.libs.pdf_check import extract_pages
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton


def get_model_config():
    local_models_dir = get_local_models_dir()
    device = get_device()
    current_file_path = os.path.abspath(__file__)
    root_dir = Path(current_file_path).parents[3]
    model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
    config_path = os.path.join(model_config_dir, 'model_configs.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return root_dir, local_models_dir, device, configs


def get_text_images(simple_images):
    _, local_models_dir, device, configs = get_model_config()
    atom_model_manager = AtomModelSingleton()
    temp_layout_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.Layout,
        layout_model_name=MODEL_NAME.DocLayout_YOLO,
        doclayout_yolo_weights=str(
            os.path.join(
                local_models_dir, configs['weights'][MODEL_NAME.DocLayout_YOLO]
            )
        ),
        device=device,
    )
    text_images = []
    for simple_image in simple_images:
        image = simple_image['img']
        layout_res = temp_layout_model.predict(image)
        # 给textblock截图
        for res in layout_res:
            if res['category_id'] in [1]:
                x1, y1, _, _, x2, y2, _, _ = res['poly']
                # 初步清洗（宽和高都小于100）
                if x2 - x1 < 100 and y2 - y1 < 100:
                    continue
                text_images.append(image[y1:y2, x1:x2])
    return text_images


def auto_detect_lang(pdf_bytes: bytes):
    sample_docs = extract_pages(pdf_bytes)
    sample_pdf_bytes = sample_docs.tobytes()
    simple_images = load_images_from_pdf(sample_pdf_bytes, dpi=200)
    text_images = get_text_images(simple_images)
    langdetect_model = model_init(MODEL_NAME.YOLO_V11_LangDetect)
    lang = langdetect_model.do_detect(text_images)
    return lang


def model_init(model_name: str):
    atom_model_manager = AtomModelSingleton()

    if model_name == MODEL_NAME.YOLO_V11_LangDetect:
        root_dir, _, device, _ = get_model_config()
        model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.LangDetect,
            langdetect_model_name=MODEL_NAME.YOLO_V11_LangDetect,
            langdetect_model_weight=str(os.path.join(root_dir, 'resources', 'yolov11-langdetect', 'yolo_v11_ft.pt')),
            device=device,
        )
    else:
        raise ValueError(f"model_name {model_name} not found")
    return model

