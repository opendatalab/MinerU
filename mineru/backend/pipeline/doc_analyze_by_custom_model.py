import os
import time
import numpy as np
import torch
from mineru.backend.pipeline.model_init import MineruPipelineModel

os.environ['FLAGS_npu_jit_compile'] = '0'  # 关闭paddle的jit编译
os.environ['FLAGS_use_stride_kernel'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新


from loguru import logger

from ...utils.model_utils import get_vram, clean_memory
from magic_pdf.libs.config_reader import (get_device, get_formula_config,
                                          get_layout_config,
                                          get_local_models_dir,
                                          get_table_recog_config)

class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        lang=None,
        formula_enable=None,
        table_enable=None,
    ):
        key = (lang, formula_enable, table_enable)
        if key not in self._models:
            self._models[key] = custom_model_init(
                lang=lang,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=None,
    table_enable=None,
):
    model_init_start = time.time()
    # 从配置文件读取model-dir和device
    local_models_dir = get_local_models_dir()
    device = get_device()

    formula_config = get_formula_config()
    if formula_enable is not None:
        formula_config['enable'] = formula_enable

    table_config = get_table_recog_config()
    if table_enable is not None:
        table_config['enable'] = table_enable

    model_input = {
        'models_dir': local_models_dir,
        'device': device,
        'table_config': table_config,
        'formula_config': formula_config,
        'lang': lang,
    }

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model

def doc_analyze(
    dataset: Dataset,
    ocr: bool = False,
    start_page_id=0,
    end_page_id=None,
    lang=None,
    formula_enable=None,
    table_enable=None,
):
    end_page_id = (
        end_page_id
        if end_page_id is not None and end_page_id >= 0
        else len(dataset) - 1
    )

    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 100))
    images = []
    page_wh_list = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            images.append(img_dict['img'])
            page_wh_list.append((img_dict['width'], img_dict['height']))

    images_with_extra_info = [(images[index], ocr, dataset._lang) for index in range(len(images))]

    if len(images) >= MIN_BATCH_INFERENCE_SIZE:
        batch_size = MIN_BATCH_INFERENCE_SIZE
        batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    else:
        batch_images = [images_with_extra_info]

    results = []
    processed_images_count = 0
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        logger.info(f'Batch {index + 1}/{len(batch_images)}: {processed_images_count} pages/{len(images_with_extra_info)} pages')
        result = may_batch_image_analyze(batch_image, formula_enable, table_enable)
        results.extend(result)

    model_json = []
    for index in range(len(dataset)):
        if start_page_id <= index <= end_page_id:
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
        else:
            result = []
            page_height = 0
            page_width = 0

        page_info = {'page_no': index, 'width': page_width, 'height': page_height}
        page_dict = {'layout_dets': result, 'page_info': page_info}
        model_json.append(page_dict)

    return model_json

def batch_doc_analyze(
    datasets: list[Dataset],
    parse_method: str = 'auto',
    lang=None,
    formula_enable=None,
    table_enable=None,
):
    MIN_BATCH_INFERENCE_SIZE = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 100))
    batch_size = MIN_BATCH_INFERENCE_SIZE
    page_wh_list = []

    images_with_extra_info = []
    for dataset in datasets:

        ocr = False
        if parse_method == 'auto':
            if dataset.classify() == 'txt':
                ocr = False
            elif dataset.classify() == 'ocr':
                ocr = True
        elif parse_method == 'ocr':
            ocr = True
        elif parse_method == 'txt':
            ocr = False

        _lang = dataset._lang

        for index in range(len(dataset)):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            page_wh_list.append((img_dict['width'], img_dict['height']))
            images_with_extra_info.append((img_dict['img'], ocr, _lang))

    batch_images = [images_with_extra_info[i:i+batch_size] for i in range(0, len(images_with_extra_info), batch_size)]
    results = []
    processed_images_count = 0
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        logger.info(f'Batch {index + 1}/{len(batch_images)}: {processed_images_count} pages/{len(images_with_extra_info)} pages')
        result = may_batch_image_analyze(batch_image, formula_enable, table_enable)
        results.extend(result)

    infer_results = []
    for index in range(len(datasets)):
        dataset = datasets[index]
        model_json = []
        for i in range(len(dataset)):
            result = results.pop(0)
            page_width, page_height = page_wh_list.pop(0)
            page_info = {'page_no': i, 'width': page_width, 'height': page_height}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)
        infer_results.append(model_json)
    return infer_results


def may_batch_image_analyze(
        images_with_extra_info: list[(np.ndarray, bool, str)],
        formula_enable=None,
        table_enable=None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)

    from .batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()

    batch_ratio = 1
    device = get_device()

    if str(device).startswith('npu'):
        import torch_npu
        if torch_npu.npu.is_available():
            torch.npu.set_compile_mode(jit_compile=False)

    if str(device).startswith('npu') or str(device).startswith('cuda'):
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
        else:
            # Default batch_ratio when VRAM can't be determined
            batch_ratio = 1
            logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_ratio}')

    batch_model = BatchAnalyze(model_manager, batch_ratio, formula_enable, table_enable)
    results = batch_model(images_with_extra_info)

    clean_memory(get_device())

    return results