import os
import time

# 关闭paddle的信号处理
import paddle
import torch
from loguru import logger

from magic_pdf.model.batch_analyze import BatchAnalyze
from magic_pdf.model.sub_modules.model_utils import get_vram

paddle.disable_signal_handler()

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

try:
    import torchtext

    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

import magic_pdf.model as model_config
from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import (get_device, get_formula_config,
                                          get_layout_config,
                                          get_local_models_dir,
                                          get_table_recog_config)
from magic_pdf.model.model_list import MODEL
from magic_pdf.operators.models import InferenceResult


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


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        ocr: bool,
        show_log: bool,
        lang=None,
        layout_model=None,
        formula_enable=None,
        table_enable=None,
    ):
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        if key not in self._models:
            self._models[key] = custom_model_init(
                ocr=ocr,
                show_log=show_log,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        return self._models[key]


def custom_model_init(
    ocr: bool = False,
    show_log: bool = False,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):

    model = None

    if model_config.__model_mode__ == 'lite':
        logger.warning(
            'The Lite mode is provided for developers to conduct testing only, and the output quality is '
            'not guaranteed to be reliable.'
        )
        model = MODEL.Paddle
    elif model_config.__model_mode__ == 'full':
        model = MODEL.PEK

    if model_config.__use_inside_model__:
        model_init_start = time.time()
        if model == MODEL.Paddle:
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel

            custom_model = CustomPaddleModel(ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel

            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()

            layout_config = get_layout_config()
            if layout_model is not None:
                layout_config['model'] = layout_model

            formula_config = get_formula_config()
            if formula_enable is not None:
                formula_config['enable'] = formula_enable

            table_config = get_table_recog_config()
            if table_enable is not None:
                table_config['enable'] = table_enable

            model_input = {
                'ocr': ocr,
                'show_log': show_log,
                'models_dir': local_models_dir,
                'device': device,
                'table_config': table_config,
                'layout_config': layout_config,
                'formula_config': formula_config,
                'lang': lang,
            }

            custom_model = CustomPEKModel(**model_input)
        else:
            logger.error('Not allow model_name!')
            exit(1)
        model_init_cost = time.time() - model_init_start
        logger.info(f'model init cost: {model_init_cost}')
    else:
        logger.error('use_inside_model is False, not allow to use inside model')
        exit(1)

    return custom_model


def doc_analyze(
    dataset: Dataset,
    ocr: bool = False,
    show_log: bool = False,
    start_page_id=0,
    end_page_id=None,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
) -> InferenceResult:

    end_page_id = end_page_id if end_page_id else len(dataset) - 1

    model_manager = ModelSingleton()
    custom_model = model_manager.get_model(
        ocr, show_log, lang, layout_model, formula_enable, table_enable
    )

    batch_analyze = False
    device = get_device()

    npu_support = False
    if str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            npu_support = True

    override_batch_ratio = int(os.getenv("MINERU_OVERRIDE_BATCH_RATIO", 0))

    if torch.cuda.is_available() and device != 'cpu' or npu_support:
        gpu_memory = int(os.getenv("VIRTUAL_VRAM_SIZE", round(get_vram(device))))
        if gpu_memory is not None and gpu_memory >= 8:
            batch_ratio = 0

            if override_batch_ratio > 0:
                batch_ratio = override_batch_ratio 
            else:
                if 8 <= gpu_memory < 10:
                    batch_ratio = 2
                elif 10 <= gpu_memory <= 12:
                    batch_ratio = 4
                elif 12 < gpu_memory <= 16:
                    batch_ratio = 8
                elif 16 < gpu_memory <= 24:
                    batch_ratio = 16
                else:
                    batch_ratio = 32

            if batch_ratio >= 1:
                logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
                batch_model = BatchAnalyze(model=custom_model, batch_ratio=batch_ratio)
                batch_analyze = True

    model_json = []
    doc_analyze_start = time.time()

    if batch_analyze:
        # batch analyze
        images = []
        for index in range(len(dataset)):
            if start_page_id <= index <= end_page_id:
                page_data = dataset.get_page(index)
                img_dict = page_data.get_image()
                images.append(img_dict['img'])
        analyze_result = batch_model(images)

        for index in range(len(dataset)):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            page_width = img_dict['width']
            page_height = img_dict['height']
            if start_page_id <= index <= end_page_id:
                result = analyze_result.pop(0)
            else:
                result = []

            page_info = {'page_no': index, 'height': page_height, 'width': page_width}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)

    else:
        # single analyze

        for index in range(len(dataset)):
            page_data = dataset.get_page(index)
            img_dict = page_data.get_image()
            img = img_dict['img']
            page_width = img_dict['width']
            page_height = img_dict['height']
            if start_page_id <= index <= end_page_id:
                page_start = time.time()
                result = custom_model(img)
                logger.info(f'-----page_id : {index}, page total time: {round(time.time() - page_start, 2)}-----')
            else:
                result = []

            page_info = {'page_no': index, 'height': page_height, 'width': page_width}
            page_dict = {'layout_dets': result, 'page_info': page_info}
            model_json.append(page_dict)

    gc_start = time.time()
    clean_memory(get_device())
    gc_time = round(time.time() - gc_start, 2)
    logger.info(f'gc time: {gc_time}')

    doc_analyze_time = round(time.time() - doc_analyze_start, 2)
    doc_analyze_speed = round((end_page_id + 1 - start_page_id) / doc_analyze_time, 2)
    logger.info(
        f'doc analyze time: {round(time.time() - doc_analyze_start, 2)},'
        f' speed: {doc_analyze_speed} pages/second'
    )

    return InferenceResult(model_json, dataset)
