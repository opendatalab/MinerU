import os
import time
from typing import List, Tuple

import pypdfium2 as pdfium
from PIL import Image
from loguru import logger
from tqdm import tqdm

from .model_init import MineruPipelineModel, PIPELINE_MODEL_INIT_LOCK
from .model_json_to_middle_json import (
    append_batch_results_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from ..utils import exclude_progress_bar_idle_time
from mineru.utils.config_reader import get_device, get_processing_window_size
from ...utils.enum_class import ImageType
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import load_images_from_pdf_doc
from ...utils.model_utils import get_vram, clean_memory
from ...utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

class ModelSingleton:
    _instance = None
    _models = {}
    _lock = PIPELINE_MODEL_INIT_LOCK

    def __new__(cls, *args, **kwargs):
        with cls._lock:
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
        with self._lock:
            if key not in self._models:
                self._models[key] = custom_model_init(
                    lang=lang,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
):
    model_init_start = time.time()
    # 从配置文件读取model-dir和device
    device = get_device()

    formula_config = {"enable": formula_enable}
    table_config = {"enable": table_enable}

    model_input = {
        'device': device,
        'table_config': table_config,
        'formula_config': formula_config,
        'lang': lang,
    }

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model


def _get_ocr_enable(pdf_bytes, parse_method: str) -> bool:
    if parse_method == 'auto':
        return classify(pdf_bytes) == 'ocr'
    if parse_method == 'ocr':
        return True
    return False


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get('img_pil')
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def _format_doc_slices(batch_slices):
    return ",".join(
        f"doc{item['doc_index']}:{item['page_start'] + 1}-{item['page_end'] + 1}"
        for item in batch_slices
    )


def _finalize_processing_window_context(context, on_doc_ready):
    if context['closed']:
        return
    finalize_middle_json(
        context['middle_json']['pdf_info'],
        lang=context['lang'],
        ocr_enable=context['ocr_enable'],
    )
    logger.debug(
        f"Pipeline doc ready: doc{context['doc_index']} pages={context['page_count']}"
    )
    on_doc_ready(
        context['doc_index'],
        context['model_list'],
        context['middle_json'],
        context['ocr_enable'],
    )
    close_pdfium_document(context['pdf_doc'])
    context['closed'] = True


def _emit_zero_page_contexts(doc_contexts, on_doc_ready):
    for context in doc_contexts:
        if context['page_count'] == 0 and not context['closed']:
            _finalize_processing_window_context(context, on_doc_ready)


def doc_analyze_streaming(
        pdf_bytes_list,
        image_writer_list,
        lang_list,
        on_doc_ready,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
):
    if not (len(pdf_bytes_list) == len(image_writer_list) == len(lang_list)):
        raise ValueError("pdf_bytes_list, image_writer_list, and lang_list must have the same length")

    doc_contexts = []
    total_pages = 0
    for doc_index, (pdf_bytes, image_writer, lang) in enumerate(
        zip(pdf_bytes_list, image_writer_list, lang_list)
    ):
        _ocr_enable = _get_ocr_enable(pdf_bytes, parse_method)
        pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
        page_count = get_pdfium_document_page_count(pdf_doc)
        total_pages += page_count
        doc_contexts.append(
            {
                'doc_index': doc_index,
                'pdf_bytes': pdf_bytes,
                'pdf_doc': pdf_doc,
                'page_count': page_count,
                'next_page_idx': 0,
                'middle_json': init_middle_json(),
                'model_list': [],
                'image_writer': image_writer,
                'lang': lang,
                'ocr_enable': _ocr_enable,
                'closed': False,
            }
        )

    if total_pages == 0:
        _emit_zero_page_contexts(doc_contexts, on_doc_ready)
        return

    window_size = get_processing_window_size(default=64)
    total_batches = (total_pages + window_size - 1) // window_size
    logger.info(
        f'Pipeline processing-window multi-file run. doc_count={len(doc_contexts)}, '
        f'total_pages={total_pages}, window_size={window_size}, total_batches={total_batches}'
    )

    _emit_zero_page_contexts(doc_contexts, on_doc_ready)
    processed_pages = 0
    infer_start = time.time()
    try:
        progress_bar = None
        last_append_end_time = None
        try:
            batch_index = 0
            while processed_pages < total_pages:
                batch_index += 1
                batch_capacity = window_size
                batch_images = []
                batch_slices = []
                batch_payloads = []

                for context in doc_contexts:
                    if batch_capacity == 0:
                        break
                    page_start = context['next_page_idx']
                    if page_start >= context['page_count']:
                        continue
                    take_count = min(batch_capacity, context['page_count'] - page_start)
                    page_end = page_start + take_count - 1
                    images_list = load_images_from_pdf_doc(
                        context['pdf_doc'],
                        start_page_id=page_start,
                        end_page_id=page_end,
                        image_type=ImageType.PIL,
                        pdf_bytes=context['pdf_bytes'],
                    )
                    images_with_extra_info = [
                        (image_dict['img_pil'], context['ocr_enable'], context['lang'])
                        for image_dict in images_list
                    ]
                    batch_images.extend(images_with_extra_info)
                    batch_slices.append(
                        {
                            'doc_index': context['doc_index'],
                            'page_start': page_start,
                            'page_end': page_end,
                            'count': take_count,
                        }
                    )
                    batch_payloads.append((context, images_list, page_start, take_count))
                    context['next_page_idx'] = page_end + 1
                    batch_capacity -= take_count

                logger.info(
                    f'Pipeline processing window batch {batch_index}/{total_batches}: '
                    f'{processed_pages + len(batch_images)}/{total_pages} pages, '
                    f'batch_pages={len(batch_images)}, doc_slices={_format_doc_slices(batch_slices)}'
                )

                batch_results = batch_image_analyze(
                    batch_images,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                )
                if progress_bar is None:
                    progress_bar = tqdm(total=total_pages, desc="Processing pages")
                else:
                    exclude_progress_bar_idle_time(
                        progress_bar,
                        last_append_end_time,
                        now=time.time(),
                    )

                result_offset = 0
                for context, images_list, page_start, take_count in batch_payloads:
                    result_slice = batch_results[result_offset: result_offset + take_count]
                    append_batch_results_to_middle_json(
                        context['middle_json'],
                        result_slice,
                        images_list,
                        context['pdf_doc'],
                        context['image_writer'],
                        page_start_index=page_start,
                        ocr_enable=context['ocr_enable'],
                        model_list=context['model_list'],
                        progress_bar=progress_bar,
                    )
                    result_offset += take_count
                    _close_images(images_list)
                    images_list.clear()

                    if context['next_page_idx'] >= context['page_count'] and not context['closed']:
                        _finalize_processing_window_context(context, on_doc_ready)

                last_append_end_time = time.time()
                processed_pages += len(batch_images)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0:
            logger.debug(
                f"processing-window multi-file infer finished, cost: {infer_time}, "
                f"speed: {round(total_pages / infer_time, 3)} page/s"
            )
    finally:
        for context in doc_contexts:
            if not context['closed']:
                close_pdfium_document(context['pdf_doc'])
                context['closed'] = True


def batch_image_analyze(
        images_with_extra_info: List[Tuple[Image.Image, bool, str]],
        formula_enable=True,
        table_enable=True):

    from .batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()

    device = get_device()

    if str(device).startswith('npu'):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        batch_ratio = 16
    elif gpu_memory >= 16:
        batch_ratio = 8
    elif gpu_memory >= 8:
        batch_ratio = 4
    elif gpu_memory >= 6:
        batch_ratio = 2
    else:
        batch_ratio = 1
    logger.info(
            f'GPU Memory: {gpu_memory} GB, Batch Ratio: {batch_ratio}. '
    )

    # 检测torch的版本号
    import torch
    from packaging import version
    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
    if device_type.lower() in ["corex"]:
        enable_ocr_det_batch = False
    else:
        if version.parse(torch.__version__) >= version.parse("2.8.0"):
            os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
        enable_ocr_det_batch = True

    batch_model = BatchAnalyze(model_manager, batch_ratio, formula_enable, table_enable, enable_ocr_det_batch)
    results = batch_model(images_with_extra_info)

    clean_memory(get_device())

    return results
