import os
import time
from typing import List, Tuple

import pypdfium2 as pdfium
from PIL import Image
from loguru import logger
from tqdm import tqdm

from .model_init import MineruPipelineModel
from .model_json_to_middle_json import append_batch_results_to_middle_json, finalize_middle_json, init_middle_json
from mineru.utils.config_reader import get_device, get_low_memory_window_size
from ...utils.enum_class import ImageType
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import load_images_from_pdf, load_images_from_pdf_doc
from ...utils.model_utils import get_vram, clean_memory


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

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


def _get_low_memory_window_size(page_count: int) -> int:
    return min(page_count, get_low_memory_window_size(default=64))


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

def doc_analyze(
        pdf_bytes_list,
        lang_list,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
):
    """
    适当调大MIN_BATCH_INFERENCE_SIZE可以提高性能，更大的 MIN_BATCH_INFERENCE_SIZE会消耗更多内存，
    可通过环境变量MINERU_MIN_BATCH_INFERENCE_SIZE设置，默认值为384。
    """
    min_batch_inference_size = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 384))

    # 收集所有页面信息
    all_pages_info = []  # 存储(dataset_index, page_index, img, ocr, lang, width, height)

    all_image_lists = []
    all_pdf_docs = []
    ocr_enabled_list = []
    load_images_start = time.time()
    for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
        # 确定OCR设置
        _ocr_enable = False
        if parse_method == 'auto':
            if classify(pdf_bytes) == 'ocr':
                _ocr_enable = True
        elif parse_method == 'ocr':
            _ocr_enable = True

        ocr_enabled_list.append(_ocr_enable)
        _lang = lang_list[pdf_idx]

        # 收集每个数据集中的页面
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        all_image_lists.append(images_list)
        all_pdf_docs.append(pdf_doc)
        for page_idx in range(len(images_list)):
            img_dict = images_list[page_idx]
            all_pages_info.append((
                pdf_idx, page_idx,
                img_dict['img_pil'], _ocr_enable, _lang,
            ))
    load_images_time = round(time.time() - load_images_start, 2)
    logger.debug(f"load images cost: {load_images_time}, speed: {round(len(all_pages_info) / load_images_time, 3)} images/s")

    # 准备批处理
    images_with_extra_info = [(info[2], info[3], info[4]) for info in all_pages_info]
    batch_size = min_batch_inference_size
    batch_images = [
        images_with_extra_info[i:i + batch_size]
        for i in range(0, len(images_with_extra_info), batch_size)
    ]

    # 执行批处理
    results = []
    processed_images_count = 0
    infer_start = time.time()
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        logger.info(
            f'Batch {index + 1}/{len(batch_images)}: '
            f'{processed_images_count} pages/{len(images_with_extra_info)} pages'
        )
        batch_results = batch_image_analyze(batch_image, formula_enable, table_enable)
        results.extend(batch_results)
    infer_time = round(time.time() - infer_start, 2)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results) / infer_time, 3)} page/s")

    # 构建返回结果
    infer_results = []

    for _ in range(len(pdf_bytes_list)):
        infer_results.append([])

    for i, page_info in enumerate(all_pages_info):
        pdf_idx, page_idx, pil_img, _, _ = page_info
        result = results[i]

        page_info_dict = {'page_no': page_idx, 'width': pil_img.width, 'height': pil_img.height}
        page_dict = {'layout_dets': result, 'page_info': page_info_dict}

        infer_results[pdf_idx].append(page_dict)

    return infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list


def doc_analyze_low_memory(
        pdf_bytes,
        image_writer,
        lang,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
):
    middle_json_list, model_list_list, _ = doc_analyze_low_memory_multi(
        [pdf_bytes],
        [image_writer],
        [lang],
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )
    return middle_json_list[0], model_list_list[0]


def doc_analyze_low_memory_multi(
        pdf_bytes_list,
        image_writer_list,
        lang_list,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
):
    if not (len(pdf_bytes_list) == len(image_writer_list) == len(lang_list)):
        raise ValueError("pdf_bytes_list, image_writer_list, and lang_list must have the same length")

    doc_contexts = []
    total_pages = 0
    ocr_enabled_list = []
    for doc_index, (pdf_bytes, image_writer, lang) in enumerate(zip(pdf_bytes_list, image_writer_list, lang_list)):
        _ocr_enable = _get_ocr_enable(pdf_bytes, parse_method)
        pdf_doc = pdfium.PdfDocument(pdf_bytes)
        page_count = len(pdf_doc)
        total_pages += page_count
        ocr_enabled_list.append(_ocr_enable)
        doc_contexts.append(
            {
                'doc_index': doc_index,
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
        for context in doc_contexts:
            context['pdf_doc'].close()
            context['closed'] = True
        return (
            [context['middle_json'] for context in doc_contexts],
            [context['model_list'] for context in doc_contexts],
            ocr_enabled_list,
        )

    window_size = get_low_memory_window_size(default=64)
    total_batches = (total_pages + window_size - 1) // window_size
    logger.info(
        f'Pipeline low-memory multi-file mode enabled. doc_count={len(doc_contexts)}, '
        f'total_pages={total_pages}, window_size={window_size}, total_batches={total_batches}'
    )

    processed_pages = 0
    infer_start = time.time()
    try:
        with tqdm(total=total_pages, desc="Processing pages") as progress_bar:
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
                    f'Pipeline low-memory batch {batch_index}/{total_batches}: '
                    f'{processed_pages + len(batch_images)}/{total_pages} pages, '
                    f'batch_pages={len(batch_images)}, doc_slices={_format_doc_slices(batch_slices)}'
                )

                batch_results = batch_image_analyze(
                    batch_images,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
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
                        finalize_middle_json(
                            context['middle_json']['pdf_info'],
                            lang=context['lang'],
                            ocr_enable=context['ocr_enable'],
                        )
                        context['pdf_doc'].close()
                        context['closed'] = True

                processed_pages += len(batch_images)

        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0:
            logger.debug(
                f"low-memory multi-file infer finished, cost: {infer_time}, "
                f"speed: {round(total_pages / infer_time, 3)} page/s"
            )
    finally:
        for context in doc_contexts:
            if not context['closed']:
                context['pdf_doc'].close()
                context['closed'] = True

    return (
        [context['middle_json'] for context in doc_contexts],
        [context['model_list'] for context in doc_contexts],
        ocr_enabled_list,
    )


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
