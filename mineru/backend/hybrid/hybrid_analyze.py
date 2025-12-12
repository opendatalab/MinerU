#  Copyright (c) Opendatalab. All rights reserved.
import os
import time

import cv2
from PIL import Image
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from loguru import logger
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import BlockType
from packaging import version

from mineru.backend.hybrid.model_output_to_middle_json import result_to_middle_json
from mineru.backend.hybrid.utils import set_default_batch_size, set_default_gpu_memory_utilization, \
    enable_custom_logits_processors, set_lmdeploy_backend
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.check_sys_env import is_mac_os_version_supported
from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ImageType
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf

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
        backend: str,
        model_path: str | None,
        server_url: str | None,
        **kwargs,
    ) -> MinerUClient:
        key = (backend, model_path, server_url)
        if key not in self._models:
            start_time = time.time()
            model = None
            processor = None
            vllm_llm = None
            lmdeploy_engine = None
            vllm_async_llm = None
            batch_size = kwargs.get("batch_size", 0)  # for transformers backend only
            max_concurrency = kwargs.get("max_concurrency", 100)  # for http-client backend only
            http_timeout = kwargs.get("http_timeout", 600)  # for http-client backend only
            server_headers = kwargs.get("server_headers", None)  # for http-client backend only
            max_retries = kwargs.get("max_retries", 3)  # for http-client backend only
            retry_backoff_factor = kwargs.get("retry_backoff_factor", 0.5)  # for http-client backend only
            # 从kwargs中移除这些参数，避免传递给不相关的初始化函数
            for param in ["batch_size", "max_concurrency", "http_timeout", "server_headers", "max_retries", "retry_backoff_factor"]:
                if param in kwargs:
                    del kwargs[param]
            if backend not in ["http-client"] and not model_path:
                model_path = auto_download_and_get_model_root_path("/","vlm")
            if backend == "transformers":
                try:
                    from transformers import (
                        AutoProcessor,
                        Qwen2VLForConditionalGeneration,
                    )
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")

                if version.parse(transformers_version) >= version.parse("4.56.0"):
                    dtype_key = "dtype"
                else:
                    dtype_key = "torch_dtype"
                device = get_device()
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map={"": device},
                    **{dtype_key: "auto"},  # type: ignore
                )
                processor = AutoProcessor.from_pretrained(
                    model_path,
                    use_fast=True,
                )
                if batch_size == 0:
                    batch_size = set_default_batch_size()
            elif backend == "mlx-engine":
                mlx_supported = is_mac_os_version_supported()
                if not mlx_supported:
                    raise EnvironmentError("mlx-engine backend is only supported on macOS 13.5+ with Apple Silicon.")
                try:
                    from mlx_vlm import load as mlx_load
                except ImportError:
                    raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")
                model, processor = mlx_load(model_path)
            else:
                if os.getenv('OMP_NUM_THREADS') is None:
                    os.environ["OMP_NUM_THREADS"] = "1"

                if backend == "vllm-engine":
                    try:
                        import vllm
                    except ImportError:
                        raise ImportError("Please install vllm to use the vllm-engine backend.")
                    if "gpu_memory_utilization" not in kwargs:
                        kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                    if "model" not in kwargs:
                        kwargs["model"] = model_path
                    if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                        from mineru_vl_utils import MinerULogitsProcessor
                        kwargs["logits_processors"] = [MinerULogitsProcessor]
                    # 使用kwargs为 vllm初始化参数
                    vllm_llm = vllm.LLM(**kwargs)
                elif backend == "vllm-async-engine":
                    try:
                        from vllm.engine.arg_utils import AsyncEngineArgs
                        from vllm.v1.engine.async_llm import AsyncLLM
                    except ImportError:
                        raise ImportError("Please install vllm to use the vllm-async-engine backend.")
                    if "gpu_memory_utilization" not in kwargs:
                        kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                    if "model" not in kwargs:
                        kwargs["model"] = model_path
                    if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                        from mineru_vl_utils import MinerULogitsProcessor
                        kwargs["logits_processors"] = [MinerULogitsProcessor]
                    # 使用kwargs为 vllm初始化参数
                    vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
                elif backend == "lmdeploy-engine":
                    try:
                        from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
                        from lmdeploy.serve.vl_async_engine import VLAsyncEngine
                    except ImportError:
                        raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")
                    if "cache_max_entry_count" not in kwargs:
                        kwargs["cache_max_entry_count"] = 0.5

                    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
                    if device_type == "":
                        if "lmdeploy_device" in kwargs:
                            device_type = kwargs.pop("lmdeploy_device")
                            if device_type not in ["cuda", "ascend", "maca", "camb"]:
                                raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
                        else:
                            device_type = "cuda"
                    lm_backend = os.getenv("MINERU_LMDEPLOY_BACKEND", "")
                    if lm_backend == "":
                        if "lmdeploy_backend" in kwargs:
                            lm_backend = kwargs.pop("lmdeploy_backend")
                            if lm_backend not in ["pytorch", "turbomind"]:
                                raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")
                        else:
                            lm_backend = set_lmdeploy_backend(device_type)
                    logger.info(f"lmdeploy device is: {device_type}, lmdeploy backend is: {lm_backend}")

                    if lm_backend == "pytorch":
                        kwargs["device_type"] = device_type
                        backend_config = PytorchEngineConfig(**kwargs)
                    elif lm_backend == "turbomind":
                        backend_config = TurbomindEngineConfig(**kwargs)
                    else:
                        raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")

                    log_level = 'ERROR'
                    from lmdeploy.utils import get_logger
                    lm_logger = get_logger('lmdeploy')
                    lm_logger.setLevel(log_level)
                    if os.getenv('TM_LOG_LEVEL') is None:
                        os.environ['TM_LOG_LEVEL'] = log_level

                    lmdeploy_engine = VLAsyncEngine(
                        model_path,
                        backend=lm_backend,
                        backend_config=backend_config,
                    )
            self._models[key] = MinerUClient(
                backend=backend,
                model=model,
                processor=processor,
                lmdeploy_engine=lmdeploy_engine,
                vllm_llm=vllm_llm,
                vllm_async_llm=vllm_async_llm,
                server_url=server_url,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                http_timeout=http_timeout,
                server_headers=server_headers,
                max_retries=max_retries,
                retry_backoff_factor=retry_backoff_factor,
            )
            elapsed = round(time.time() - start_time, 2)
            logger.info(f"get {backend} predictor cost: {elapsed}s")
        return self._models[key]


not_extract_list = [
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.TABLE_CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.CODE_CAPTION,
]

def ocr_classify(pdf_bytes, parse_method: str = 'auto',) -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == 'auto':
        if classify(pdf_bytes) == 'ocr':
            _ocr_enable = True
    elif parse_method == 'ocr':
        _ocr_enable = True
    return _ocr_enable

def ocr_det_batch_setting():
    device = get_device()
    # 检测torch的版本号
    import torch
    from packaging import version
    if version.parse(torch.__version__) >= version.parse("2.8.0") or str(device).startswith('mps'):
        enable_ocr_det_batch = False
    else:
        enable_ocr_det_batch = True
    return enable_ocr_det_batch

def ocr_det(enable_ocr_det_batch):
    # OCR det
    if enable_ocr_det_batch:
        # 批处理模式 - 按语言和分辨率分组
        # 收集所有需要OCR检测的裁剪图像
        all_cropped_images_info = []

        for ocr_res_list_dict in ocr_res_list_all_page:
            _lang = ocr_res_list_dict['lang']

            for res in ocr_res_list_dict['ocr_res_list']:
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                )

                # BGR转换
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                all_cropped_images_info.append((
                    bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang
                ))

        # 按语言分组
        lang_groups = defaultdict(list)
        for crop_info in all_cropped_images_info:
            lang = crop_info[5]
            lang_groups[lang].append(crop_info)

        # 对每种语言按分辨率分组并批处理
        for lang, lang_crop_list in lang_groups.items():
            if not lang_crop_list:
                continue

            # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

            # 获取OCR模型
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                det_db_box_thresh=0.3,
                lang=lang
            )

            # 按分辨率分组并同时完成padding
            # RESOLUTION_GROUP_STRIDE = 32
            RESOLUTION_GROUP_STRIDE = 64

            resolution_groups = defaultdict(list)
            for crop_info in lang_crop_list:
                cropped_img = crop_info[0]
                h, w = cropped_img.shape[:2]
                # 直接计算目标尺寸并用作分组键
                target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                group_key = (target_h, target_w)
                resolution_groups[group_key].append(crop_info)

            # 对每个分辨率组进行批处理
            for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                # 对所有图像进行padding到统一尺寸
                batch_images = []
                for crop_info in group_crops:
                    img = crop_info[0]
                    h, w = img.shape[:2]
                    # 创建目标尺寸的白色背景
                    padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                    padded_img[:h, :w] = img
                    batch_images.append(padded_img)

                # 批处理检测
                det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

                # 处理批处理结果
                for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                    bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info

                    if dt_boxes is not None and len(dt_boxes) > 0:
                        # 处理检测框
                        dt_boxes_sorted = sorted_boxes(dt_boxes)
                        dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                        # 根据公式位置更新检测框
                        dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                          if dt_boxes_merged and adjusted_mfdetrec_res
                                          else dt_boxes_merged)

                        if dt_boxes_final:
                            ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                            ocr_result_list = get_ocr_result_list(
                                ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang
                            )
                            ocr_res_list_dict['layout_res'].extend(ocr_result_list)

    else:
        # 原始单张处理模式
        for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
            # Process each area that requires OCR processing
            _lang = ocr_res_list_dict['lang']
            # Get OCR results for this language's images
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                ocr_show_log=False,
                det_db_box_thresh=0.3,
                lang=_lang
            )
            for res in ocr_res_list_dict['ocr_res_list']:
                new_image, useful_list = crop_img(
                    res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                )
                # OCR-det
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = ocr_model.ocr(
                    bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang
                    )

                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)

def doc_analyze_core(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = 'auto',
    language: str = 'ch',
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    # load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    # load_images_time = round(time.time() - load_images_start, 2)
    # logger.debug(f"load images cost: {load_images_time}, speed: {round(len(images_pil_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)

    # infer_start = time.time()
    if _ocr_enable and language in ["ch", "chinese_cht", "ch_lite", "ch_server", "en"] and inline_formula_enable:
        results = predictor.batch_two_step_extract(images=images_pil_list)
    else:
        results = predictor.batch_two_step_extract(images=images_pil_list, not_extract_list=not_extract_list)
        # 遍历results，对文本块截图交由OCR识别
        # 根据_ocr_enable决定ocr只开det还是det+rec
        # 根据inline_formula_enable决定是使用mfd和ocr结合的方式，还是纯ocr方式

    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")
    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = 'auto',
    language: str = 'ch',
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    return doc_analyze_core(
        pdf_bytes,
        image_writer,
        predictor,
        backend,
        parse_method,
        language,
        inline_formula_enable,
        model_path,
        server_url,
        **kwargs,
    )


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = 'auto',
    language: str = 'ch',
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):

    return doc_analyze_core(
        pdf_bytes,
        image_writer,
        predictor,
        backend,
        parse_method,
        language,
        inline_formula_enable,
        model_path,
        server_url,
        **kwargs,
    )


if __name__ == "__main__":
    kwargs = {}
    kwargs["cache_max_entry_count"] = 0.5
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    model_path = auto_download_and_get_model_root_path("/", "vlm")

    log_level = 'ERROR'
    from lmdeploy.utils import get_logger

    lm_logger = get_logger('lmdeploy')
    lm_logger.setLevel(log_level)
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level

    lmdeploy_engine = VLAsyncEngine(
        model_path,
        backend="turbomind",
        backend_config=TurbomindEngineConfig(**kwargs),
    )

    client = MinerUClient(
        backend="lmdeploy-engine",
        lmdeploy_engine=lmdeploy_engine,
    )
    not_extract_list = ["text", "title"]
    image = Image.open(r"C:\Users\zhaoxiaomeng\Downloads\ch_hand_writer.jpg")
    def run_sync():
        extracted_blocks = client.two_step_extract(image=image, not_extract_list=not_extract_list)
        return extracted_blocks
    async def run_async():
        extracted_blocks = await client.aio_two_step_extract(image=image, not_extract_list=not_extract_list)
        return extracted_blocks

    extracted_blocks = run_sync()
    print(f"extracted_blocks: {extracted_blocks}")

    import asyncio
    async_extracted_blocks = asyncio.run(run_async())
    print(f"async_extracted_blocks: {async_extracted_blocks}")
