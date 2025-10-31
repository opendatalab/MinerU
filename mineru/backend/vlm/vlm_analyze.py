# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import pypdfium2 as pdfium

from loguru import logger

from .utils import enable_custom_logits_processors, set_default_gpu_memory_utilization, set_default_batch_size
from .model_output_to_middle_json import result_to_middle_json,result_to_middle_json_split,merge_middle_json_results
from ...data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import load_images_from_pdf,load_images_from_pdf_split
from ...utils.config_reader import get_device

from ...utils.enum_class import ImageType
from ...utils.models_download_utils import auto_download_and_get_model_root_path

from mineru_vl_utils import MinerUClient
from packaging import version


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
            vllm_async_llm = None
            batch_size = kwargs.get("batch_size", 0)  # for transformers backend only
            max_concurrency = kwargs.get("max_concurrency", 100)  # for http-client backend only
            http_timeout = kwargs.get("http_timeout", 600)  # for http-client backend only
            # 从kwargs中移除这些参数，避免传递给不相关的初始化函数
            for param in ["batch_size", "max_concurrency", "http_timeout"]:
                if param in kwargs:
                    del kwargs[param]
            if backend in ['transformers', 'vllm-engine', "vllm-async-engine"] and not model_path:
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
                else:
                    if os.getenv('OMP_NUM_THREADS') is None:
                        os.environ["OMP_NUM_THREADS"] = "1"

                    if backend == "vllm-engine":
                        try:
                            import vllm
                            from mineru_vl_utils import MinerULogitsProcessor
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-engine backend.")
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_llm = vllm.LLM(**kwargs)
                    elif backend == "vllm-async-engine":
                        try:
                            from vllm.engine.arg_utils import AsyncEngineArgs
                            from vllm.v1.engine.async_llm import AsyncLLM
                            from mineru_vl_utils import MinerULogitsProcessor
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-async-engine backend.")
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
            self._models[key] = MinerUClient(
                backend=backend,
                model=model,
                processor=processor,
                vllm_llm=vllm_llm,
                vllm_async_llm=vllm_async_llm,
                server_url=server_url,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                http_timeout=http_timeout,
            )
            elapsed = round(time.time() - start_time, 2)
            logger.info(f"get {backend} predictor cost: {elapsed}s")
        return self._models[key]


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
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
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = predictor.batch_two_step_extract(images=images_pil_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")

    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results


def doc_analyze_split(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    chunk_size=10,
    **kwargs,
):
    print("================= mydebug: call doc_analyze_split ====================")
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)

    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    pdf_page_num = len(pdf_doc)
    
    middle_json_list = []  # 收集所有分块的middle_json
    all_results = []
    # 分片处理PDF
    for chunk_start in range(0, pdf_page_num, chunk_size):
        chunk_end = min(chunk_start + chunk_size, pdf_page_num)
        
        logger.info(f"Processing pages {chunk_start} to {chunk_end-1}")
        
        # 加载当前分片的图像
        images_list = load_images_from_pdf_split(
            pdf_bytes, 
            pdf_doc, 
            start_page_id=chunk_start, 
            end_page_id=chunk_end-1,  # end_page_id是包含的
            image_type=ImageType.PIL
        )
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
        
        # 处理当前分片
        results = predictor.batch_two_step_extract(images=images_pil_list)
        all_results.extend(results)
        # 转换为middle_json（分块版本）
        middle_json_chunk = result_to_middle_json_split(
            results, 
            images_list, 
            pdf_doc, 
            image_writer,
            page_offset=chunk_start  # 传递页面偏移量
        )
        middle_json_list.append(middle_json_chunk)
        
        logger.info(f"Finished processing pages {chunk_start} to {chunk_end-1}")
    
    # 合并所有分块的结果并进行跨页操作
    final_middle_json = merge_middle_json_results(middle_json_list)
    
    # 关闭pdf文档（在合并完成后关闭）
    pdf_doc.close()
    

async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
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
    # logger.info(f"load images cost: {load_images_time}, speed: {round(len(images_base64_list)/load_images_time, 3)} images/s")

    # infer_start = time.time()
    results = await predictor.aio_batch_two_step_extract(images=images_pil_list)
    # infer_time = round(time.time() - infer_start, 2)
    # logger.info(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")
    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results
