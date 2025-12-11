#  Copyright (c) Opendatalab. All rights reserved.
import os
import time

from PIL import Image
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from loguru import logger
from mineru_vl_utils import MinerUClient
from packaging import version

from mineru.backend.hybrid.utils import set_default_batch_size, set_default_gpu_memory_utilization, \
    enable_custom_logits_processors, set_lmdeploy_backend
from mineru.utils.check_sys_env import is_mac_os_version_supported
from mineru.utils.config_reader import get_device
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


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
