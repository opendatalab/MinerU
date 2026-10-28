# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import atexit
import gc
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from typing import Any, Literal

from loguru import logger
from mineru_vl_utils import MinerUClient

from ..registry import MINERU_2_5_PRO_2605_1_2B, vlm_model_repo
from ..runtime.device import get_device
from ..runtime.platform import is_mac_os_version_supported
from ..runtime.closing import in_runtime_close_scope, runtime_close_scope
from ..runtime.ownership import RuntimeOwner, current_runtime_owner, runtime_initialization_scope
from .async_runtime import AsyncVlmPredictor
from .handles import RuntimeHandle
from .engine_utils import (
    enable_custom_logits_processors,
    mod_kwargs_by_device_type,
    set_default_batch_size,
    set_default_gpu_memory_utilization,
    set_lmdeploy_backend,
)


@dataclass(eq=False)
class _RuntimeInitialization:
    """保存同键初始化屏障，直到构造发布或失败清理完整结束。"""

    finished: threading.Event = field(default_factory=threading.Event)


_initializing_runtime: ContextVar[_RuntimeInitialization | None] = ContextVar("mineru_initializing_vlm", default=None)


class ModelSingleton:
    _instance = None
    _models: dict[tuple[object, ...], RuntimeHandle | AsyncVlmPredictor] = {}
    _closing: dict[tuple[object, ...], threading.Event] = {}
    _initializing: dict[tuple[object, ...], _RuntimeInitialization] = {}
    _lock = threading.RLock()
    _shutdown_registered = False

    def __new__(cls, *args: object, **kwargs: object) -> ModelSingleton:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        backend: Literal[
            "http-client",
            "llama-cpp-engine",
            "transformers",
            "mlx-engine",
            "lmdeploy-engine",
            "vllm-engine",
            "vllm-async-engine",
        ],
        model_path: str | None,
        server_url: str | None,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> RuntimeHandle | AsyncVlmPredictor:
        """先接纳应用操作，再沿原缓存键复用运行时；路径解析和构造都不持有缓存锁。"""
        with runtime_initialization_scope():
            if backend == "vllm-engine":
                backend = "vllm-async-engine"
            native_async = backend in {"http-client", "vllm-async-engine"}
            options = {"model_name": model_name, **kwargs}
            if backend == "vllm-async-engine":
                options["device"] = get_device()
                model_path = (
                    str(Path(model_path).expanduser().resolve()) if model_path else str(MINERU_2_5_PRO_2605_1_2B.ensure())
                )
            fingerprint = hashlib.sha256(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()
            cache_key = (backend, model_path, server_url, fingerprint)
            factory = partial(self._create_model, backend, model_path, server_url, model_name=model_name, **kwargs)
            return self._get_or_initialize(
                cache_key,
                factory,
                native_async=native_async,
                backend=backend,
                max_concurrency=kwargs.get("max_concurrency", 100),
            )

    def _retain(self, predictor: RuntimeHandle | AsyncVlmPredictor) -> None:
        """发布或复用前重新校验应用，防止关闭后的预热结果进入缓存。"""
        owner = current_runtime_owner()
        if owner is None:
            predictor.pinned = True
        else:
            predictor.owners.add(owner)

    def _get_or_initialize(
        self,
        key: tuple[object, ...],
        factory: Callable[[], RuntimeHandle],
        *,
        native_async: bool,
        backend: str,
        max_concurrency: int,
    ) -> RuntimeHandle | AsyncVlmPredictor:
        """同键共享初始化或关闭屏障，外部构造和关闭动作始终在全局锁之外。"""
        while True:
            detached = []
            reservation = None
            with self._lock:
                current_runtime_owner()
                closing = self._closing.get(key)
                initializing = self._initializing.get(key)
                cached = self._models.get(key)
                if in_runtime_close_scope() and (
                    closing is not None or initializing is not None or (cached is not None and cached.is_closed)
                ):
                    raise RuntimeError("Cannot acquire a closing VLM runtime from a close callback")
                if initializing is not None and initializing is _initializing_runtime.get():
                    raise RuntimeError("Cannot recursively acquire a VLM runtime during its initialization")
                if closing is not None:
                    pending = closing
                elif initializing is not None:
                    pending = initializing.finished
                elif cached is not None and cached.is_closed:
                    pending = None
                    detached.append(self._detach(key, cached))
                elif cached is not None:
                    self._retain(cached)
                    return cached
                else:
                    pending = None
                    reservation = _RuntimeInitialization()
                    self._initializing[key] = reservation
                    if not self._shutdown_registered:
                        atexit.register(shutdown_cached_models)
                        type(self)._shutdown_registered = True
            if pending is not None:
                pending.wait()
            elif detached:
                self._close_detached(detached)
            elif reservation is not None:
                return self._initialize_model(
                    key, reservation, factory, native_async=native_async, backend=backend, max_concurrency=max_concurrency
                )

    def _initialize_model(
        self,
        key: tuple[object, ...],
        reservation: _RuntimeInitialization,
        factory: Callable[[], RuntimeHandle],
        *,
        native_async: bool,
        backend: str,
        max_concurrency: int,
    ) -> RuntimeHandle | AsyncVlmPredictor:
        """锁外构造并等待就绪，失效应用或失败构造完整清理后才解除同键屏障。"""
        predictor: RuntimeHandle | AsyncVlmPredictor | None = None
        token = _initializing_runtime.set(reservation)
        try:
            if native_async:
                factory_context = copy_context()
                predictor = AsyncVlmPredictor(
                    partial(factory_context.run, factory), backend=backend, max_concurrency=max_concurrency
                )
                predictor.wait_ready()
            else:
                predictor = factory()
            with self._lock:
                self._retain(predictor)
                self._models[key] = predictor
            return predictor
        except BaseException as initialization_error:
            if predictor is not None:
                try:
                    with runtime_close_scope():
                        predictor.shutdown()
                except BaseException as cleanup_error:
                    raise initialization_error from cleanup_error
            raise
        finally:
            _initializing_runtime.reset(token)
            with self._lock:
                self._initializing.pop(key, None)
                reservation.finished.set()

    def _detach(
        self, key: tuple[object, ...], predictor: RuntimeHandle | AsyncVlmPredictor
    ) -> tuple[tuple[object, ...], RuntimeHandle | AsyncVlmPredictor, threading.Event]:
        """在缓存锁内撤销条目并保留关闭屏障，防止相同权重在关闭完成前重新创建。"""
        del self._models[key]
        finished = threading.Event()
        self._closing[key] = finished
        return key, predictor, finished

    def _close_detached(
        self, entries: list[tuple[tuple[object, ...], RuntimeHandle | AsyncVlmPredictor, threading.Event]]
    ) -> None:
        """锁外逐个关闭已撤销条目，某个后端失败也释放其余条目和等待屏障。"""
        first_error: BaseException | None = None
        for key, predictor, finished in entries:
            try:
                with runtime_close_scope():
                    predictor.shutdown()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                with self._lock:
                    self._closing.pop(key, None)
                    finished.set()
        if first_error is not None:
            raise first_error

    def _create_model(
        self,
        backend: str,
        model_path: str | None,
        server_url: str | None,
        *,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> RuntimeHandle:
        """只构造一个底层客户端，不操作缓存；原生异步引擎由所属运行时调用。"""
        start_time = time.time()
        model = None
        processor = None
        vllm_llm = None
        lmdeploy_engine = None
        vllm_async_llm = None
        close_engine = None
        llama_cpp_engine = None
        batch_size = kwargs.get("batch_size", 0)  # 本地引擎批次大小；MLX 的 0 由客户端解析为保守默认值。
        max_concurrency = kwargs.get("max_concurrency", 100)  # 原生异步运行时在所有文档间共用该额度。
        http_timeout = kwargs.get("http_timeout", 600)  # for http-client backend only
        server_headers = kwargs.get("server_headers", None)  # for http-client backend only
        max_retries = kwargs.get("max_retries", 3)  # for http-client backend only
        retry_backoff_factor = kwargs.get("retry_backoff_factor", 0.5)  # for http-client backend only
        # 从kwargs中移除这些参数，避免传递给不相关的初始化函数
        for param in [
            "batch_size",
            "max_concurrency",
            "http_timeout",
            "server_headers",
            "max_retries",
            "retry_backoff_factor",
        ]:
            if param in kwargs:
                del kwargs[param]
        if backend not in ["http-client", "llama-cpp-engine"] and not model_path:
            model_path = str(MINERU_2_5_PRO_2605_1_2B.ensure())

        if backend == "llama-cpp-engine":
            # The GGUF repo carries two files: the main model and its
            # multi-modal projector (mmproj). ModelRepo.paths is
            # {"main": "...", "mmproj": "..."} — resolve each to its
            # absolute path under model_dir and hand both to Engine.
            repo = vlm_model_repo("llama-cpp")
            model_dir = Path(model_path).expanduser() if model_path else repo.ensure()
            model_gguf = model_dir / repo.paths["main"]
            mmproj_gguf = model_dir / repo.paths["mmproj"]

            # Engine-specific kwargs are pulled out of **kwargs so they
            # are not forwarded to MinerUClient (which doesn't accept
            # them). Defaults mirror Engine's own constructor defaults.
            engine_kwargs = {}
            for key in ("n_ctx_seq", "n_gpu_layers", "n_parallel", "verbosity", "n_threads"):
                if key in kwargs:
                    engine_kwargs[key] = kwargs.pop(key)

            from ..runtime.llama_device import llama_engine_scope, resolve_llama_device

            selection = resolve_llama_device(n_gpu_layers=engine_kwargs.get("n_gpu_layers"))
            if selection.device == "cpu":
                engine_kwargs["n_gpu_layers"] = 0
            try:
                from mineru_llama_cpp import Engine
            except ImportError:
                raise ImportError("Please install mineru-llama-cpp to use the llama-cpp-engine backend.")
            with llama_engine_scope(selection):
                llama_cpp_engine = Engine(model_gguf, mmproj_gguf, **engine_kwargs)
            close_engine = llama_cpp_engine.close

        elif backend == "transformers":
            from ..runtime.cpu_threads import initialize_torch_threads

            initialize_torch_threads()
            try:
                from mineru_vl_utils.transformers_loading import (
                    load_transformers_model,
                    load_transformers_processor,
                )

                model = load_transformers_model(model_path, device_map={"": get_device()})
                processor = load_transformers_processor(model_path)
            except ImportError as exc:
                raise ImportError("Please install transformers to use the transformers backend.") from exc
            if batch_size == 0:
                batch_size = set_default_batch_size()
        elif backend == "mlx-engine":
            mlx_supported = is_mac_os_version_supported("14.0")
            if not mlx_supported:
                raise EnvironmentError("mlx-engine backend is only supported on macOS 14.0+ with Apple Silicon.")
            from mineru_vl_utils.mlx_compat import load_mlx_model

            model, processor = load_mlx_model(model_path)
        elif backend != "http-client":
            if os.getenv("OMP_NUM_THREADS") is None:
                os.environ["OMP_NUM_THREADS"] = "1"

            if backend == "vllm-async-engine":
                try:
                    from vllm.config import CompilationConfig
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                kwargs = mod_kwargs_by_device_type(kwargs, vllm_mode="async_engine")

                if "compilation_config" in kwargs:
                    if isinstance(kwargs["compilation_config"], dict):
                        # 如果是字典，转换为 CompilationConfig 对象
                        kwargs["compilation_config"] = CompilationConfig(**kwargs["compilation_config"])
                    elif isinstance(kwargs["compilation_config"], str):
                        # 如果是 JSON 字符串，先解析再转换
                        try:
                            config_dict = json.loads(kwargs["compilation_config"])
                            kwargs["compilation_config"] = CompilationConfig(**config_dict)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse compilation_config: {kwargs['compilation_config']}, error: {e}")
                            del kwargs["compilation_config"]
                if "gpu_memory_utilization" not in kwargs:
                    kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                if "model" not in kwargs:
                    kwargs["model"] = model_path
                if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                    from mineru_vl_utils import MinerULogitsProcessor

                    kwargs["logits_processors"] = [MinerULogitsProcessor]
                # 使用kwargs为 vllm初始化参数
                vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
                close_engine = vllm_async_llm.shutdown
            elif backend == "lmdeploy-engine":
                try:
                    from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, pipeline
                except ImportError:
                    raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")
                if "cache_max_entry_count" not in kwargs:
                    kwargs["cache_max_entry_count"] = set_default_gpu_memory_utilization(backend="lmdeploy")

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

                log_level = "ERROR"
                from lmdeploy.utils import get_logger

                lm_logger = get_logger("lmdeploy")
                lm_logger.setLevel(log_level)
                if os.getenv("TM_LOG_LEVEL") is None:
                    os.environ["TM_LOG_LEVEL"] = log_level

                lmdeploy_engine = pipeline(
                    model_path,
                    backend_config=backend_config,
                    log_level=log_level,
                )
                close_engine = lmdeploy_engine.close
        try:
            predictor = MinerUClient(
                backend=backend,
                model_name=model_name,
                model=model,
                processor=processor,
                lmdeploy_engine=lmdeploy_engine,
                vllm_llm=vllm_llm,
                vllm_async_llm=vllm_async_llm,
                llama_cpp_engine=llama_cpp_engine,
                server_url=server_url,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                http_timeout=http_timeout,
                server_headers=server_headers,
                max_retries=max_retries,
                retry_backoff_factor=retry_backoff_factor,
                enable_table_formula_eq_wrap=True,
                image_analysis=True,
                enable_cross_page_table_merge=False,
            )
        except BaseException as initialization_error:
            # 客户端构造失败后，宿主仍负责关闭已经成功创建的引擎。
            if close_engine is not None:
                try:
                    with runtime_close_scope():
                        close_engine()
                except BaseException as cleanup_error:
                    raise initialization_error from cleanup_error
            raise
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"get {backend} predictor cost: {elapsed}s")
        return RuntimeHandle(predictor, backend=backend, close_engine=close_engine)

    def release_owner(self, owner: RuntimeOwner) -> None:
        """排空该应用文档后撤销全部后端租约，保留其它应用和独立解析持有的实例。"""
        owner.close()
        with self._lock:
            closing = []
            for key, predictor in tuple(self._models.items()):
                predictor.owners.discard(owner)
                if not predictor.owners and not predictor.pinned:
                    closing.append(self._detach(key, predictor))
        self._close_detached(closing)

    def shutdown(self) -> None:
        """等待已开始的初始化收尾，撤销缓存后在锁外排空运行时。"""
        while True:
            with self._lock:
                pending = tuple(self._initializing.values())
                if not pending:
                    closing = [self._detach(key, predictor) for key, predictor in tuple(self._models.items())]
                    break
                if _initializing_runtime.get() in pending or in_runtime_close_scope():
                    raise RuntimeError("Cannot wait for VLM initialization from its cleanup callback")
            for item in pending:
                item.finished.wait()
        try:
            self._close_detached(closing)
        finally:
            gc.collect()


async def _get_model_async(
    backend: Literal[
        "http-client",
        "llama-cpp-engine",
        "transformers",
        "mlx-engine",
        "lmdeploy-engine",
        "vllm-engine",
        "vllm-async-engine",
    ],
    model_path: str | None,
    server_url: str | None,
    **kwargs: Any,
) -> RuntimeHandle | AsyncVlmPredictor:
    """模型初始化被取消时等待后台加载结束，防止清理后又写入迟到的模型句柄。"""
    from mineru_vl_utils.vlm_client.utils import run_in_thread_until_complete

    return await run_in_thread_until_complete(
        ModelSingleton().get_model,
        backend,
        model_path,
        server_url,
        **kwargs,
    )


def shutdown_cached_models() -> None:
    """进程结束时关闭包含独立解析固定租约在内的全部 VLM 运行时。"""
    ModelSingleton().shutdown()


__all__ = ["ModelSingleton", "shutdown_cached_models"]
