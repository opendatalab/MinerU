# Copyright (c) Opendatalab. All rights reserved.
"""ONNX Runtime CPU 会话与自动选择设备的表格会话。"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

import onnxruntime as ort
from loguru import logger

from .onnx_device import OnnxSelection, confirm_gpu_session, metacommands_enabled, resolve_onnx_device
from .windows_gpu import GpuAdapter
from .webgpu import WEBGPU_PROVIDER, WebGPUDevice, attach_webgpu_provider, execution_lock, webgpu_session_scope


class _SerializedSession:
    """Serialize native calls independently of host inference-lock settings."""

    def __init__(self, session: ort.InferenceSession, lock: Any) -> None:
        self._session = session
        self._lock = lock

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any], run_options: Any = None) -> list[Any]:
        with self._lock:
            return self._session.run(output_names, input_feed, run_options)

    def get_inputs(self) -> list[Any]:
        return self._session.get_inputs()

    def get_outputs(self) -> list[Any]:
        return self._session.get_outputs()

    def get_providers(self) -> list[str]:
        return self._session.get_providers()

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        return self._session.get_provider_options()

    def end_profiling(self) -> str:
        with self._lock:
            return self._session.end_profiling()


class DirectMLSession(_SerializedSession):
    """DML requires serialization within each session."""

    def __init__(self, session: ort.InferenceSession) -> None:
        super().__init__(session, threading.Lock())


class WebGPUSession(_SerializedSession):
    """WebGPU sessions share one process-wide Dawn runtime and execution lock."""

    def __init__(self, session: ort.InferenceSession) -> None:
        super().__init__(session, execution_lock())


def _cpu_fallback_options(source: ort.SessionOptions | None) -> ort.SessionOptions:
    """Preserve the runtime's options without copying a plugin provider attachment."""
    target = ort.SessionOptions()
    if source is not None:
        for name in (
            "intra_op_num_threads",
            "inter_op_num_threads",
            "enable_cpu_mem_arena",
            "enable_mem_pattern",
            "enable_mem_reuse",
            "enable_profiling",
            "profile_file_prefix",
            "execution_mode",
            "execution_order",
            "graph_optimization_level",
            "log_severity_level",
            "log_verbosity_level",
            "logid",
            "optimized_model_filepath",
            "use_deterministic_compute",
        ):
            setattr(target, name, getattr(source, name))
    return target


def get_op_num_threads(env_name: str) -> int:
    """读取 ONNX 算子线程数，缺失或非法时返回负一。"""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return -1
    try:
        value = int(raw_value)
    except ValueError:
        return -1
    return value if value > 0 else -1


def configure_ort_threads(
    opts: ort.SessionOptions,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
) -> None:
    """统一设置线程数：显式正数及已有会话配置优先，其次环境变量，最后回退为 4/1。"""
    for attr, configured, env_name, default in (
        ("intra_op_num_threads", intra_op_num_threads, "MINERU_INTRA_OP_NUM_THREADS", 4),
        ("inter_op_num_threads", inter_op_num_threads, "MINERU_INTER_OP_NUM_THREADS", 1),
    ):
        threads = configured if configured > 0 else getattr(opts, attr)
        if threads <= 0:
            threads = get_op_num_threads(env_name)
        setattr(opts, attr, threads if threads > 0 else default)


def ort_providers(device: str | None = None) -> list[tuple[str, dict[str, object]]]:
    """通用 ONNX 模型固定使用 CPU；表格工厂独立选择 CUDA provider。"""
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def create_ort_session(
    model_path: str,
    *,
    adapter: GpuAdapter | WebGPUDevice | None = None,
    model_kind: str = "",
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
    profile_prefix: str | None = None,
    sess_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession | DirectMLSession | WebGPUSession:
    """Create an explicit CPU/DML/WebGPU session without selection or config writes."""
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    configure_ort_threads(opts, intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads)
    providers = ort_providers()
    if isinstance(adapter, GpuAdapter):
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DmlExecutionProvider is unavailable")
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options: dict[str, object] = {"device_id": adapter.index}
        enabled = metacommands_enabled(adapter.vendor, model_kind)
        if enabled is not None:
            options["disable_metacommands"] = "false" if enabled else "true"
        providers = [("DmlExecutionProvider", options), *providers]
    if profile_prefix is not None:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    if isinstance(adapter, WebGPUDevice):
        with webgpu_session_scope(adapter):
            attach_webgpu_provider(opts, adapter)
            # Passing providers= here overrides (and removes) the plugin attached to opts.
            session = ort.InferenceSession(model_path, sess_options=opts)
            if not session.get_providers() or session.get_providers()[0] != WEBGPU_PROVIDER:
                raise RuntimeError("Requested WebGPU session fell back to CPU during initialization")
            session.disable_fallback()
            return WebGPUSession(session)
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    if adapter is not None:
        if session.get_providers()[0] != "DmlExecutionProvider":
            raise RuntimeError("Requested DML session fell back to CPU during initialization")
        session.disable_fallback()
        return DirectMLSession(session)
    return session


def ort_session(
    model_path: str,
    device: str | None = None,
    intra_op_num_threads: int = 0,
    *,
    model_kind: str = "",
    selection: OnnxSelection | None = None,
    sess_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession | DirectMLSession | WebGPUSession:
    """Route supported models through DML/ WebGPU; detection is GPU-eligible only on Linux."""
    eligible = model_kind in {"layout", "ocr_rec", "mfr", "unet"} or (model_kind == "ocr_det" and sys.platform == "linux")
    selection_threads = intra_op_num_threads
    if selection_threads <= 0 and sess_options is not None:
        selection_threads = sess_options.intra_op_num_threads
    route = OnnxSelection()
    if eligible:
        route = selection or resolve_onnx_device(
            intra_op_num_threads=selection_threads,
            inter_op_num_threads=sess_options.inter_op_num_threads if sess_options is not None else 0,
        )
    adapter = route.adapter if route.device == "gpu" else None
    if model_kind == "ocr_det" and not isinstance(adapter, WebGPUDevice):
        adapter = None
    fallback_opts = _cpu_fallback_options(sess_options) if isinstance(adapter, WebGPUDevice) else sess_options
    started = time.perf_counter()
    try:
        session = create_ort_session(
            model_path,
            adapter=adapter,
            model_kind=model_kind,
            intra_op_num_threads=intra_op_num_threads,
            sess_options=sess_options,
        )
    except Exception as error:
        if adapter is None:
            raise
        logger.warning("ONNX {} {} initialization failed; using CPU: {}", model_kind, route.backend, error)
        session = create_ort_session(model_path, intra_op_num_threads=intra_op_num_threads, sess_options=fallback_opts)
        adapter = None
    else:
        if adapter is not None:
            confirm_gpu_session(route)
    logger.info(
        "ONNX session ready: kind={}, providers={}, gpu={}, metacommands={}, init_s={:.3f}",
        model_kind,
        session.get_providers(),
        adapter,
        metacommands_enabled(adapter.vendor, model_kind) if isinstance(adapter, GpuAdapter) else None,
        time.perf_counter() - started,
    )
    return session


def table_ort_session(
    model_path: str,
    *,
    sess_options: ort.SessionOptions | None = None,
    small_backend: str | None = None,
    model_kind: str = "",
) -> ort.InferenceSession | DirectMLSession | WebGPUSession:
    """ONNX 栈的 UNet 复用统一设备选择；其他表格保留原有 CPU/CUDA 策略。"""
    from .device import resolve_small_model_backend

    onnx_backend = resolve_small_model_backend(small_backend) == "onnx"
    unified_unet = onnx_backend and model_kind == "unet"
    device = "cpu"
    if not unified_unet:
        device = os.getenv("MINERU_TABLE_DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("MINERU_TABLE_DEVICE must be 'auto', 'cpu' or 'cuda'")
    if onnx_backend:
        device = "cpu"
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    configure_ort_threads(opts)
    profile_dir = os.getenv("MINERU_ONNX_PROFILE_DIR")
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        opts.enable_profiling = True
        opts.profile_file_prefix = str(Path(profile_dir) / f"{Path(model_path).stem}-{os.getpid()}")
    if unified_unet:
        return ort_session(model_path, model_kind="unet", sess_options=opts)
    providers = ort_providers()
    use_cuda = False
    if device != "cpu":
        use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                *providers,
            ]
        elif device == "cuda":
            logger.warning("Table CUDA unavailable; falling back to CPU: {}", model_path)
    started = time.perf_counter()
    try:
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception as exc:
        if not use_cuda:
            raise
        logger.warning("Table CUDA initialization failed; falling back to CPU: {}: {}", model_path, exc)
        session = ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers())
    actual = session.get_providers()
    if use_cuda and "CUDAExecutionProvider" not in actual:
        logger.warning("Table requested CUDA but actual providers are {}: {}", actual, model_path)
    logger.info(
        "Table ONNX ready: model={}, requested={}, providers={}, init_s={:.3f}, pid={}",
        Path(model_path).name,
        device,
        actual,
        time.perf_counter() - started,
        os.getpid(),
    )
    return session


__all__ = [
    "DirectMLSession",
    "WebGPUSession",
    "configure_ort_threads",
    "create_ort_session",
    "get_op_num_threads",
    "ort_providers",
    "ort_session",
    "table_ort_session",
]
