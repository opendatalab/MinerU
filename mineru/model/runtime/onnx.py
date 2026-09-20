# Copyright (c) Opendatalab. All rights reserved.
"""ONNX Runtime CPU 会话与自动选择设备的表格会话。"""

import os
import time
from pathlib import Path

import onnxruntime as ort
from loguru import logger


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


def ort_session(model_path: str, device: str | None = None, intra_op_num_threads: int = 0) -> ort.InferenceSession:
    """创建 CPU 会话；显式线程数优先，其次环境配置，默认使用 4/1 线程。"""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    configure_ort_threads(opts, intra_op_num_threads=intra_op_num_threads)
    return ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers(device))


def table_ort_session(model_path: str, *, sess_options: ort.SessionOptions | None = None) -> ort.InferenceSession:
    """表格默认按可用 provider 选择设备；支持固定 CPU/CUDA，并明确记录 CUDA 失败回退。"""
    device = os.getenv("MINERU_TABLE_DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("MINERU_TABLE_DEVICE must be 'auto', 'cpu' or 'cuda'")
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    configure_ort_threads(opts)
    profile_dir = os.getenv("MINERU_ONNX_PROFILE_DIR")
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        opts.enable_profiling = True
        opts.profile_file_prefix = str(Path(profile_dir) / f"{Path(model_path).stem}-{os.getpid()}")
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


__all__ = ["configure_ort_threads", "get_op_num_threads", "ort_providers", "ort_session", "table_ort_session"]
