# Copyright (c) Opendatalab. All rights reserved.
"""ONNX Runtime CPU 会话与可选的表格 CUDA 会话。"""

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


def ort_providers(device: str | None = None) -> list[tuple[str, dict[str, object]]]:
    """所有 ONNX 模型固定使用 CPU，宿主 Torch 设备不参与 provider 选择。"""
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def ort_session(model_path: str, device: str | None = None, intra_op_num_threads: int = 0) -> ort.InferenceSession:
    """创建 CPU 会话；显式线程数优先，否则使用统一环境配置。"""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    intra_threads = intra_op_num_threads or get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
    inter_threads = get_op_num_threads("MINERU_INTER_OP_NUM_THREADS")
    if intra_threads > 0:
        opts.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        opts.inter_op_num_threads = inter_threads
    return ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers(device))


def table_ort_session(model_path: str, *, sess_options: ort.SessionOptions | None = None) -> ort.InferenceSession:
    """表格独立选择 CPU/CUDA；创建失败或 provider 静默降级时明确记录并保留 CPU 回退。"""
    device = os.getenv("MINERU_TABLE_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("MINERU_TABLE_DEVICE must be 'cpu' or 'cuda'")
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    for attr, env_name in (
        ("intra_op_num_threads", "MINERU_INTRA_OP_NUM_THREADS"),
        ("inter_op_num_threads", "MINERU_INTER_OP_NUM_THREADS"),
    ):
        threads = get_op_num_threads(env_name)
        if threads > 0:
            setattr(opts, attr, threads)
    profile_dir = os.getenv("MINERU_ONNX_PROFILE_DIR")
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        opts.enable_profiling = True
        opts.profile_file_prefix = str(Path(profile_dir) / f"{Path(model_path).stem}-{os.getpid()}")
    providers = ort_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
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
        else:
            logger.warning("Table CUDA unavailable; falling back to CPU: {}", model_path)
    started = time.perf_counter()
    try:
        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception as exc:
        if device != "cuda" or providers == ort_providers():
            raise
        logger.warning("Table CUDA initialization failed; falling back to CPU: {}: {}", model_path, exc)
        session = ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers())
    actual = session.get_providers()
    if device == "cuda" and "CUDAExecutionProvider" not in actual:
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


__all__ = ["get_op_num_threads", "ort_providers", "ort_session", "table_ort_session"]
