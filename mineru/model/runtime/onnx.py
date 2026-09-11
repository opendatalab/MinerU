# Copyright (c) Opendatalab. All rights reserved.
"""ONNX Runtime CPU 会话与线程配置。"""

import os

import onnxruntime as ort


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


__all__ = ["get_op_num_threads", "ort_providers", "ort_session"]
