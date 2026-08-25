# Copyright (c) Opendatalab. All rights reserved.
"""ONNX Runtime provider、会话与线程配置。"""

import os

import onnxruntime as ort

AVAILABLE_PROVIDERS: list[str] = ort.get_available_providers()


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
    """根据 device 选择 onnxruntime providers。"""
    norm = (device or "").lower().split(":", 1)[0]
    if norm != "cpu" and "CUDAExecutionProvider" in AVAILABLE_PROVIDERS:
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]
    # if "CoreMLExecutionProvider" in AVAILABLE_PROVIDERS:
    #     return [
    #         ("CoreMLExecutionProvider", {}),
    #         ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    #     ]
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def ort_session(model_path: str, device: str | None = None, intra_op_num_threads: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_num_threads > 0:
        opts.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(model_path, sess_options=opts, providers=ort_providers(device))


__all__ = ["get_op_num_threads", "ort_providers", "ort_session"]
