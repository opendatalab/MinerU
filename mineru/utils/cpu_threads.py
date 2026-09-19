# Copyright (c) Opendatalab. All rights reserved.
"""Torch CPU 线程上限：避免长驻服务的 OpenMP 线程随线程池增长。"""

from __future__ import annotations

import os

# 部署方可通过这些环境变量固定原生数学库（libgomp / MKL / OpenBLAS）的 CPU
# 线程池大小。只要其中任意一个已被设置，就不能再覆盖运维方的选择。
CPU_THREAD_LIMIT_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)

# 未显式配置线程数时使用的上限。
# libgomp 会为每个进入过并行区的 master 线程常驻一支 ``num_threads`` 大小的
# worker team，而 ``asyncio.to_thread`` 所用 ``ThreadPoolExecutor`` 的 worker
# 生命周期等同于整个进程。没有上限时，每个碰过 torch CPU 算子的池 worker 都会
# 永久占用一支按整机规格创建的 team，长驻服务的 OS 线程数因而向
# ``pool_size * os.cpu_count()`` 增长且空闲时不回收。见 issue #5472。
DEFAULT_CPU_THREAD_LIMIT = 8


def get_op_num_threads(env_name: str) -> int:
    """读取线程数环境变量，缺失或非法时返回负一。"""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return -1
    try:
        value = int(raw_value)
    except ValueError:
        return -1
    return value if value > 0 else -1


def cpu_thread_limit_already_configured() -> bool:
    """部署方是否已经固定了原生 CPU 线程池。"""
    return any(os.getenv(env_name) is not None for env_name in CPU_THREAD_LIMIT_ENV_NAMES)


def resolve_cpu_thread_limit(
    default_limit: int = DEFAULT_CPU_THREAD_LIMIT,
    cpu_count: int | None = None,
) -> int:
    """解析单个 torch CPU 算子可用的线程数。

    ``MINERU_CPU_NUM_THREADS`` 优先，其次是 ``MINERU_INTRA_OP_NUM_THREADS``，
    使同一个变量能同时调节 ONNX 与 torch 两条 CPU 路径。非法或非正值回退到
    ``default_limit``，结果按 ``mineru.model.runtime.onnx`` 的同样口径夹在
    ``[1, cpu_count]`` 区间内。
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1

    configured = get_op_num_threads("MINERU_CPU_NUM_THREADS")
    if configured == -1:
        configured = get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
    if configured == -1:
        configured = default_limit

    return max(1, min(configured, max(1, cpu_count)))


def apply_cpu_thread_limit(
    default_limit: int = DEFAULT_CPU_THREAD_LIMIT,
    cpu_count: int | None = None,
) -> int | None:
    """在部署方未配置时限制每线程的 torch CPU 池大小。

    通过 ``torch.set_num_threads`` 施加，而不是导出 ``OMP_NUM_THREADS``，
    这样 VLM 引擎仍能保留自己的 ``OMP_NUM_THREADS=1`` 默认值（它们只在该变量
    仍未设置时才写入）。

    :return: 实际施加的上限；若因部署方已固定线程池、torch 未安装或当前值
        本就更低而未做改动，则返回 ``None``。
    """
    if cpu_thread_limit_already_configured():
        return None

    try:
        import torch
    except ImportError:
        return None

    limit = resolve_cpu_thread_limit(default_limit=default_limit, cpu_count=cpu_count)
    if torch.get_num_threads() <= limit:
        return None

    torch.set_num_threads(limit)
    return limit


__all__ = [
    "CPU_THREAD_LIMIT_ENV_NAMES",
    "DEFAULT_CPU_THREAD_LIMIT",
    "apply_cpu_thread_limit",
    "cpu_thread_limit_already_configured",
    "get_op_num_threads",
    "resolve_cpu_thread_limit",
]
