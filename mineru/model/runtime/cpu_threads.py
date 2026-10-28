# Copyright (c) Opendatalab. All rights reserved.
"""在首次 Torch 初始化关闭 CPU NNPACK，并限制后续工作线程的 CPU 算子线程。"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

_lock = threading.RLock()
_threads = threading.local()
_limit: int | None = None


def effective_cpu_count() -> int:
    """取 CPU 数、进程亲和性和 Linux cgroup 配额的最小有效值。"""
    count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)) or 1)
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            values = Path(quota_path).read_text().split()
            quota = int(values[0])
            period = int(Path(period_path).read_text()) if period_path else int(values[1])
            if quota > 0 and period > 0:
                count = min(count, max(1, math.ceil(quota / period)))
        except (OSError, ValueError, IndexError):
            continue
    return max(1, count)


def _positive_env(name: str) -> int | None:
    """忽略缺失、零和非法线程配置，沿既有配置优先级回退。"""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def initialize_torch_threads() -> int:
    """惰性初始化 Torch 的 CPU 默认后端与线程，不调高调用方已有的更低线程数。"""
    global _limit
    import torch

    with _lock:
        if _limit is None:
            # NNPACK 只参与 CPU 卷积；首次默认关闭，后续线程准备保留调用方显式设置。
            torch.backends.nnpack.set_flags(False)
            explicit = [_positive_env(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")]
            configured = (
                min(value for value in explicit if value is not None)
                if any(explicit)
                else (_positive_env("MINERU_CPU_NUM_THREADS") or _positive_env("MINERU_INTRA_OP_NUM_THREADS") or 8)
            )
            torch.init_num_threads()
            _limit = min(configured, effective_cpu_count(), torch.get_num_threads())
        if getattr(_threads, "limit", None) != _limit:
            torch.init_num_threads()
            current = torch.get_num_threads()
            if current > _limit:
                torch.set_num_threads(_limit)
            _threads.limit = _limit
        return min(_limit, torch.get_num_threads())


def prepare_torch_thread() -> None:
    """把已选线程上限传播到后续线程；纯 ONNX/Flash 路径不会触发 Torch 导入。"""
    if _limit is not None:
        initialize_torch_threads()


__all__ = ["effective_cpu_count", "initialize_torch_threads", "prepare_torch_thread"]
