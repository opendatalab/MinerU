"""按需记录阶段墙钟耗时，不同步 CUDA，也不改变推理结果。"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    """仅在诊断开关开启时记录完整阶段耗时，包含惰性初始化与 CPU 准备工作。"""
    enabled = os.getenv("MINERU_PROFILE_STAGES", "").lower() in {"1", "true", "yes"}
    started = time.perf_counter() if enabled else 0.0
    try:
        yield
    finally:
        if enabled:
            logger.info("MinerU stage: name={}, elapsed_s={:.6f}, pid={}", stage, time.perf_counter() - started, os.getpid())


__all__ = ["stage_timer"]
