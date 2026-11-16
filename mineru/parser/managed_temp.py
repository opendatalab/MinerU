"""为托管服务分配有清理所有权且适合 Unix IPC 的临时目录。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 为 UUID socket（37 字节）和 multiprocessing 的 pymp/sock 名称预留空间，兼容 macOS/Linux。
_MAX_SCRATCH_PATH_BYTES = 60
_WORKER_SUFFIX_BYTES = len("/mu-xxxxxxxx/tmp")


def create_managed_temp_dir(*, reserve_worker: bool = False) -> tempfile.TemporaryDirectory[str]:
    """优先沿用临时根目录；过长时改用短根，并为嵌套 worker 预留同树清理空间。"""
    parents = [tempfile.gettempdir()]
    if os.name != "nt" and "/tmp" not in parents:
        parents.append("/tmp")
    for parent in parents:
        # tempfile 在支持的 CPython 版本使用八位随机名；创建后仍检查实际结果。
        reserved = _WORKER_SUFFIX_BYTES if reserve_worker else 0
        projected = Path(parent) / "mu-xxxxxxxx" / "tmp"
        if os.name != "nt" and len(os.fsencode(projected)) + reserved > _MAX_SCRATCH_PATH_BYTES:
            continue
        directory = tempfile.TemporaryDirectory(prefix="mu-", dir=parent)
        scratch = Path(directory.name) / "tmp"
        if os.name == "nt" or len(os.fsencode(scratch)) + reserved <= _MAX_SCRATCH_PATH_BYTES:
            return directory
        directory.cleanup()
    raise RuntimeError("No temporary root is short enough for managed worker IPC")


__all__ = ["create_managed_temp_dir"]
