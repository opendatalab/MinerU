# Copyright (c) Opendatalab. All rights reserved.
"""共享设备的文档租约与本地模型阶段串行边界。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _DeviceExecution:
    """同一设备共享原子模型及缓存回收，因此统一保护其执行。"""

    lock: threading.RLock = field(default_factory=threading.RLock)
    documents: int = 0


_registry_lock = threading.Lock()
_devices: dict[str, _DeviceExecution] = {}


def _device_execution(device: str) -> _DeviceExecution:
    """首次真实使用设备时创建执行状态。"""
    with _registry_lock:
        return _devices.setdefault(str(device), _DeviceExecution())


@contextmanager
def local_model_stage(device: str) -> Iterator[None]:
    """只保护同步本地模型阶段，不能跨越 VLM 等待持有该锁。"""
    with _device_execution(device).lock:
        yield


def acquire_document(device: str) -> None:
    """在模型初始化后、执行文档前取得设备缓存使用租约。"""
    state = _device_execution(device)
    with state.lock:
        state.documents += 1


def release_document(device: str, cleanup: Callable[[str], None]) -> None:
    """仅最后一份文档退出时回收共享缓存，并防止新文档与回收交错。"""
    state = _device_execution(device)
    with state.lock:
        state.documents -= 1
        if state.documents == 0:
            cleanup(device)


__all__ = ["acquire_document", "local_model_stage", "release_document"]
