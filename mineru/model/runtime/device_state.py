# Copyright (c) Opendatalab. All rights reserved.
"""进程默认设备的一次决策及不影响默认状态的显式模型覆盖。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
import threading
from typing import Literal

from loguru import logger

from .device_types import RuntimeDevice, TorchDeviceSelection


@dataclass
class _ProcessState:
    """每个 PID 独立持有初始化锁及完整快照，不使用继承的父进程锁。"""

    lock: threading.RLock = field(default_factory=threading.RLock)
    decision: RuntimeDevice | None = None
    initializing: bool = False


_states: dict[int, _ProcessState] = {}
_model_device: ContextVar[tuple[int, RuntimeDevice] | None] = ContextVar("mineru_model_device", default=None)


def cached_runtime_device() -> RuntimeDevice | None:
    """无副作用地读取本进程已发布快照，初始化期间也不阻塞调用者。"""
    state = _states.get(os.getpid())
    return state.decision if state is not None else None


def resolve_runtime_device(
    *, allow_download: bool = True, intra_op_num_threads: int = 0, inter_op_num_threads: int = 0
) -> RuntimeDevice:
    """首次正式选型后固定整个进程，后续参数和配置变化不重新选型。"""
    from .device import resolve_small_model_backend, resolve_torch_device

    pid = os.getpid()
    state = _states.get(pid)
    if state is None:
        # setdefault 原子地选出同一进程的初始化锁，子进程不会等待父进程持有的锁。
        state = _states.setdefault(pid, _ProcessState())
    with state.lock:
        if state.decision is not None:
            return state.decision
        if state.initializing:
            raise RuntimeError("Recursive runtime device selection")
        state.initializing = True
        try:
            backend = resolve_small_model_backend()
            if backend == "torch":
                selection = resolve_torch_device()
            else:
                from .onnx_device import resolve_onnx_device

                selection = resolve_onnx_device(
                    allow_download=allow_download,
                    intra_op_num_threads=intra_op_num_threads,
                    inter_op_num_threads=inter_op_num_threads,
                )
            state.decision = RuntimeDevice(selection)
            logger.info(
                "Runtime device selected: backend={}, mode={}, reason={}", backend, state.decision.device, selection.reason
            )
            return state.decision
        finally:
            state.initializing = False


def resolve_model_device(
    backend: str | None = None,
    device: str | None = None,
    *,
    intra_op_num_threads: int = 0,
    inter_op_num_threads: int = 0,
) -> RuntimeDevice:
    """默认模型复用进程决策；显式后端或 Torch 设备覆盖仅作用于当前构造。"""
    from .device import resolve_small_model_backend, resolve_torch_device
    from .onnx_device import resolve_onnx_device

    scoped = _model_device.get()
    current = scoped[1] if scoped is not None and scoped[0] == os.getpid() else cached_runtime_device()
    if backend in (None, "auto"):
        current = current or resolve_runtime_device(
            intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads
        )
        selected = current.backend
    else:
        selected = resolve_small_model_backend(backend)
    if selected == "torch" and device is not None:
        if current is not None and current.backend == "torch" and current.model_device == device:
            return current
        return RuntimeDevice(TorchDeviceSelection(device, "explicit_model_device"))
    if current is not None and current.backend == selected:
        return current
    if current is None and selected == resolve_small_model_backend():
        return resolve_runtime_device(intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads)
    selection = (
        resolve_torch_device()
        if selected == "torch"
        else resolve_onnx_device(intra_op_num_threads=intra_op_num_threads, inter_op_num_threads=inter_op_num_threads)
    )
    return RuntimeDevice(selection)


def resolve_model_backend(backend: str | None = None) -> Literal["onnx", "torch"]:
    """CPU 专用会话只解析后端，已初始化时复用快照，不为了后端名称启动校准。"""
    from .device import resolve_small_model_backend

    if backend not in (None, "auto"):
        return resolve_small_model_backend(backend)
    scoped = _model_device.get()
    current = scoped[1] if scoped is not None and scoped[0] == os.getpid() else cached_runtime_device()
    return current.backend if current is not None else resolve_small_model_backend()


@contextmanager
def model_device_scope(decision: RuntimeDevice) -> Iterator[None]:
    """将显式选型传给嵌套模型和会话，不污染其他线程或进程默认决策。"""
    token = _model_device.set((os.getpid(), decision))
    try:
        yield
    finally:
        _model_device.reset(token)


__all__ = [
    "cached_runtime_device",
    "model_device_scope",
    "resolve_model_backend",
    "resolve_model_device",
    "resolve_runtime_device",
]
