"""Windows 托管进程树归属；父进程退出后仍能回收 Job 内的模型子进程。"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from typing import Any


class _BasicLimits(ctypes.Structure):
    """对应 Win32 JOBOBJECT_BASIC_LIMIT_INFORMATION 的固定 ABI。"""

    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", ctypes.c_uint32),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32),
    ]


class _ExtendedLimits(ctypes.Structure):
    """只设置 KILL_ON_JOB_CLOSE，不改变模型内存、CPU 或调度限制。"""

    _fields_ = [
        ("basic", _BasicLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    """读取仍存活的进程数，不能仅以最外层进程结束判断整个树已退出。"""

    _fields_ = [
        ("times", ctypes.c_int64 * 4),
        ("faults", ctypes.c_uint32),
        ("total", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("terminated", ctypes.c_uint32),
    ]


class WindowsProcessJob:
    """显式持有 Win32 Job 句柄，不通过模块导入或析构触发进程管理。"""

    def __init__(self, pid: int) -> None:
        """把刚创建的受管 Python 进程加入 Job，其后代自动继承该归属。"""
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures: dict[str, tuple[list[Any], Any]] = {
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
            "SetInformationJobObject": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            "QueryInformationJobObject": (
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "OpenProcess": ([ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
            "AssignProcessToJobObject": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
            "TerminateJobObject": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = arguments, result
        self._kernel = kernel
        self._handle = kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        process = None
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            process = kernel.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            if not process or not kernel.AssignProcessToJobObject(self._handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise
        finally:
            if process:
                kernel.CloseHandle(process)

    def stop(self, deadline: float) -> None:
        """终止所有仍存活的 Job 成员，失败时保留句柄供宿主继续回收。"""
        if not self._handle:
            return
        if not self._kernel.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        while True:
            accounting = _Accounting()
            if not self._kernel.QueryInformationJobObject(
                self._handle,
                1,
                ctypes.byref(accounting),
                ctypes.sizeof(accounting),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.active == 0:
                self.close()
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Managed Windows process job did not stop")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        """释放自有 Job 句柄，系统的关闭策略负责兜底清理仍存活的子进程。"""
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


def create_process_job(process: subprocess.Popen) -> WindowsProcessJob | None:
    """仅 Windows 创建 Job；失败时立即停止刚启动的进程，不留下未托管实例。"""
    if os.name != "nt":
        return None
    try:
        return WindowsProcessJob(process.pid)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        raise


__all__ = ["WindowsProcessJob", "create_process_job"]
