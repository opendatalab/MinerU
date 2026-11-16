"""解析执行与客户端等待使用的显式时间预算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

DEFAULT_TASK_TIMEOUT_SECONDS = 3600


def validate_timeout(value: float, name: str) -> float:
    """拒绝布尔值、非有限值和非正数，Python 调用允许精确到小数秒。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class ExecutionLease:
    """同机父子进程共享单调时钟，文件身份用于关联 Router 的执行记录。"""

    execution_id: str
    file_id: str
    deadline: float


class ExecutionReporter(Protocol):
    """仅报告生命周期，不传递文档内容或模型对象。"""

    def execution_started(self, lease: ExecutionLease) -> None:
        """报告取得解析名额后的实际执行期限。"""
        ...

    def execution_finished(self, execution_id: str) -> None:
        """仅在实际执行及清理结束后撤销期限。"""
        ...

    def execution_timed_out(self, lease: ExecutionLease) -> None:
        """保留超时身份，即使取消马上完成也不能丢失父进程恢复原因。"""
        ...


__all__ = ["DEFAULT_TASK_TIMEOUT_SECONDS", "ExecutionLease", "ExecutionReporter", "validate_timeout"]
