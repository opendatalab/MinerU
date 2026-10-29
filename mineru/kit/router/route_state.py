"""Router 内部执行与结果租约状态，不混入上游 HTTP 响应缓存。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...tasks.execution import TaskExecution

if TYPE_CHECKING:
    from .workers import JobReservation


@dataclass
class ResultLease:
    """记录父任务持有的结果租约，以及墙上时间期限和单调时钟续租顺序。"""

    held: bool = False
    retain_until: float | None = None
    refreshed_at: float = 0.0
    # 调度按尝试时间轮转，失败或被整轮超时中断也不能长期占据队首。
    attempted_at: float = 0.0


@dataclass
class ExecutionRouteState:
    """保存单文件执行快照、下游文件身份和本次预占的唯一释放责任。"""

    execution: TaskExecution
    upstream_file_id: str | None = None
    reservation: JobReservation | None = None
    upstream_headers: dict[str, str] = field(default_factory=dict)
    upstream_lost: bool = False
    # 租约指向的结果可能在执行快照因 worker 丢失而失效后仍需清理。
    result_file_id: str | None = None

    def release_reservation(self) -> None:
        """先交还本地所有权再幂等释放，仅由预占对象处理 worker generation。"""
        reservation, self.reservation = self.reservation, None
        if reservation is not None:
            reservation.release()


__all__ = ["ExecutionRouteState", "ResultLease"]
