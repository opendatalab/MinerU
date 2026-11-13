"""同步句柄与异步代理共享的实例生命周期，不负责执行或重建引擎。"""

from __future__ import annotations

import logging
import threading

from mineru_vl_utils.vlm_client.base_client import EngineHealth

from ..runtime.ownership import RuntimeOwner, runtime_owner
from .errors import EngineDeadError, EngineNotReadyError

logger = logging.getLogger(__name__)


class RuntimeLifecycle:
    """锁存首个致命错误，并在同一状态锁下管理宿主关联及暂不可用状态。"""

    def __init__(self, backend: str) -> None:
        """仅创建轻量状态，不触发线程、循环或模型加载。"""
        self.backend = backend
        self.condition = threading.Condition()
        self.owners: set[RuntimeOwner] = set()
        self.pinned = False
        self.closing = False
        self._fatal_error: BaseException | None = None
        self._unavailable: str | None = None

    def raise_if_dead(self) -> None:
        """所有入口共用永久失效判定，异常链指回最初故障。"""
        with self.condition:
            if self._fatal_error is not None:
                raise EngineDeadError(str(self._fatal_error)) from self._fatal_error

    def require_ready(self, *, allow_unavailable: bool = False) -> None:
        """允许暂不可用实例接受健康复查，但不允许新的推理。"""
        with self.condition:
            self.raise_if_dead()
            if self.closing:
                raise RuntimeError("VLM runtime is shutting down")
            if self._unavailable is not None and not allow_unavailable:
                raise EngineNotReadyError(self._unavailable)

    def retain(self, owner: RuntimeOwner | None) -> None:
        """拒绝新宿主借用死亡实例，并同步既有的暂不可用状态。"""
        with self.condition:
            if owner is not None and self._fatal_error is not None and owner.engine_error is None:
                owner.engine_error = str(self._fatal_error)
            self.require_ready(allow_unavailable=True)
            if owner is None:
                self.pinned = True
            else:
                self.owners.add(owner)
                owner.set_engine_unavailable(self, self._unavailable)

    def release(self, owner: RuntimeOwner) -> None:
        """撤销一个宿主的关联，不清除其他宿主的故障或租约。"""
        with self.condition:
            self.owners.discard(owner)
            owner.set_engine_unavailable(self, None)

    def mark_fatal(self, error: BaseException) -> None:
        """只记录首次完整异常，后续故障仍通知全部关联宿主。"""
        with self.condition:
            first = self._fatal_error is None
            if first:
                self._fatal_error = error
            message = str(self._fatal_error)
            owners = set(self.owners)
            current = runtime_owner.get()
            if current is not None:
                owners.add(current)
            self._unavailable = None
            for owner in owners:
                if owner.engine_error is None:
                    owner.engine_error = message
                owner.set_engine_unavailable(self, None)
        if first:
            logger.error(
                "VLM engine failed: backend=%s error_type=%s",
                self.backend,
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )

    def apply_health(self, health: EngineHealth) -> None:
        """迟到的健康结果不能覆盖死亡，关闭开始后不再发布探测状态。"""
        with self.condition:
            self.raise_if_dead()
            if self.closing:
                return
            if health.state == "dead":
                error = EngineDeadError(health.message or "VLM engine is unavailable")
                self.mark_fatal(error)
                raise error
            if health.state not in {"healthy", "unavailable"}:
                raise ValueError("Invalid engine health state")
            self._unavailable = (
                (health.message or "VLM engine health is unavailable") if health.state == "unavailable" else None
            )
            for owner in tuple(self.owners):
                owner.set_engine_unavailable(self, self._unavailable)
            self.require_ready()

    def begin_close(self) -> None:
        """停止接纳新工作；实际关闭动作仍由各运行时按执行机制排空。"""
        with self.condition:
            self.closing = True

    def finish_close(self) -> None:
        """资源释放后移除暂不可用记录，不撤销永久死亡的宿主通知。"""
        with self.condition:
            for owner in tuple(self.owners):
                owner.set_engine_unavailable(self, None)


__all__ = ["RuntimeLifecycle"]
