"""单个 Doclib 应用拥有的解析服务探测状态与托管进程归属。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import ManagedServerTier, Tier

if TYPE_CHECKING:
    from ..parser.process_control import ManagedProcessControl
DEFAULT_MANAGED_URL = "http://127.0.0.1:16580"


@dataclass
class ProbeResult:
    """保存一次服务探测的可用档位和稳定错误信息。"""

    healthy: bool = False
    tiers: list[Tier] = field(default_factory=list)
    error_code: str | None = None
    error_msg: str | None = None


@dataclass
class ProbeState:
    """保存一个目标的当前探测结果与时间，不共享可变列表。"""

    url: str | None = None
    probe: ProbeResult = field(default_factory=ProbeResult)
    last_probe_at: int | None = None
    last_success_at: int | None = None
    last_failure_at: int | None = None


@dataclass
class ParseServerHealth:
    """归属单个应用的探测、档位切换和托管进程生命周期状态。"""

    local: ProbeState = field(default_factory=ProbeState)
    remote: ProbeState = field(default_factory=ProbeState)
    local_starting: bool = False
    local_started_at: float = 0.0
    local_mode: str = "disabled"
    self_hosted_url: str | None = None
    managed_url: str = DEFAULT_MANAGED_URL
    managed_tier: ManagedServerTier | None = None
    running_managed_tier: ManagedServerTier | None = None
    restart_count: int = 0
    managed_proc: subprocess.Popen | None = None
    managed_control: ManagedProcessControl | None = None
    managed_tier_transition: bool = False
    local_generation: int = 0

    def begin_managed_tier_transition(self, tier: ManagedServerTier) -> None:
        """立即撤销旧服务的可用状态；保留子进程真实启动状态以正确计算停止预算。"""
        self.managed_tier = tier
        self.managed_tier_transition = True
        self.local_generation += 1
        self.local = ProbeState(
            url=self.managed_url,
            probe=ProbeResult(
                error_code="parse_server_unavailable",
                error_msg=f"Managed parse-server is preparing tier '{tier}'.",
            ),
        )


__all__ = ["DEFAULT_MANAGED_URL", "ParseServerHealth", "ProbeResult", "ProbeState"]
