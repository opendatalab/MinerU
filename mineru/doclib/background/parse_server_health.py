"""Parse-server health state and periodic health check task."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import cast

import httpx

from ...types import TIERS, Tier

logger = logging.getLogger("mineru.health_check")

MAX_RESTART_ATTEMPTS = 3
DEFAULT_MANAGED_URL = "http://127.0.0.1:15981"


@dataclass
class ParseServerHealth:
    local_healthy: bool = False
    local_starting: bool = False
    local_started_at: float = 0.0
    local_supported_tiers: list[Tier] = field(default_factory=list)
    local_mode: str = "disabled"
    self_hosted_url: str | None = None
    managed_url: str = DEFAULT_MANAGED_URL
    managed_tier: Tier | None = None
    local_last_probe_at: int | None = None
    local_last_success_at: int | None = None
    local_last_failure_at: int | None = None
    remote_healthy: bool = False
    remote_url: str | None = None
    remote_last_probe_at: int | None = None
    remote_last_success_at: int | None = None
    remote_last_failure_at: int | None = None
    remote_supported_tiers: list[Tier] = field(default_factory=list)
    restart_count: int = 0
    managed_proc: subprocess.Popen | None = None


_parse_server_health = ParseServerHealth()


def get_health() -> ParseServerHealth:
    return _parse_server_health


def api_server_args_for_tier(tier: Tier) -> list[str]:
    """Return managed api-server process args for a doclib tier.

    Managed doclib startup uses ``--tier`` so the api-server resolves its own
    backend. Backend remains an api-server implementation detail and must not
    leak into runtime doclib parse requests.
    """
    return ["--tier", tier, "--port", "15981"]


class ParseServerHealthCheck:
    """Periodically probes parse-server health via GET /v1/tiers."""

    def __init__(
        self,
        config_svc,
        *,
        interval_sec: int,
        probe_timeout_sec: int,
        startup_grace_sec: int,
        stop_timeout_sec: int,
    ) -> None:
        self.config_svc = config_svc
        self.interval_sec = interval_sec
        self.probe_timeout_sec = probe_timeout_sec
        self.startup_grace_sec = startup_grace_sec
        self.stop_timeout_sec = stop_timeout_sec
        self.running = False

    async def run(self) -> None:
        self.running = True
        health = get_health()

        while self.running:
            # refresh config on each cycle (hot-reload)
            mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
            health.local_mode = mode
            managed_tier = (await self.config_svc.get("parse_server.local.managed_tier")) or "standard"
            health.managed_tier = cast(Tier, managed_tier)
            self_hosted_url = await self.config_svc.get("parse_server.local.self_hosted_url")
            health.self_hosted_url = self_hosted_url if self_hosted_url else None

            # probe local
            if health.local_mode != "disabled":
                url = self._local_url(health)
                if url:
                    healthy, tiers = await self._probe(url)
                    now_ms = int(time.time() * 1000)
                    health.local_last_probe_at = now_ms
                    health.local_healthy = healthy
                    health.local_supported_tiers = tiers
                    if healthy:
                        health.local_last_success_at = now_ms
                        health.local_starting = False
                    else:
                        health.local_last_failure_at = now_ms

            # probe remote
            remote_url = cast(str, await self.config_svc.get("parse_server.remote.url"))
            health.remote_url = remote_url
            healthy, tiers = await self._probe(remote_url)
            now_ms = int(time.time() * 1000)
            health.remote_last_probe_at = now_ms
            health.remote_healthy = healthy
            health.remote_supported_tiers = tiers
            if healthy:
                health.remote_last_success_at = now_ms
            else:
                health.remote_last_failure_at = now_ms

            # managed mode: restart if crashed (skip if still starting — give it 30s)
            if health.local_mode == "managed" and not health.local_healthy:
                if health.local_starting:
                    elapsed = asyncio.get_event_loop().time() - health.local_started_at
                    if elapsed < self.startup_grace_sec:
                        continue  # still loading models, don't restart
                logger.warning("Managed parse-server is unhealthy, attempting restart")
                await self._try_restart_managed(health)

            await asyncio.sleep(self.interval_sec)

    async def _try_restart_managed(self, health: ParseServerHealth) -> None:
        if health.restart_count >= MAX_RESTART_ATTEMPTS:
            logger.error(
                "Managed parse-server failed %d restarts, disabling", MAX_RESTART_ATTEMPTS
            )
            health.local_mode = "disabled"
            return

        health.restart_count += 1
        managed_tier = (await self.config_svc.get("parse_server.local.managed_tier")) or "standard"
        cmd = [sys.executable, "-m", "mineru.parser.api_server", *api_server_args_for_tier(managed_tier)]
        logger.info("Restarting managed parse-server (attempt %d/%d): %s",
                    health.restart_count, MAX_RESTART_ATTEMPTS, " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Managed parse-server restarted (PID %d, tier=%s)", proc.pid, managed_tier)
            health.managed_proc = proc
            health.local_starting = True
            health.local_started_at = asyncio.get_event_loop().time()
        except Exception as exc:
            logger.error(f"Failed to restart managed parse-server: {exc}")

    async def _probe(self, base_url: str) -> tuple[bool, list[Tier]]:
        try:
            async with httpx.AsyncClient(timeout=self.probe_timeout_sec) as client:
                resp = await client.get(f"{base_url}/v1/tiers")
                if resp.status_code == 200:
                    data = resp.json()
                    tiers: list[Tier] = []
                    for t in data.get("data", []):
                        tier_id = t.get("id")
                        if tier_id in TIERS:
                            tiers.append(cast(Tier, tier_id))
                    return True, tiers
        except Exception:
            pass
        return False, []

    @staticmethod
    def _local_url(health: ParseServerHealth) -> str | None:
        if health.local_mode == "managed":
            return "http://127.0.0.1:15981"
        if health.local_mode == "self_hosted":
            return health.self_hosted_url
        return None

    async def stop(self) -> None:
        self.running = False
        health = get_health()
        if health.managed_proc:
            try:
                health.managed_proc.terminate()
                health.managed_proc.wait(timeout=self.stop_timeout_sec)
            except Exception:
                pass
