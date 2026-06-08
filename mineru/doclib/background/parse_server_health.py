"""Parse-server health state and periodic health check task."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("mineru.health_check")

CHECK_INTERVAL_SEC = 60
MAX_RESTART_ATTEMPTS = 3


@dataclass
class ParseServerHealth:
    local_healthy: bool = False
    local_supported_tiers: list[str] = field(default_factory=list)
    local_mode: str = "disabled"
    self_hosted_url: str | None = None
    remote_healthy: bool = False
    remote_supported_tiers: list[str] = field(default_factory=list)
    restart_count: int = 0
    managed_proc: subprocess.Popen | None = None


_parse_server_health = ParseServerHealth()


def get_health() -> ParseServerHealth:
    return _parse_server_health


class ParseServerHealthCheck:
    """Periodically probes parse-server health via GET /v1/tiers."""

    def __init__(self, config_svc) -> None:
        self.config_svc = config_svc
        self.running = False

    async def run(self) -> None:
        self.running = True
        health = get_health()

        while self.running:
            await asyncio.sleep(CHECK_INTERVAL_SEC)
            if not self.running:
                break

            # refresh config on each cycle (hot-reload)
            mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
            health.local_mode = mode
            self_hosted_url = await self.config_svc.get("parse_server.local.self_hosted_url")
            health.self_hosted_url = self_hosted_url if self_hosted_url else None

            # probe local
            if health.local_mode != "disabled":
                url = self._local_url(health)
                if url:
                    healthy, tiers = await self._probe(url)
                    health.local_healthy = healthy
                    health.local_supported_tiers = tiers

            # probe remote
            remote_url = (await self.config_svc.get("parse_server.remote.url")) or "https://mineru.net/api"
            healthy, tiers = await self._probe(remote_url)
            health.remote_healthy = healthy
            health.remote_supported_tiers = tiers

            # managed mode: restart if crashed
            if health.local_mode == "managed" and not health.local_healthy:
                await self._try_restart_managed(health)

    async def _try_restart_managed(self, health: ParseServerHealth) -> None:
        if health.restart_count >= MAX_RESTART_ATTEMPTS:
            logger.error(
                f"Managed parse-server failed {health.restart_count} restarts, disabling"
            )
            health.local_mode = "disabled"
            return

        health.restart_count += 1
        managed_tier = (await self.config_svc.get("parse_server.local.managed_tier")) or "standard"
        logger.info(f"Restarting managed parse-server (attempt {health.restart_count}/{MAX_RESTART_ATTEMPTS})")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "mineru.parser.api_server", "--backend", managed_tier, "--port", "15981"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            health.managed_proc = proc
        except Exception as exc:
            logger.error(f"Failed to restart managed parse-server: {exc}")

    @staticmethod
    async def _probe(base_url: str) -> tuple[bool, list[str]]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base_url}/v1/tiers")
                if resp.status_code == 200:
                    data = resp.json()
                    tiers = [t.get("id") for t in data.get("data", [])]
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
                health.managed_proc.wait(timeout=10)
            except Exception:
                pass
