"""Parse-server health state and periodic health check task."""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, Iterator, cast

import httpx

from ...config import LogConfig, ManagedParseServerConfig, config
from ...parser.api_client import should_trust_env_for_url
from ...types import TIERS, Tier

logger = logging.getLogger("mineru.health_check")

MAX_RESTART_ATTEMPTS = 3
DEFAULT_MANAGED_URL = "http://127.0.0.1:16580"
MANAGED_PARSE_SERVER_ENV = "MINERU_MANAGED_PARSE_SERVER"


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
    running_managed_tier: Tier | None = None
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


def api_server_args_for_tier(tier: Tier, *, host: str, port: int) -> list[str]:
    """Return managed api-server process args for a doclib tier.

    Managed doclib startup uses ``--tier`` so the api-server resolves its own
    backend. Backend remains an api-server implementation detail and must not
    leak into runtime doclib parse requests.
    """
    return ["--tier", tier, "--host", host, "--port", str(port)]


def managed_parse_server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def select_available_managed_port(host: str, port: int, *, strict_port: bool, port_probe_count: int) -> int:
    last_exc: OSError | None = None
    candidate_ports = (port,) if strict_port else range(port, port + port_probe_count)
    for candidate_port in candidate_ports:
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe_sock.bind((host, candidate_port))
            return int(probe_sock.getsockname()[1])
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or strict_port:
                raise
            last_exc = exc
        finally:
            probe_sock.close()

    raise RuntimeError(f"No available managed parse-server port in range {port}-{port + port_probe_count - 1}.") from last_exc


def get_parse_server_stdout_log_path() -> str:
    return os.path.expanduser(config.doclib.log.resolved_parse_server_stdout_path)


def get_parse_server_stderr_log_path() -> str:
    return os.path.expanduser(config.doclib.log.resolved_parse_server_stderr_path)


def _parse_server_stdout_log_path(log_cfg: LogConfig | None) -> str:
    if log_cfg is None:
        return get_parse_server_stdout_log_path()
    return os.path.expanduser(log_cfg.resolved_parse_server_stdout_path)


def _parse_server_stderr_log_path(log_cfg: LogConfig | None) -> str:
    if log_cfg is None:
        return get_parse_server_stderr_log_path()
    return os.path.expanduser(log_cfg.resolved_parse_server_stderr_path)


def _ensure_log_dir(path: str) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


@contextmanager
def open_managed_parse_server_logs(*, marker: str, log_cfg: LogConfig | None = None) -> Iterator[tuple[IO[str], IO[str]]]:
    stdout_log_path = _parse_server_stdout_log_path(log_cfg)
    stderr_log_path = _parse_server_stderr_log_path(log_cfg)
    _ensure_log_dir(stdout_log_path)
    _ensure_log_dir(stderr_log_path)

    with (
        open(stdout_log_path, "a", encoding="utf-8") as stdout_log_file,
        open(stderr_log_path, "a", encoding="utf-8") as stderr_log_file,
    ):
        stdout_log_file.write(f"\n--- managed parse-server stdout: {marker} ---\n")
        stderr_log_file.write(f"\n--- managed parse-server stderr: {marker} ---\n")
        stdout_log_file.flush()
        stderr_log_file.flush()
        yield stdout_log_file, stderr_log_file


def start_managed_parse_server(
    *,
    tier: Tier,
    managed_cfg: ManagedParseServerConfig,
    log_cfg: LogConfig | None,
    marker: str,
) -> tuple[subprocess.Popen, str]:
    port = select_available_managed_port(
        managed_cfg.host,
        managed_cfg.port,
        strict_port=managed_cfg.strict_port,
        port_probe_count=managed_cfg.port_probe_count,
    )
    managed_url = managed_parse_server_url(managed_cfg.host, port)
    cmd = [
        sys.executable,
        "-m",
        "mineru.parser.api_server",
        *api_server_args_for_tier(tier, host=managed_cfg.host, port=port),
    ]
    env = os.environ.copy()
    env[MANAGED_PARSE_SERVER_ENV] = "1"
    logger.info("Starting managed parse-server (%s): %s", marker, " ".join(cmd))
    with open_managed_parse_server_logs(marker=marker, log_cfg=log_cfg) as (
        stdout_log_file,
        stderr_log_file,
    ):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout_log_file, stderr=stderr_log_file, env=env)
    return proc, managed_url


def stop_managed_parse_server(proc: subprocess.Popen | None, *, timeout_sec: int, reason: str) -> None:
    if proc is None or proc.poll() is not None:
        return

    pid = proc.pid
    stdin = getattr(proc, "stdin", None)
    if stdin is not None and not getattr(stdin, "closed", False):
        try:
            stdin.close()
        except Exception as exc:
            logger.debug("Failed to close managed parse-server stdin (PID %d, reason=%s): %s", pid, reason, exc)

    try:
        proc.wait(timeout=timeout_sec)
        logger.info("Managed parse-server stopped after stdin EOF (PID %d, reason=%s)", pid, reason)
        return
    except subprocess.TimeoutExpired:
        logger.warning("Managed parse-server did not stop after stdin EOF (PID %d, reason=%s), terminating", pid, reason)

    try:
        proc.terminate()
        proc.wait(timeout=timeout_sec)
        logger.info("Managed parse-server terminated (PID %d, reason=%s)", pid, reason)
        return
    except subprocess.TimeoutExpired:
        logger.warning("Managed parse-server did not terminate within timeout (PID %d, reason=%s), killing", pid, reason)
    except Exception as exc:
        logger.warning("Failed to terminate managed parse-server (PID %d, reason=%s): %s", pid, reason, exc)

    try:
        proc.kill()
        proc.wait(timeout=timeout_sec)
        logger.info("Managed parse-server killed (PID %d, reason=%s)", pid, reason)
    except Exception as exc:
        logger.error("Failed to kill managed parse-server (PID %d, reason=%s): %s", pid, reason, exc)


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
        managed_parse_server: ManagedParseServerConfig | None = None,
        log_cfg: LogConfig | None = None,
    ) -> None:
        self.config_svc = config_svc
        self.interval_sec = interval_sec
        self.probe_timeout_sec = probe_timeout_sec
        self.startup_grace_sec = startup_grace_sec
        self.stop_timeout_sec = stop_timeout_sec
        self.managed_parse_server = managed_parse_server or config.doclib.managed_parse_server
        self.log_cfg = log_cfg
        self.running = False

    async def run(self) -> None:
        self.running = True
        health = get_health()

        while self.running:
            # refresh config on each cycle (hot-reload)
            mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
            health.local_mode = mode
            managed_tier = (await self.config_svc.get("parse_server.local.managed_tier")) or "standard"
            desired_managed_tier = cast(Tier, managed_tier)
            health.managed_tier = desired_managed_tier
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
            elif health.local_mode == "managed" and health.local_healthy:
                await self._try_restart_managed_for_tier_change(health, desired_managed_tier)

            await asyncio.sleep(self.interval_sec)

    async def _try_restart_managed_for_tier_change(
        self, health: ParseServerHealth, desired_managed_tier: Tier
    ) -> bool:
        running_managed_tier = health.running_managed_tier
        if running_managed_tier is None or running_managed_tier == desired_managed_tier:
            return False

        logger.info(
            "Managed parse-server tier changed from %s to %s, restarting",
            running_managed_tier,
            desired_managed_tier,
        )
        await self._try_restart_managed(
            health,
            reason="tier-change",
            marker=f"tier change {running_managed_tier}->{desired_managed_tier}",
            count_restart=False,
        )
        return True

    async def _try_restart_managed(
        self,
        health: ParseServerHealth,
        *,
        reason: str = "restart",
        marker: str | None = None,
        count_restart: bool = True,
    ) -> None:
        if count_restart and health.restart_count >= MAX_RESTART_ATTEMPTS:
            logger.error(
                "Managed parse-server failed %d restarts, disabling", MAX_RESTART_ATTEMPTS
            )
            health.local_mode = "disabled"
            return

        if count_restart:
            health.restart_count += 1
        managed_tier = (await self.config_svc.get("parse_server.local.managed_tier")) or "standard"
        stop_managed_parse_server(health.managed_proc, timeout_sec=self.stop_timeout_sec, reason=reason)
        health.managed_proc = None
        health.running_managed_tier = None
        try:
            proc, managed_url = start_managed_parse_server(
                tier=cast(Tier, managed_tier),
                managed_cfg=self.managed_parse_server,
                log_cfg=self.log_cfg,
                marker=marker or f"restart attempt {health.restart_count}",
            )
            health.managed_url = managed_url
            logger.info("Managed parse-server restarted (PID %d, tier=%s)", proc.pid, managed_tier)
            health.managed_proc = proc
            health.running_managed_tier = cast(Tier, managed_tier)
            health.local_starting = True
            health.local_started_at = asyncio.get_event_loop().time()
        except Exception as exc:
            logger.error(f"Failed to restart managed parse-server: {exc}")

    async def _probe(self, base_url: str) -> tuple[bool, list[Tier]]:
        try:
            async with httpx.AsyncClient(
                timeout=self.probe_timeout_sec,
                trust_env=should_trust_env_for_url(base_url),
            ) as client:
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
            return health.managed_url
        if health.local_mode == "self_hosted":
            return health.self_hosted_url
        return None

    async def stop(self) -> None:
        self.running = False
        health = get_health()
        stop_managed_parse_server(health.managed_proc, timeout_sec=self.stop_timeout_sec, reason="health check stop")
        health.managed_proc = None
