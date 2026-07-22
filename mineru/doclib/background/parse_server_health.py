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
from typing import IO, TYPE_CHECKING, Iterator, cast

import httpx

from ...config import LogConfig, ManagedParseServerConfig, config
from ...parser.api_client import should_trust_env_for_url
from ...types import DEPLOYMENT_TIERS, TIERS, DeploymentTier, Tier
from ..config_defaults import CONFIG_DEFAULTS
from ..remote_api import resolve_remote_api_key

logger = logging.getLogger("mineru.health_check")

if TYPE_CHECKING:
    from ..services.config_svc import ConfigService

MAX_RESTART_ATTEMPTS = 3
DEFAULT_MANAGED_URL = "http://127.0.0.1:16580"
MANAGED_PARSE_SERVER_ENV = "MINERU_MANAGED_PARSE_SERVER"
_NON_RETRYABLE_MODEL_PRELOAD_ERRORS = frozenset(
    {
        "model_preload_dependency_missing",
        "model_preload_files_missing",
        "model_preload_device_unavailable",
    }
)


@dataclass
class ProbeResult:
    healthy: bool = False
    tiers: list[Tier] = field(default_factory=list)
    error_code: str | None = None
    error_msg: str | None = None


@dataclass
class ProbeState:
    url: str | None = None
    probe: ProbeResult = field(default_factory=ProbeResult)
    last_probe_at: int | None = None
    last_success_at: int | None = None
    last_failure_at: int | None = None


@dataclass
class ParseServerHealth:
    local: ProbeState = field(default_factory=ProbeState)
    remote: ProbeState = field(default_factory=ProbeState)
    local_starting: bool = False
    local_started_at: float = 0.0
    local_mode: str = "disabled"
    self_hosted_url: str | None = None
    managed_url: str = DEFAULT_MANAGED_URL
    managed_tier: DeploymentTier | None = None
    running_managed_tier: DeploymentTier | None = None
    restart_count: int = 0
    managed_proc: subprocess.Popen | None = None


_parse_server_health = ParseServerHealth()


def get_health() -> ParseServerHealth:
    return _parse_server_health


async def get_managed_parse_server_tier(config_svc: "ConfigService") -> DeploymentTier:
    """读取 managed parse-server tier；忽略非法存量值并回退到默认值。"""
    key = "parse_server.local.managed_tier"
    raw_tier = await config_svc.get(key) or CONFIG_DEFAULTS[key]
    if raw_tier in DEPLOYMENT_TIERS:
        return cast(DeploymentTier, raw_tier)
    logger.warning(
        "Ignoring invalid managed parse-server tier override %r and using default %r",
        raw_tier,
        CONFIG_DEFAULTS[key],
    )
    return cast(DeploymentTier, CONFIG_DEFAULTS[key])


def api_server_args_for_tier(tier: DeploymentTier, *, host: str, port: int) -> list[str]:
    """Return managed api-server process args for a doclib tier.

    Managed doclib startup uses ``--tier`` so the api-server resolves its own
    backend. Backend remains an api-server implementation detail and must not
    leak into runtime doclib parse requests.
    """
    return [
        "--tier",
        tier,
        "--host",
        host,
        "--port",
        str(port),
        "--allow-local-source",
        "--no-flash",
        "--preload-models",
    ]


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
    tier: DeploymentTier,
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


def _managed_parse_server_needs_restart(
    health: ParseServerHealth,
    *,
    now: float,
    startup_timeout_sec: int,
) -> bool:
    if health.local.probe.error_code in _NON_RETRYABLE_MODEL_PRELOAD_ERRORS:
        return False
    proc = health.managed_proc
    if health.local_starting and proc is not None and proc.poll() is None:
        return now - health.local_started_at >= startup_timeout_sec
    return True


class ParseServerHealthCheck:
    """Periodically probes parse-server health via GET /v1/tiers."""

    def __init__(
        self,
        config_svc: "ConfigService",
        *,
        interval_sec: int,
        probe_timeout_sec: int,
        startup_grace_sec: int,
        stop_timeout_sec: int,
        startup_timeout_sec: int | None = None,
        managed_parse_server: ManagedParseServerConfig | None = None,
        log_cfg: LogConfig | None = None,
    ) -> None:
        self.config_svc = config_svc
        self.interval_sec = interval_sec
        self.probe_timeout_sec = probe_timeout_sec
        self.startup_grace_sec = startup_grace_sec
        self.startup_timeout_sec = (
            startup_timeout_sec if startup_timeout_sec is not None else config.doclib.parse_server_startup_timeout_sec
        )
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
            desired_managed_tier = await get_managed_parse_server_tier(self.config_svc)
            health.managed_tier = desired_managed_tier
            self_hosted_url = await self.config_svc.get("parse_server.local.self_hosted_url")
            health.self_hosted_url = self_hosted_url if self_hosted_url else None

            # probe local
            if health.local_mode != "disabled":
                url = self._local_url(health)
                if url:
                    api_key = None
                    if health.local_mode == "self_hosted":
                        api_key = (await self.config_svc.get("parse_server.local.self_hosted_api_key")) or None
                    probe = await self._probe(url, api_key=api_key)
                    now_ms = int(time.time() * 1000)
                    health.local.url = url
                    health.local.probe = probe
                    health.local.last_probe_at = now_ms
                    if probe.healthy:
                        health.local.last_success_at = now_ms
                        health.local_starting = False
                    else:
                        health.local.last_failure_at = now_ms
                        if probe.error_code != "parse_server_unavailable":
                            health.local_starting = False
                else:
                    health.local = ProbeState()
            else:
                health.local = ProbeState()

            # probe remote
            remote_url = cast(str, await self.config_svc.get("parse_server.remote.url"))
            remote_api_key = (await resolve_remote_api_key(self.config_svc)).value
            probe = await self._probe(remote_url, api_key=remote_api_key)
            now_ms = int(time.time() * 1000)
            health.remote.url = remote_url
            health.remote.probe = probe
            health.remote.last_probe_at = now_ms
            if probe.healthy:
                health.remote.last_success_at = now_ms
            else:
                health.remote.last_failure_at = now_ms

            if health.local_mode == "managed":
                tier_changed = await self._try_restart_managed_for_tier_change(health, desired_managed_tier)
                proc = health.managed_proc
                now = asyncio.get_event_loop().time()
                startup_elapsed_sec = now - health.local_started_at
                if (
                    not tier_changed
                    and health.local_starting
                    and proc is not None
                    and proc.poll() is None
                    and startup_elapsed_sec >= self.startup_grace_sec
                    and startup_elapsed_sec < self.startup_timeout_sec
                ):
                    logger.info(
                        "Managed parse-server is still preparing startup tier %s (PID %d)",
                        health.running_managed_tier,
                        proc.pid,
                    )
                if (
                    not tier_changed
                    and not health.local.probe.healthy
                    and _managed_parse_server_needs_restart(
                        health,
                        now=now,
                        startup_timeout_sec=self.startup_timeout_sec,
                    )
                ):
                    if health.local_starting and proc is not None and proc.poll() is None:
                        logger.warning(
                            "Managed parse-server startup exceeded %ds for tier %s (PID %d), attempting restart",
                            self.startup_timeout_sec,
                            health.running_managed_tier,
                            proc.pid,
                        )
                    else:
                        logger.warning("Managed parse-server is unhealthy, attempting restart")
                    await self._try_restart_managed(health)

            await asyncio.sleep(self.interval_sec)

    async def _try_restart_managed_for_tier_change(
        self,
        health: ParseServerHealth,
        desired_managed_tier: DeploymentTier,
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
            logger.error("Managed parse-server failed %d restarts, disabling", MAX_RESTART_ATTEMPTS)
            health.local_mode = "disabled"
            return

        if count_restart:
            health.restart_count += 1
        managed_tier = await get_managed_parse_server_tier(self.config_svc)
        stop_managed_parse_server(health.managed_proc, timeout_sec=self.stop_timeout_sec, reason=reason)
        health.managed_proc = None
        health.running_managed_tier = None
        try:
            proc, managed_url = start_managed_parse_server(
                tier=managed_tier,
                managed_cfg=self.managed_parse_server,
                log_cfg=self.log_cfg,
                marker=marker or f"restart attempt {health.restart_count}",
            )
            health.managed_url = managed_url
            logger.info("Managed parse-server restarted (PID %d, tier=%s)", proc.pid, managed_tier)
            health.managed_proc = proc
            health.running_managed_tier = managed_tier
            health.local_starting = True
            health.local_started_at = asyncio.get_event_loop().time()
        except Exception as exc:
            logger.error(f"Failed to restart managed parse-server: {exc}")

    async def _probe(self, base_url: str, *, api_key: str | None = None) -> ProbeResult:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            async with httpx.AsyncClient(
                timeout=self.probe_timeout_sec,
                trust_env=should_trust_env_for_url(base_url),
            ) as client:
                resp = await client.get(f"{base_url}/v1/tiers", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    tiers: list[Tier] = []
                    for t in data.get("data", []):
                        tier_id = t.get("id")
                        if tier_id in TIERS:
                            tiers.append(cast(Tier, tier_id))
                    return ProbeResult(healthy=True, tiers=tiers)
                error_code, error_msg = _probe_error(resp)
                return ProbeResult(error_code=error_code, error_msg=error_msg)
        except httpx.TimeoutException as exc:
            return ProbeResult(error_code="parse_server_unavailable", error_msg=str(exc) or "Parse-server probe timed out.")
        except httpx.TransportError as exc:
            return ProbeResult(
                error_code="parse_server_unavailable",
                error_msg=str(exc) or "Parse-server probe transport failed.",
            )
        except Exception as exc:
            return ProbeResult(error_code="parse_server_unavailable", error_msg=str(exc) or "Parse-server probe failed.")

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


def _probe_error(resp: httpx.Response) -> tuple[str, str]:
    data: dict[str, object] = {}
    try:
        loaded = resp.json()
        if isinstance(loaded, dict):
            data = loaded
    except Exception:
        data = {}

    error = data.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "http_error")
        message = str(error.get("message") or error)
        return code, message

    remote_message = _remote_auth_message(data)
    if resp.status_code == 401 or remote_message is not None:
        return "invalid_api_key", remote_message or "API key invalid or remote authentication failed."

    text = resp.text[:500]
    return "http_error", f"HTTP {resp.status_code}: {text}"


def _remote_auth_message(data: dict[str, object]) -> str | None:
    msg_code = data.get("msgCode")
    msg = data.get("msg")
    if msg_code == "A0202":
        return str(msg or "user authenticate failed")
    if isinstance(msg, str) and "authenticate failed" in msg.lower():
        return msg
    return None
