"""mineru server — server lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...config import config
from ...doclib.endpoint import read_endpoint_file
from ...doclib.instance_lock import DoclibLockUnavailable, build_doclib_home_owned_message, doclib_home_lock
from ...doclib.types import ServerStatusResponse, TCPServerStatus
from ...errors import MineruError
from ...utils.i18n import t
from ...utils.stdio import utf8_subprocess_env
from ...version import __version__
from ..contracts import CliContext, RenderableObject
from ..runtime import run_cli

app = typer.Typer(help=t("Server lifecycle management"), no_args_is_help=True)

SERVER_START_TIMEOUT_SEC = 30.0


@dataclass(frozen=True)
class _ServerStartingStatus:
    status: str = "starting"
    pid: int | None = None


def _socket_path() -> str:
    return config.doclib.uds.path


def _endpoint_path() -> str:
    return config.doclib.endpoint_path


def _server_log_path() -> str:
    return os.path.expanduser(config.doclib.log.resolved_app_path)


def _server_stdout_log_path() -> str:
    return os.path.expanduser(config.doclib.log.resolved_stdout_path)


def _server_stderr_log_path() -> str:
    return os.path.expanduser(config.doclib.log.resolved_stderr_path)


def _server_start_lock_path() -> str:
    endpoint_dir = os.path.dirname(os.path.expanduser(_endpoint_path()))
    return os.path.join(endpoint_dir or ".", "doclib.start.lock")


class _ServerStartLock:
    def __init__(self, path: str, *, timeout: float = 20.0, stale_after: float = 60.0) -> None:
        self._path = os.path.expanduser(path)
        self._timeout = timeout
        self._stale_after = stale_after
        self._token = f"{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}"
        self._fd: int | None = None
        self.acquired = False

    def __enter__(self) -> "_ServerStartLock":
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            try:
                self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if _server_running():
                    return self
                stale_token = self._stale_token()
                if stale_token is not None:
                    self._remove_if_token(stale_token)
                    continue
                time.sleep(0.2)
            else:
                self.acquired = True
                self._write_owner()
                return self

        if _server_running():
            return self
        raise RuntimeError(t("Another mineru server start is already in progress."))

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.acquired:
            self._close_fd()
            self._remove_if_token(self._token)

    def _stale_token(self) -> str | None:
        if not self._is_stale_path(self._path):
            return None
        return self._read_token()

    def _is_stale_path(self, path: str) -> bool:
        try:
            return time.time() - os.path.getmtime(path) > self._stale_after
        except OSError:
            return False

    def _write_owner(self) -> None:
        try:
            if self._fd is None:
                return
            payload = f"token={self._token}\npid={os.getpid()}\ncreated_at={int(time.time())}\n"
            os.write(self._fd, payload.encode("utf-8"))
            os.fsync(self._fd)
        except OSError:
            pass

    def _read_token(self) -> str | None:
        try:
            with open(self._path, encoding="utf-8") as f:
                first_line = f.readline().strip()
        except OSError:
            return None
        if not first_line.startswith("token="):
            return None
        return first_line.removeprefix("token=")

    def _remove_if_token(self, expected_token: str | None) -> None:
        if expected_token is None:
            return
        if self._read_token() != expected_token:
            return
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def _close_fd(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        finally:
            self._fd = None


def _server_running() -> bool:
    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=3)
        c.get_server_status()
        return True
    except Exception:
        return False


def _doclib_lock_available() -> bool:
    try:
        with doclib_home_lock():
            return True
    except DoclibLockUnavailable:
        return False


def _home_owner_unavailable_error() -> MineruError:
    return MineruError("service_unavailable", build_doclib_home_owned_message())


def _wait_for_started_server(proc: subprocess.Popen[bytes], timeout: float = SERVER_START_TIMEOUT_SEC) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_running():
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.3)
    return False


def _wait_for_server_stop(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _server_running():
            try:
                with doclib_home_lock():
                    if _server_running():
                        continue
                    return True
            except DoclibLockUnavailable:
                pass
        time.sleep(0.3)
    return False


@app.command(help=t("Start the mineru server in the background."))
def start() -> None:
    """Start the mineru server in the background."""
    run_cli(CliContext(json_mode=False), _start)


def _start() -> str:
    if _server_running():
        return t("Server is already running.")

    log_path = _server_log_path()
    stdout_log_path = _server_stdout_log_path()
    stderr_log_path = _server_stderr_log_path()
    _ensure_log_dir(log_path)
    _ensure_log_dir(stdout_log_path)
    _ensure_log_dir(stderr_log_path)

    try:
        with _ServerStartLock(_server_start_lock_path()) as start_lock:
            if not start_lock.acquired or _server_running():
                return t("Server is already running.")
            if not _doclib_lock_available():
                raise _home_owner_unavailable_error()

            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\n--- mineru server start ---\n")
                log_file.flush()
            with (
                open(stdout_log_path, "a", encoding="utf-8") as stdout_log_file,
                open(stderr_log_path, "a", encoding="utf-8") as stderr_log_file,
            ):
                stdout_log_file.write("\n--- mineru server stdout ---\n")
                stderr_log_file.write("\n--- mineru server stderr ---\n")
                stdout_log_file.flush()
                stderr_log_file.flush()
                proc = subprocess.Popen(
                    [sys.executable, "-m", "mineru.doclib.app"],
                    stdout=stdout_log_file,
                    stderr=stderr_log_file,
                    start_new_session=True,
                    env=utf8_subprocess_env(),
                )

                if not _wait_for_started_server(proc):
                    if proc.poll() is None:
                        return f"{t('Server is still starting (PID {pid}).', pid=proc.pid)}\nCheck status: mineru server status"
                    raise MineruError(
                        "service_unavailable",
                        t(
                            "Server failed to start within {seconds} seconds. "
                            "See log: {log}; stdout: {stdout}; stderr: {stderr}",
                            seconds=int(SERVER_START_TIMEOUT_SEC),
                            log=log_path,
                            stdout=stdout_log_path,
                            stderr=stderr_log_path,
                        ),
                    )
    except MineruError:
        raise
    except Exception as exc:
        raise MineruError(
            "service_unavailable",
            t(
                "Server failed to start: {error}. See log: {log}; stdout: {stdout}; stderr: {stderr}",
                error=exc,
                log=log_path,
                stdout=stdout_log_path,
                stderr=stderr_log_path,
            ),
        ) from exc

    return t("Server started (PID {pid}).", pid=proc.pid)


@app.command(help=t("Stop the mineru server gracefully."))
def stop() -> None:
    """Stop the mineru server gracefully."""
    run_cli(CliContext(json_mode=False), _stop)


def _stop() -> str:
    if not _server_running():
        if not _doclib_lock_available():
            raise _home_owner_unavailable_error()
        return t("Server is not running.")

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        c.shutdown_server()
    except Exception as exc:
        raise MineruError("service_unavailable", t("Failed to request MinerU server shutdown: {error}", error=exc)) from exc

    if not _wait_for_server_stop():
        raise MineruError(
            "service_unavailable",
            t("MinerU server did not stop within 15 seconds. The server was not restarted."),
        )

    return t("Server stopped.")


@app.command(help=t("Restart the mineru server."))
def restart() -> None:
    """Restart the mineru server."""
    run_cli(CliContext(json_mode=False), _restart)


def _restart() -> str:
    if _server_running():
        stop_message = _stop()
        start_message = _start()
        return f"{stop_message}\n{start_message}"
    return _start()


@app.command(help=t("Show server status."))
def status(json_mode: bool = typer.Option(False, "--json", help=t("JSON output"))) -> None:
    """Show server status."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, _server_status, render=_render_server_status)


def _server_status() -> ServerStatusResponse | _ServerStartingStatus:
    if not _server_running():
        if not _doclib_lock_available():
            endpoint = read_endpoint_file(_endpoint_path())
            return _ServerStartingStatus(pid=endpoint.pid if endpoint is not None else None)
        return _not_running_status()
    from ...doclib.client import DoclibClient

    c = DoclibClient(timeout=5)
    return c.get_server_status()


def _render_server_status(data: ServerStatusResponse | _ServerStartingStatus) -> Iterator[RenderableObject]:
    if isinstance(data, _ServerStartingStatus):
        if data.pid is None:
            yield t("Server is still starting.")
        else:
            yield t("Server is still starting (PID {pid}).", pid=data.pid)
        yield "Check again: mineru server status"
        return
    if not _get(data, "running"):
        yield t("Server is not running.")
        return

    table = Table(title=t("MinerU Server"))
    table.add_column(t("Field"), style="cyan")
    table.add_column(t("Current"), style="green")
    table.add_row(t("PID"), str(_get(data, "pid", "?")))
    table.add_row(t("Uptime"), f"{_get(data, 'uptime_seconds', 0):.0f}s")
    table.add_row(t("Home"), _get(data, "mineru_home", ""))
    table.add_row(t("Version"), _get(data, "version", ""))
    table.add_row(t("Python"), _get(data, "python_version", ""))
    table.add_row(t("Socket"), _get(data, "socket_path", ""))
    table.add_row(t("Data dir"), _get(data, "data_dir", ""))
    table.add_row(t("SQLite"), _get(data, "sqlite_path", ""))
    table.add_row(t("SQLite size"), _format_bytes(_get(data, "sqlite_size_bytes")))
    table.add_row(t("Log"), _get(data, "log_path", ""))
    tcp_data = _get(data, "tcp")
    tcp_enabled = bool(_get(tcp_data, "enabled", False))
    tcp_host = _get(tcp_data, "host", "") or "-"
    tcp_port = _get(tcp_data, "port")
    if tcp_enabled:
        tcp_value = f"http://{tcp_host}:{tcp_port}" if tcp_port is not None else f"http://{tcp_host}:{t('(pending)')}"
    else:
        tcp_value = t("disabled")
    table.add_row(t("TCP"), tcp_value)
    table.add_row(t("Files tracked"), str(_get(data, "files_total", 0)))
    table.add_row(t("Docs indexed"), str(_get(data, "docs_total", 0)))
    table.add_row(t("Active scans"), str(_get(data, "active_scan_count", 0)))
    table.add_row(t("Last scan"), _format_timestamp_ms(_get(data, "last_scan_at")))
    table.add_row(t("Parse queue"), str(_get(data, "parse_queue_length", 0)))
    table.add_row(t("Ingest queue"), str(_get(data, "ingest_queue_length", 0)))
    table.add_row(t("Watches"), str(_get(data, "watch_count", 0)))
    yield table

    workers = _get(data, "workers")
    if workers:
        worker_table = Table(title=t("Workers"))
        worker_table.add_column(t("Component"), style="cyan")
        worker_table.add_column(t("Running"), style="green")
        worker_table.add_column(t("Workers"), justify="right")
        worker_table.add_row(t("Watch"), t("yes") if _get(workers, "watch_running", False) else t("no"), "-")
        worker_table.add_row(
            t("Scan"),
            t("yes") if _get(workers, "scan_running", False) else t("no"),
            str(_get(workers, "scan_workers", 0)),
        )
        worker_table.add_row(
            t("Ingest"),
            t("yes") if _get(workers, "ingest_running", False) else t("no"),
            str(_get(workers, "ingest_workers", 0)),
        )
        worker_table.add_row(
            t("Parse"),
            t("yes") if _get(workers, "parse_running", False) else t("no"),
            str(_get(workers, "parse_workers", 0)),
        )
        worker_table.add_row(
            t("Device monitor"),
            t("yes") if _get(workers, "device_monitor_running", False) else t("no"),
            "-",
        )
        worker_table.add_row(
            t("Compaction"),
            t("yes") if _get(workers, "compaction_running", False) else t("no"),
            "-",
        )
        worker_table.add_row(
            t("Health check"),
            t("yes") if _get(workers, "health_check_running", False) else t("no"),
            "-",
        )
        yield worker_table

    watch_stats = _get(data, "watch_stats", [])
    if watch_stats:
        watch_table = Table(title=t("Watch Stats"))
        watch_table.add_column(t("Path"), style="cyan", no_wrap=True)
        watch_table.add_column(t("Status"), style="green")
        watch_table.add_column(t("Files"), justify="right")
        watch_table.add_column(t("Active"), justify="right")
        watch_table.add_column(t("Deleted"), justify="right")
        watch_table.add_column(t("Unreachable"), justify="right")
        watch_table.add_column(t("Pending ingest"), justify="right")
        watch_table.add_column(t("Errors"), justify="right")
        watch_table.add_column(t("Docs"), justify="right")
        watch_table.add_column(t("Parses done/pending/parsing/failed"), justify="right")
        for item in watch_stats:
            parse_counts = (
                f"{_get(item, 'parse_done_count', 0)}/"
                f"{_get(item, 'parse_pending_count', 0)}/"
                f"{_get(item, 'parse_parsing_count', 0)}/"
                f"{_get(item, 'parse_failed_count', 0)}"
            )
            watch_table.add_row(
                _get(item, "path", ""),
                _get(item, "status", ""),
                str(_get(item, "total_files", 0)),
                str(_get(item, "active_files", 0)),
                str(_get(item, "deleted_files", 0)),
                str(_get(item, "unreachable_files", 0)),
                str(_get(item, "pending_ingest_files", 0)),
                str(_get(item, "file_error_count", 0)),
                str(_get(item, "doc_count", 0)),
                parse_counts,
            )
        yield watch_table

    error_rows = _error_summary_rows(_get(data, "error_summary"))
    if error_rows:
        error_table = Table(title=t("Error Summary"))
        error_table.add_column(t("Scope"), style="cyan")
        error_table.add_column(t("Code"), style="red")
        error_table.add_column(t("Count"), justify="right")
        for scope, code, count in error_rows:
            error_table.add_row(scope, code, str(count))
        yield error_table

    recent_scans = _get(data, "recent_scans", [])
    if recent_scans:
        scan_table = Table(title=t("Recent Scans"))
        scan_table.add_column(t("ID"), justify="right")
        scan_table.add_column(t("Kind"), style="cyan")
        scan_table.add_column(t("Source"))
        scan_table.add_column(t("Status"), style="green")
        scan_table.add_column(t("Path"), no_wrap=True)
        scan_table.add_column(t("Seen"), justify="right")
        scan_table.add_column(t("New"), justify="right")
        scan_table.add_column(t("Changed"), justify="right")
        scan_table.add_column(t("Deleted"), justify="right")
        scan_table.add_column(t("Errors"), justify="right")
        scan_table.add_column(t("Error code"), style="red")
        for item in recent_scans:
            scan_table.add_row(
                str(_get(item, "id", "")),
                _get(item, "kind", ""),
                _get(item, "source", ""),
                _get(item, "status", ""),
                _get(item, "path", ""),
                str(_get(item, "files_seen", 0)),
                str(_get(item, "files_new", 0)),
                str(_get(item, "files_changed", 0)),
                str(_get(item, "files_deleted", 0)),
                str(_get(item, "files_error", 0)),
                _get(item, "error_code", "") or "",
            )
        yield scan_table

    ps_data = _get(data, "parse_server")
    if ps_data:
        ps_table = Table(title=t("Parse Server"))
        ps_table.add_column(t("Target"), style="cyan")
        ps_table.add_column(t("Healthy"), style="green")
        ps_table.add_column(t("Endpoint"), style="dim")
        ps_table.add_column(t("Managed"), style="dim")
        ps_table.add_column(t("Restart"), justify="right")
        ps_table.add_column(t("Last probe"), style="dim")
        ps_table.add_column(t("Last ok"), style="dim")
        ps_table.add_column(t("Last fail"), style="dim")
        ps_table.add_column(t("Tiers"), style="green")
        for label, key in [(t("Local"), "local"), (t("Remote"), "remote")]:
            ps = _get(ps_data, key, {})
            if _get(ps, "starting"):
                healthy_str = t("starting")
            elif _get(ps, "healthy"):
                healthy_str = t("yes")
            else:
                healthy_str = t("no")
            tiers_str = ", ".join(_get(ps, "supported_tiers", [])) or "-"
            mode = _get(ps, "mode", "")
            label_str = f"{label} ({mode})" if mode else label
            endpoint = _get(ps, "url", "") or "-"
            managed_str = "-"
            restart_str = "-"
            if key == "local":
                mode_text = _get(ps, "mode", "")
                if mode_text == "managed":
                    managed_tier = _get(ps, "managed_tier", "") or "-"
                    managed_pid = _get(ps, "managed_pid")
                    managed_running = t("yes") if _get(ps, "managed_running", False) else t("no")
                    managed_str = f"tier={managed_tier}, pid={managed_pid or '-'}, running={managed_running}"
                elif mode_text == "self_hosted":
                    managed_str = _get(ps, "self_hosted_url", "") or "-"
                restart_str = f"{_get(ps, 'restart_count', 0)}/{_get(ps, 'max_restart_attempts', 0)}"
            ps_table.add_row(
                label_str,
                healthy_str,
                endpoint,
                managed_str,
                restart_str,
                _format_age_ms(_get(ps, "last_probe_at")),
                _format_age_ms(_get(ps, "last_success_at")),
                _format_age_ms(_get(ps, "last_failure_at")),
                tiers_str,
            )
        yield ps_table

    for title, key in (
        (t("Recent App Logs"), "app_logs"),
        (t("Recent Access Logs"), "access_logs"),
        (t("Recent Stderr Logs"), "stderr_logs"),
        (t("Recent Stdout Logs"), "stdout_logs"),
        (t("Recent Parse Server Stderr Logs"), "parse_server_stderr_logs"),
        (t("Recent Parse Server Stdout Logs"), "parse_server_stdout_logs"),
    ):
        logs = _get(data, key, [])
        if logs:
            log_text = "".join(logs)
            panel = Panel(Text(log_text.strip() or t("(empty)")), title=title, border_style="dim")
            yield panel


def _not_running_status() -> ServerStatusResponse:
    return ServerStatusResponse(
        running=False,
        mineru_home=os.path.expanduser(os.getenv("MINERU_HOME", "~/.mineru")),
        version=__version__,
        python_version=sys.version.split()[0],
        socket_path=_socket_path(),
        data_dir=os.path.expanduser(config.doclib.data_dir),
        sqlite_path=os.path.expanduser(config.doclib.sqlite.path),
        log_path=os.path.expanduser(config.doclib.log.resolved_app_path),
        access_log_path=os.path.expanduser(config.doclib.log.resolved_access_path),
        stdout_log_path=os.path.expanduser(config.doclib.log.resolved_stdout_path),
        stderr_log_path=os.path.expanduser(config.doclib.log.resolved_stderr_path),
        tcp=TCPServerStatus(enabled=False, host=None, port=None),
    )


def _get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_timestamp_ms(ts: int | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1000))


def _format_age_ms(ts: int | None) -> str:
    if ts is None:
        return "-"
    age = max(0.0, time.time() - ts / 1000)
    if age < 60:
        return t("{age}s ago", age=f"{age:.0f}")
    if age < 3600:
        return t("{age}m ago", age=f"{age / 60:.0f}")
    if age < 86400:
        return t("{age}h ago", age=f"{age / 3600:.0f}")
    return t("{age}d ago", age=f"{age / 86400:.0f}")


def _error_summary_rows(error_summary: Any) -> list[tuple[str, str, int]]:
    if not error_summary:
        return []
    rows: list[tuple[str, str, int]] = []
    for scope, attr in (("file", "file_errors"), ("doc", "doc_errors"), ("parse", "parse_errors")):
        for bucket in _get(error_summary, attr, []):
            rows.append((scope, _get(bucket, "code", ""), int(_get(bucket, "count", 0))))
    return rows


def _ensure_log_dir(path: str) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
