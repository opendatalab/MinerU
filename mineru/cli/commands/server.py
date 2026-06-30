"""mineru server — server lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from ...config import config
from ...doclib.endpoint import remove_endpoint_file
from ...doclib.types import ServerStatusResponse, TCPServerStatus
from ...errors import MineruError
from ...version import __version__
from ..contracts import CliContext, RenderableObject
from ..runtime import run_cli

app = typer.Typer(help="Server lifecycle management", no_args_is_help=True)


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
        raise RuntimeError("Another mineru server start is already in progress.")

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


def _wait_for_server(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_running():
            return True
        time.sleep(0.3)
    return False


def _wait_for_sock(timeout: float = 15.0) -> bool:
    return _wait_for_server(timeout)


@app.command()
def start() -> None:
    """Start the mineru server in the background."""
    run_cli(CliContext(json_mode=False), _start)


def _start() -> str:
    if _server_running():
        return "Server is already running."

    log_path = _server_log_path()
    stdout_log_path = _server_stdout_log_path()
    stderr_log_path = _server_stderr_log_path()
    _ensure_log_dir(log_path)
    _ensure_log_dir(stdout_log_path)
    _ensure_log_dir(stderr_log_path)

    try:
        with _ServerStartLock(_server_start_lock_path()) as start_lock:
            if not start_lock.acquired or _server_running():
                return "Server is already running."

            socket_path = _socket_path()
            # Clean stale socket
            try:
                os.unlink(socket_path)
            except OSError:
                pass
            remove_endpoint_file(_endpoint_path())

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
                )

                if not _wait_for_server():
                    proc.kill()
                    raise MineruError(
                        "service_unavailable",
                        "Server failed to start within 15 seconds. "
                        f"See log: {log_path}; stdout: {stdout_log_path}; stderr: {stderr_log_path}",
                    )
    except MineruError:
        raise
    except Exception as exc:
        raise MineruError(
            "service_unavailable",
            f"Server failed to start: {exc}. See log: {log_path}; stdout: {stdout_log_path}; stderr: {stderr_log_path}",
        ) from exc

    return f"Server started (PID {proc.pid})."


@app.command()
def stop() -> None:
    """Stop the mineru server gracefully."""
    run_cli(CliContext(json_mode=False), _stop)


def _stop() -> str:
    if not _server_running():
        _cleanup_local_endpoint_files()
        return "Server is not running."

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        c.shutdown_server()
    except Exception:
        pass

    time.sleep(0.5)
    _cleanup_local_endpoint_files()

    return "Server stopped."


@app.command()
def restart() -> None:
    """Restart the mineru server."""
    run_cli(CliContext(json_mode=False), _restart)


def _restart() -> str:
    if _server_running():
        stop_message = _stop()
        time.sleep(1)
        start_message = _start()
        return f"{stop_message}\n{start_message}"
    return _start()


@app.command()
def status(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show server status."""
    ctx = CliContext(json_mode=json_mode)
    run_cli(ctx, _server_status, render=_render_server_status)


def _server_status() -> ServerStatusResponse:
    if not _server_running():
        return _not_running_status()
    from ...doclib.client import DoclibClient

    c = DoclibClient(timeout=5)
    return c.get_server_status()


def _render_server_status(data: ServerStatusResponse) -> Iterator[RenderableObject]:
    if not _get(data, "running"):
        yield "Server is not running."
        return

    table = Table(title="MinerU Server")
    table.add_column("Field", style="cyan")
    table.add_column("Current", style="green")
    table.add_row("PID", str(_get(data, "pid", "?")))
    table.add_row("Uptime", f"{_get(data, 'uptime_seconds', 0):.0f}s")
    table.add_row("Home", _get(data, "mineru_home", ""))
    table.add_row("Version", _get(data, "version", ""))
    table.add_row("Python", _get(data, "python_version", ""))
    table.add_row("Socket", _get(data, "socket_path", ""))
    table.add_row("Data dir", _get(data, "data_dir", ""))
    table.add_row("SQLite", _get(data, "sqlite_path", ""))
    table.add_row("SQLite size", _format_bytes(_get(data, "sqlite_size_bytes")))
    table.add_row("Log", _get(data, "log_path", ""))
    tcp_data = _get(data, "tcp")
    tcp_enabled = bool(_get(tcp_data, "enabled", False))
    tcp_host = _get(tcp_data, "host", "") or "-"
    tcp_port = _get(tcp_data, "port")
    if tcp_enabled:
        tcp_value = f"http://{tcp_host}:{tcp_port}" if tcp_port is not None else f"http://{tcp_host}:(pending)"
    else:
        tcp_value = "disabled"
    table.add_row("TCP", tcp_value)
    table.add_row("Files tracked", str(_get(data, "files_total", 0)))
    table.add_row("Docs indexed", str(_get(data, "docs_total", 0)))
    table.add_row("Active scans", str(_get(data, "active_scan_count", 0)))
    table.add_row("Last scan", _format_timestamp_ms(_get(data, "last_scan_at")))
    table.add_row("Parse queue", str(_get(data, "parse_queue_length", 0)))
    table.add_row("Ingest queue", str(_get(data, "ingest_queue_length", 0)))
    table.add_row("Watches", str(_get(data, "watch_count", 0)))
    yield table

    workers = _get(data, "workers")
    if workers:
        worker_table = Table(title="Workers")
        worker_table.add_column("Component", style="cyan")
        worker_table.add_column("Running", style="green")
        worker_table.add_column("Workers", justify="right")
        worker_table.add_row("Watch", "yes" if _get(workers, "watch_running", False) else "no", "-")
        worker_table.add_row(
            "Scan",
            "yes" if _get(workers, "scan_running", False) else "no",
            str(_get(workers, "scan_workers", 0)),
        )
        worker_table.add_row(
            "Ingest",
            "yes" if _get(workers, "ingest_running", False) else "no",
            str(_get(workers, "ingest_workers", 0)),
        )
        worker_table.add_row(
            "Parse",
            "yes" if _get(workers, "parse_running", False) else "no",
            str(_get(workers, "parse_workers", 0)),
        )
        worker_table.add_row(
            "Device monitor",
            "yes" if _get(workers, "device_monitor_running", False) else "no",
            "-",
        )
        worker_table.add_row(
            "Compaction",
            "yes" if _get(workers, "compaction_running", False) else "no",
            "-",
        )
        worker_table.add_row(
            "Health check",
            "yes" if _get(workers, "health_check_running", False) else "no",
            "-",
        )
        yield worker_table

    watch_stats = _get(data, "watch_stats", [])
    if watch_stats:
        watch_table = Table(title="Watch Stats")
        watch_table.add_column("Path", style="cyan", no_wrap=True)
        watch_table.add_column("Status", style="green")
        watch_table.add_column("Files", justify="right")
        watch_table.add_column("Active", justify="right")
        watch_table.add_column("Deleted", justify="right")
        watch_table.add_column("Unreachable", justify="right")
        watch_table.add_column("Pending ingest", justify="right")
        watch_table.add_column("Errors", justify="right")
        watch_table.add_column("Docs", justify="right")
        watch_table.add_column("Parses done/pending/parsing/failed", justify="right")
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
        error_table = Table(title="Error Summary")
        error_table.add_column("Scope", style="cyan")
        error_table.add_column("Code", style="red")
        error_table.add_column("Count", justify="right")
        for scope, code, count in error_rows:
            error_table.add_row(scope, code, str(count))
        yield error_table

    recent_scans = _get(data, "recent_scans", [])
    if recent_scans:
        scan_table = Table(title="Recent Scans")
        scan_table.add_column("ID", justify="right")
        scan_table.add_column("Kind", style="cyan")
        scan_table.add_column("Source")
        scan_table.add_column("Status", style="green")
        scan_table.add_column("Path", no_wrap=True)
        scan_table.add_column("Seen", justify="right")
        scan_table.add_column("New", justify="right")
        scan_table.add_column("Changed", justify="right")
        scan_table.add_column("Deleted", justify="right")
        scan_table.add_column("Errors", justify="right")
        scan_table.add_column("Error code", style="red")
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
        ps_table = Table(title="Parse Server")
        ps_table.add_column("Target", style="cyan")
        ps_table.add_column("Healthy", style="green")
        ps_table.add_column("Endpoint", style="dim")
        ps_table.add_column("Managed", style="dim")
        ps_table.add_column("Restart", justify="right")
        ps_table.add_column("Last probe", style="dim")
        ps_table.add_column("Last ok", style="dim")
        ps_table.add_column("Last fail", style="dim")
        ps_table.add_column("Tiers", style="green")
        for label, key in [("Local", "local"), ("Remote", "remote")]:
            ps = _get(ps_data, key, {})
            if _get(ps, "starting"):
                healthy_str = "starting"
            elif _get(ps, "healthy"):
                healthy_str = "yes"
            else:
                healthy_str = "no"
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
                    managed_running = "yes" if _get(ps, "managed_running", False) else "no"
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
        ("Recent App Logs", "app_logs"),
        ("Recent Access Logs", "access_logs"),
        ("Recent Stderr Logs", "stderr_logs"),
        ("Recent Stdout Logs", "stdout_logs"),
    ):
        logs = _get(data, key, [])
        if logs:
            log_text = "".join(logs)
            panel = Panel(log_text.strip() or "(empty)", title=title, border_style="dim")
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
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.0f}m ago"
    if age < 86400:
        return f"{age / 3600:.0f}h ago"
    return f"{age / 86400:.0f}d ago"


def _error_summary_rows(error_summary: Any) -> list[tuple[str, str, int]]:
    if not error_summary:
        return []
    rows: list[tuple[str, str, int]] = []
    for scope, attr in (("file", "file_errors"), ("doc", "doc_errors"), ("parse", "parse_errors")):
        for bucket in _get(error_summary, attr, []):
            rows.append((scope, _get(bucket, "code", ""), int(_get(bucket, "count", 0))))
    return rows


def _cleanup_local_endpoint_files() -> None:
    try:
        os.unlink(_socket_path())
    except OSError:
        pass
    remove_endpoint_file(_endpoint_path())


def _ensure_log_dir(path: str) -> None:
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
