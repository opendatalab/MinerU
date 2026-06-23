"""mineru server — server lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import typer

from ...config import config
from ...doclib.endpoint import remove_endpoint_file
from ...doclib.types import ServerStatusResponse, TCPServerStatus
from ...version import __version__
from ..json_errors import exit_with_error
from ..output import format_server_status, print_error, print_info, print_success

app = typer.Typer(help="Server lifecycle management", no_args_is_help=True)


def _socket_path() -> str:
    return config.doclib.uds.path


def _endpoint_path() -> str:
    return config.doclib.endpoint_path


def _server_log_path() -> str:
    return os.path.expanduser(config.doclib.log.path)


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
    if _server_running():
        print_info("Server is already running.")
        return

    log_path = _server_log_path()
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        with _ServerStartLock(_server_start_lock_path()) as start_lock:
            if not start_lock.acquired or _server_running():
                print_info("Server is already running.")
                return

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
                proc = subprocess.Popen(
                    [sys.executable, "-m", "mineru.doclib.app"],
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                )

                if not _wait_for_server():
                    proc.kill()
                    print_error(f"Server failed to start within 15 seconds. See log: {log_path}")
                    raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Server failed to start: {exc}. See log: {log_path}")
        raise typer.Exit(1) from None

    print_success(f"Server started (PID {proc.pid}).")


@app.command()
def stop() -> None:
    """Stop the mineru server gracefully."""
    if not _server_running():
        _cleanup_local_endpoint_files()
        print_info("Server is not running.")
        return

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        c.shutdown_server()
    except Exception:
        pass

    time.sleep(0.5)
    _cleanup_local_endpoint_files()

    print_success("Server stopped.")


@app.command()
def restart() -> None:
    """Restart the mineru server."""
    if _server_running():
        stop()
        time.sleep(1)
    start()


@app.command()
def status(json_mode: bool = typer.Option(False, "--json", help="JSON output")) -> None:
    """Show server status."""
    if not _server_running():
        if json_mode:
            format_server_status(
                ServerStatusResponse(
                    running=False,
                    mineru_home=os.path.expanduser(os.getenv("MINERU_HOME", "~/.mineru")),
                    version=__version__,
                    python_version=sys.version.split()[0],
                    socket_path=_socket_path(),
                    data_dir=os.path.expanduser(config.doclib.data_dir),
                    sqlite_path=os.path.expanduser(config.doclib.sqlite.path),
                    log_path=os.path.expanduser(config.doclib.log.path),
                    tcp=TCPServerStatus(enabled=False, host=None, port=None),
                ),
                json_mode=True,
            )
            return
        print_info("Server is not running.")
        return

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        data = c.get_server_status()
        format_server_status(data, json_mode=json_mode)
    except Exception as exc:
        exit_with_error(exc, json_mode=json_mode)


def _cleanup_local_endpoint_files() -> None:
    try:
        os.unlink(_socket_path())
    except OSError:
        pass
    remove_endpoint_file(_endpoint_path())
