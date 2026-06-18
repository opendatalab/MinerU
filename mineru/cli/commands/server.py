"""mineru server — server lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import typer

from ...config import config
from ...doclib.types import ServerStatusResponse
from ...version import __version__
from ..json_errors import exit_with_error
from ..output import format_server_status, print_error, print_info, print_success

app = typer.Typer(help="Server lifecycle management", no_args_is_help=True)


def _socket_path() -> str:
    return config.doclib.uds.path


def _server_log_path() -> str:
    return os.path.expanduser(config.doclib.log.path)


def _server_running() -> bool:
    socket_path = _socket_path()
    if not os.path.exists(socket_path):
        return False
    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=3)
        c.get_server_status()
        return True
    except Exception:
        return False


def _wait_for_sock(timeout: float = 15.0) -> bool:
    socket_path = _socket_path()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(socket_path):
            time.sleep(0.3)
            if _server_running():
                return True
        time.sleep(0.3)
    return False


@app.command()
def start() -> None:
    """Start the mineru server in the background."""
    if _server_running():
        print_info("Server is already running.")
        return

    socket_path = _socket_path()
    # Clean stale socket
    try:
        os.unlink(socket_path)
    except OSError:
        pass

    log_path = _server_log_path()
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n--- mineru server start ---\n")
            log_file.flush()
            proc = subprocess.Popen(
                [sys.executable, "-m", "mineru.doclib.app"],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )

            if not _wait_for_sock():
                proc.kill()
                print_error(f"Server failed to start within 15 seconds. See log: {log_path}")
                raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as exc:
        print_error(f"Server failed to start: {exc}. See log: {log_path}")
        raise typer.Exit(1) from None

    print_success(f"Server started (PID {proc.pid}). Socket: {socket_path}")


@app.command()
def stop() -> None:
    """Stop the mineru server gracefully."""
    if not _server_running():
        print_info("Server is not running.")
        return

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        c.shutdown_server()
    except Exception:
        pass

    time.sleep(0.5)
    try:
        os.unlink(_socket_path())
    except OSError:
        pass

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
