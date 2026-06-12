"""mineru server — server lifecycle management."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import typer

from ...config import config
from ..output import format_server_status, print_error, print_info, print_success

app = typer.Typer(help="Server lifecycle management", no_args_is_help=True)


def _socket_path() -> str:
    return config.doclib.uds.path


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

    proc = subprocess.Popen(
        [sys.executable, "-m", "mineru.doclib.app"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for_sock():
        proc.kill()
        print_error("Server failed to start within 15 seconds.")
        raise typer.Exit(1)

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
def status() -> None:
    """Show server status."""
    if not _server_running():
        print_info("Server is not running.")
        return

    try:
        from ...doclib.client import DoclibClient

        c = DoclibClient(timeout=5)
        data = c.get_server_status()
        format_server_status(data)
    except Exception as exc:
        print_error(str(exc))
