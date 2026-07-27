"""Private parent-child control channel for managed subprocesses."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Callable

logger = logging.getLogger("mineru.managed_process_control")

CONTROL_ENV = "MINERU_MANAGED_CONTROL"
CONTROL_SHUTDOWN = "shutdown"
CONTROL_CONNECT_TIMEOUT_SEC = 30.0
CONTROL_CONNECT_RETRY_SEC = 0.1


def _control_family() -> str:
    return "AF_PIPE" if os.name == "nt" else "AF_UNIX"


def _create_address(family: str) -> tuple[str, Path | None]:
    token = secrets.token_hex(12)
    if family == "AF_PIPE":
        return rf"\\.\pipe\mineru-managed-{token}", None

    temp_dir = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    return str(temp_dir / f"mineru-control-{token}.sock"), None


def _encode_control_config(family: str, address: str, authkey: bytes) -> str:
    return f"{family}:{address}:{authkey.hex()}"


def _decode_control_config(value: str) -> tuple[str, str, bytes]:
    try:
        family, address, authkey_hex = value.split(":", 2)
        authkey = bytes.fromhex(authkey_hex)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid managed process control configuration") from exc
    if family not in {"AF_UNIX", "AF_PIPE"} or not isinstance(address, str) or not address or not authkey:
        raise ValueError("invalid managed process control configuration")
    if not isinstance(family, str):
        raise ValueError("invalid managed process control configuration")
    return family, address, authkey


def cleanup_control_endpoint(family: str, address: str) -> None:
    if family != "AF_UNIX":
        return
    try:
        Path(address).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to remove managed process control socket %s: %s", address, exc)


@dataclass
class ManagedProcessControl:
    """Parent-side listener and command channel for one managed process."""

    listener: Listener
    family: str
    address: str
    authkey: bytes
    cleanup_path: Path | None = None

    def __post_init__(self) -> None:
        self._connection: Connection | None = None
        self._connection_ready = threading.Event()
        self._closed = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_sent = False

    @classmethod
    def create(cls) -> "ManagedProcessControl":
        family = _control_family()
        address, cleanup_path = _create_address(family)
        authkey = secrets.token_bytes(32)
        try:
            listener = Listener(address=address, family=family, authkey=authkey)
        except Exception:
            if cleanup_path is not None:
                cleanup_control_endpoint(family, str(cleanup_path))
            raise
        return cls(listener, family, address, authkey, cleanup_path)

    def child_env(self) -> dict[str, str]:
        return {CONTROL_ENV: _encode_control_config(self.family, self.address, self.authkey)}

    def start_accepting(self) -> None:
        if self._accept_thread is not None:
            return
        self._accept_thread = threading.Thread(
            target=self._accept_connection,
            name="mineru-managed-process-control-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def request_shutdown(self, timeout_sec: float) -> bool:
        with self._lock:
            self._shutdown_requested = True
        if not self._connection_ready.wait(timeout=max(timeout_sec, 0.0)):
            return False
        return self._send_shutdown()

    def _send_shutdown(self) -> bool:
        with self._lock:
            if self._shutdown_sent:
                return True
            connection = self._connection
            if connection is None:
                return False
            try:
                connection.send(CONTROL_SHUTDOWN)
                self._shutdown_sent = True
                return True
            except (EOFError, OSError):
                return False

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self.listener.close()
        except OSError:
            pass
        with self._lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        cleanup_control_endpoint(self.family, self.address)

    def _accept_connection(self) -> None:
        connection: Connection | None = None
        while not self._closed.is_set():
            try:
                connection = self.listener.accept()
                break
            except AuthenticationError:
                logger.warning("Rejected unauthenticated managed process control connection")
            except (OSError, EOFError) as exc:
                if not self._closed.is_set():
                    logger.debug("Managed process control listener stopped before connection: %s", exc)
                return

        if connection is None:
            return
        if self._closed.is_set():
            connection.close()
            return
        with self._lock:
            self._connection = connection
        self._connection_ready.set()
        with self._lock:
            shutdown_requested = self._shutdown_requested
        if shutdown_requested:
            self._send_shutdown()


@dataclass
class ManagedProcessControlWatcher:
    """Child-side watcher for the managed process control channel."""

    on_shutdown: Callable[[], None]
    family: str
    address: str
    authkey: bytes
    connection_timeout_sec: float = CONTROL_CONNECT_TIMEOUT_SEC

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_environment(cls, on_shutdown: Callable[[], None]) -> "ManagedProcessControlWatcher | None":
        value = os.environ.get(CONTROL_ENV)
        if value is None:
            return None
        family, address, authkey = _decode_control_config(value)
        return cls(on_shutdown, family, address, authkey)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="mineru-managed-process-control-watcher",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        connection: Connection | None = None
        try:
            connection = self._connect()
            if connection is None:
                if not self._stop.is_set():
                    logger.error("Managed process control channel could not connect within %.1fs", self.connection_timeout_sec)
                    self.on_shutdown()
                return
            while not self._stop.is_set():
                try:
                    if not connection.poll(CONTROL_CONNECT_RETRY_SEC):
                        continue
                    command = connection.recv()
                except (EOFError, OSError):
                    if not self._stop.is_set():
                        self.on_shutdown()
                    return
                if command == CONTROL_SHUTDOWN:
                    self.on_shutdown()
                    return
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            cleanup_control_endpoint(self.family, self.address)

    def _connect(self) -> Connection | None:
        deadline = time.monotonic() + self.connection_timeout_sec
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                return Client(self.address, family=self.family, authkey=self.authkey)
            except (OSError, EOFError):
                time.sleep(CONTROL_CONNECT_RETRY_SEC)
        return None


__all__ = [
    "CONTROL_ENV",
    "ManagedProcessControl",
    "ManagedProcessControlWatcher",
]
