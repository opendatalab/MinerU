"""Process-held ownership lock for a MinerU doclib home."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from ..config import _mineru_home
from .endpoint import ENDPOINT_VERSION, default_endpoint_path, read_endpoint_file

DOCLIB_LOCK_FILENAME = "doclib.lock"


def default_doclib_lock_path() -> Path:
    """Return the non-configurable ownership lock path for this MinerU home."""
    return Path(_mineru_home()).expanduser() / DOCLIB_LOCK_FILENAME


def build_doclib_home_owned_message() -> str:
    """Build the user-facing ownership message from current diagnostics."""
    home = default_doclib_lock_path().parent
    endpoint = read_endpoint_file(default_endpoint_path())
    pid = endpoint.pid if endpoint is not None and endpoint.version == ENDPOINT_VERSION else None
    pid_detail = f" (reported PID {pid})" if pid is not None else ""
    return f"MinerU home [{home}] is currently owned by another doclib server process{pid_detail}."


class DoclibLockUnavailable(RuntimeError):
    """Internal control-flow marker for a doclib ownership conflict."""


@contextmanager
def doclib_home_lock() -> Iterator[Path]:
    """Acquire and retain exclusive ownership of the current MinerU home."""
    lock_path = default_doclib_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise DoclibLockUnavailable() from exc
    try:
        yield lock_path
    finally:
        lock.release()


__all__ = [
    "DOCLIB_LOCK_FILENAME",
    "DoclibLockUnavailable",
    "build_doclib_home_owned_message",
    "default_doclib_lock_path",
    "doclib_home_lock",
]
