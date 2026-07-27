from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mineru.utils.managed_process_control import (
    CONTROL_ENV,
    ManagedProcessControl,
    ManagedProcessControlWatcher,
)


def test_managed_process_control_sends_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    control = ManagedProcessControl.create()
    shutdown = threading.Event()
    monkeypatch.setenv(CONTROL_ENV, control.child_env()[CONTROL_ENV])
    watcher = ManagedProcessControlWatcher.from_environment(shutdown.set)
    assert watcher is not None

    control.start_accepting()
    watcher.start()
    try:
        assert control.request_shutdown(timeout_sec=2.0) is True
        assert shutdown.wait(timeout=2.0) is True
    finally:
        watcher.close()
        control.close()
    if control.family == "AF_UNIX":
        assert not Path(control.address).exists()


def test_managed_process_control_eof_requests_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    control = ManagedProcessControl.create()
    shutdown = threading.Event()
    monkeypatch.setenv(CONTROL_ENV, control.child_env()[CONTROL_ENV])
    watcher = ManagedProcessControlWatcher.from_environment(shutdown.set)
    assert watcher is not None

    control.start_accepting()
    watcher.start()
    assert control._connection_ready.wait(timeout=2.0)
    control.close()
    try:
        assert shutdown.wait(timeout=2.0) is True
    finally:
        watcher.close()


def test_managed_process_control_watcher_is_absent_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONTROL_ENV, raising=False)

    assert ManagedProcessControlWatcher.from_environment(lambda: None) is None


def test_managed_process_control_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONTROL_ENV, "AF_UNIX:/tmp/only-two-fields")

    with pytest.raises(ValueError, match="invalid managed process control configuration"):
        ManagedProcessControlWatcher.from_environment(lambda: None)


def test_managed_process_control_works_across_subprocess() -> None:
    control = ManagedProcessControl.create()
    env = os.environ.copy()
    env.update(control.child_env())
    child_code = """
import sys
import threading
from mineru.utils.managed_process_control import ManagedProcessControlWatcher

shutdown = threading.Event()
watcher = ManagedProcessControlWatcher.from_environment(shutdown.set)
if watcher is None:
    sys.exit(2)
watcher.start()
if not shutdown.wait(timeout=5):
    sys.exit(3)
watcher.close()
"""

    control.start_accepting()
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        assert control.request_shutdown(timeout_sec=3.0) is True
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        control.close()
