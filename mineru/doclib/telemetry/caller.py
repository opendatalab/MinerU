"""Best-effort caller/source inference for telemetry."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from .constants import TelemetryCaller

_AGENT_PROCESS_MARKERS = (
    "claude",
    "codex",
    "cursor",
    "cline",
    "copilot",
    "gemini",
    "goose",
    "opencode",
    "roo",
    "trae",
    "windsurf",
)
_MAX_PARENT_DEPTH = 8


def infer_caller_from_process_tree(max_depth: int = _MAX_PARENT_DEPTH) -> TelemetryCaller:
    """Infer whether the current process was launched by a common agent tool.

    The implementation intentionally inspects process names only. It avoids
    command-line arguments, paths, and environment variables because those can
    contain user data.
    """
    names = _parent_process_names(max_depth)
    if any(_is_agent_name(name) for name in names):
        return "agent"
    if sys.stdin is not None and sys.stdin.isatty():
        return "user"
    return "unknown"


def _parent_process_names(max_depth: int) -> list[str]:
    names = _parent_process_names_psutil(max_depth)
    if names:
        return names
    if platform.system() == "Linux":
        return _parent_process_names_procfs(max_depth)
    return []


def _parent_process_names_psutil(max_depth: int) -> list[str]:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return []

    try:
        proc = psutil.Process(os.getpid())
    except Exception:
        return []

    names: list[str] = []
    for _ in range(max(0, max_depth)):
        try:
            proc = proc.parent()
        except Exception:
            break
        if proc is None:
            break
        try:
            name = str(proc.name() or "").strip().lower()
        except Exception:
            name = ""
        if name:
            names.append(name)
    return names


def _parent_process_names_procfs(max_depth: int) -> list[str]:
    names: list[str] = []
    pid = os.getpid()
    for _ in range(max(0, max_depth)):
        stat = _read_proc_stat(pid)
        if stat is None:
            break
        parent_pid, name = stat
        if name:
            names.append(name.lower())
        if parent_pid <= 1 or parent_pid == pid:
            break
        pid = parent_pid
    return names


def _read_proc_stat(pid: int) -> tuple[int, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    left = raw.find("(")
    right = raw.rfind(")")
    if left < 0 or right <= left:
        return None
    name = raw[left + 1 : right]
    rest = raw[right + 2 :].split()
    if len(rest) < 2:
        return None
    try:
        parent_pid = int(rest[1])
    except ValueError:
        return None
    return parent_pid, name


def _is_agent_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _AGENT_PROCESS_MARKERS)


__all__ = ["infer_caller_from_process_tree"]
