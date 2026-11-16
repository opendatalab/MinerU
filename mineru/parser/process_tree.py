"""在有限预算内回收托管进程及其模型子进程。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .process_control import ManagedProcessControl
from .windows_process_job import WindowsProcessJob


def process_group_options(*, new_session: bool = True) -> dict[str, bool | int]:
    """隔离进程组；嵌套 worker 保留 Router 会话归属，父进程退出后仍可定位。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True} if new_session else {"process_group": 0}


def _process_rows() -> list[tuple[int, int, int, str]]:
    """Linux 直接读取 proc，避免容器必须安装 ps；macOS 使用系统自带的进程查询。"""
    if sys.platform == "linux":
        rows = []
        with os.scandir("/proc") as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                try:
                    # comm 可含空格和括号，必须从最后一个右括号之后读取 state/ppid/pgrp。
                    fields = (Path(entry.path) / "stat").read_text().rsplit(")", 1)[1].split()
                    rows.append((int(entry.name), int(fields[1]), int(fields[2]), fields[0]))
                except (OSError, ValueError, IndexError):
                    continue  # 读取过程中退出的进程无需再等待。
        return rows
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat="],
        capture_output=True,
        text=True,
        timeout=0.5,
        check=True,
    )
    return [
        (int(parts[0]), int(parts[1]), int(parts[2]), parts[3])
        for line in result.stdout.splitlines()
        if len(parts := line.split()) == 4
    ]


def _group_running(pid: int) -> bool:
    """仅将尚在运行的组成员视为存活，已退出的孤儿僵尸不占用模型资源。"""
    return any(group == pid and not state.startswith("Z") for _, _, group, state in _process_rows())


def _descendant_groups(pid: int) -> set[int]:
    """同时按父子关系和所属会话收集进程组，覆盖 Router 先退出后被重新托管的 worker。"""
    records = _process_rows()
    descendants = {pid}
    for child, _, group, _ in records:
        try:
            # 会话由 WebUI 启动 Router 时创建，其身份不会因 Router 退出和子进程重挂而改变。
            # worker 只拥有自己的进程组，不能通过共享会话反向终止 Router 或其它 worker。
            if group == pid or os.getsid(child) == pid:
                descendants.add(child)
        except (ProcessLookupError, PermissionError):
            continue
    while True:
        children = {child for child, parent, _, _ in records if parent in descendants}
        if children.issubset(descendants):
            break
        descendants.update(children)
    return {group for child, _, group, _ in records if child in descendants and group != os.getpgrp()} | {pid}


def stop_process_tree(
    process: subprocess.Popen,
    control: ManagedProcessControl | None,
    *,
    timeout_seconds: float = 10,
    grouped: bool = False,
    process_job: WindowsProcessJob | None = None,
) -> None:
    """统一关闭预算；仍有活进程则抛错，调用者必须保留引用且禁止启动新实例。"""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    groups = _descendant_groups(process.pid) if grouped and os.name != "nt" else set()
    if process.poll() is None and control is not None:
        sent = control.request_shutdown(min(1.0, max(0.0, deadline - time.monotonic())))
        if sent:
            try:
                process.wait(timeout=min(timeout_seconds * 0.4, max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
    if process_job is not None:
        process_job.stop(deadline)
    elif grouped and os.name != "nt":
        for sig, fraction in ((signal.SIGTERM, 0.3), (signal.SIGKILL, 1.0)):
            for group in groups:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if _group_running(group):
                        raise
            phase_deadline = min(deadline, time.monotonic() + timeout_seconds * fraction)
            while time.monotonic() < phase_deadline:
                process.poll()
                if not any(_group_running(group) for group in groups):
                    break
                time.sleep(min(0.05, max(0.0, phase_deadline - time.monotonic())))
            if not any(_group_running(group) for group in groups):
                break
        if any(_group_running(group) for group in groups):
            raise RuntimeError(f"Managed process group {process.pid} did not stop")
    elif process.poll() is None:
        if grouped and os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.01, deadline - time.monotonic()),
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=min(timeout_seconds * 0.3, max(0.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Managed process {process.pid} did not stop") from exc


__all__ = ["process_group_options", "stop_process_tree"]
