"""在真实 macOS 文件系统上验证 BSD 权限之外的扩展 ACL 和创建时保护。"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mineru import config as configuration
from mineru.config import load_doclib_server_config, update_config_file
from mineru.utils import _private_file_macos as macos
from mineru.utils.private_file import create_private_tempfile, protect_private_file

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Requires real macOS ACL support")


def _add_acl(path: Path, entry: str) -> None:
    """用系统 chmod 构造真实 ACL，独立于被测的 ctypes 实现。"""
    subprocess.run(["/bin/chmod", "+a", entry, str(path)], check=True, capture_output=True)


def _acl_entries(path: Path) -> list[str]:
    """通过系统 ls 读取真实 ACL，避免验证复用生产代码导致同源误判。"""
    result = subprocess.run(["/bin/ls", "-lde", str(path)], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()[1:]


def test_existing_public_acl_is_removed_and_protection_is_idempotent(tmp_path: Path) -> None:
    """0600 不能屏蔽 Everyone ACL；保护后必须清除额外授权并保留正文。"""
    path = tmp_path / "配置.yaml"
    path.write_text("fake credential", encoding="utf-8")
    path.chmod(0o600)
    _add_acl(path, "everyone allow read,write,append,delete,writesecurity,chown")
    assert _acl_entries(path)
    for _ in range(2):
        protect_private_file(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert _acl_entries(path) == []
        assert path.read_text(encoding="utf-8") == "fake credential"


@pytest.mark.parametrize("create", [False, True])
@pytest.mark.parametrize(
    "permission",
    ["add_file", "add_subdirectory", "delete_child", "delete", "writeattr", "writeextattr", "writesecurity", "chown"],
)
def test_unsafe_directory_acl_is_rejected(tmp_path: Path, permission: str, create: bool) -> None:
    """即使目录为 0700，额外授予其他账户的写入或接管权限也必须拒绝。"""
    path = tmp_path / "config.yaml"
    path.write_text("unchanged", encoding="utf-8")
    tmp_path.chmod(0o700)
    _add_acl(tmp_path, f"everyone allow {permission}")
    before = _acl_entries(tmp_path)
    with pytest.raises(PermissionError, match="directory must not be writable"):
        if create:
            create_private_tempfile(str(tmp_path))
        else:
            protect_private_file(path)
    assert _acl_entries(tmp_path) == before
    assert path.read_text(encoding="utf-8") == "unchanged"
    assert not list(tmp_path.glob(".config.*.tmp"))


@pytest.mark.parametrize(
    "entry",
    [
        "everyone allow list,search,readattr,readextattr,readsecurity",
        "everyone deny delete",
        "everyone allow read,write,file_inherit,only_inherit",
        "user:root allow add_file,delete_child,writesecurity",
        "current-user",
    ],
)
def test_safe_directory_acls_are_preserved(tmp_path: Path, entry: str) -> None:
    """只读、拒绝、仅继承和可信用户权限不能像 Windows 的 OWNER RIGHTS 一样被误拒。"""
    if entry == "current-user":
        import pwd

        entry = f"user:{pwd.getpwuid(os.getuid()).pw_name} allow add_file,delete_child,writesecurity"
    _add_acl(tmp_path, entry)
    before = _acl_entries(tmp_path)
    fd, filename = create_private_tempfile(str(tmp_path))
    try:
        path = Path(filename)
        assert not os.get_inheritable(fd)
        assert _acl_entries(path) == []
        os.write(fd, b"private content")
        protect_private_file(path)
        assert path.read_bytes() == b"private content"
        assert _acl_entries(path) == []
        assert _acl_entries(tmp_path) == before
    finally:
        os.close(fd)


def test_tempfile_has_no_inherited_acl_at_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """在原生创建调用刚返回时检查 ACL，防止先公开创建再清理的句柄抢占窗口。"""
    _add_acl(tmp_path, "everyone allow read,write,file_inherit,only_inherit")
    library = macos._load_security_library()
    native_open = library.openx_np
    observed: list[list[str]] = []

    def observed_open(path: bytes, flags: int, descriptor: int) -> int:
        """记录 openx_np 返回瞬间的实际 ACL，而不是只检查函数最终结果。"""
        fd = native_open(path, flags, descriptor)
        if fd >= 0:
            observed.append(_acl_entries(Path(os.fsdecode(path))))
        return fd

    monkeypatch.setattr(library, "openx_np", observed_open)
    monkeypatch.setattr(macos, "_load_security_library", lambda: library)
    fd, filename = create_private_tempfile(str(tmp_path))
    try:
        assert observed == [[]]
        assert not os.get_inheritable(fd)
        assert Path(filename).read_bytes() == b""
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
    finally:
        os.close(fd)


def test_first_start_restart_and_atomic_update_keep_private_acl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """新 home 继承只读 ACL 后仍能启动，重启和原子更新始终保持同一私密凭证。"""
    _add_acl(tmp_path, "everyone allow read,file_inherit,directory_inherit,only_inherit")
    home = tmp_path / "新配置目录"
    monkeypatch.setenv("MINERU_HOME", str(home))
    monkeypatch.delenv("MINERU_CONFIG", raising=False)
    monkeypatch.delenv("MINERU_DOCLIB_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(configuration, "_loaded_config", None)
    first = load_doclib_server_config()
    token = first.config.doclib.auth.token.get_secret_value()
    path = home / "config.yaml"
    assert _acl_entries(home)
    assert _acl_entries(path) == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_doclib_server_config().config.doclib.auth.token.get_secret_value() == token
    update_config_file({"log": {"level": "debug"}})
    assert _acl_entries(path) == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["doclib"]["auth"]["token"] == token
    assert raw["log"]["level"] == "debug"
    assert not list(home.glob(".config.*.tmp"))


def test_strict_umask_does_not_prevent_private_tempfile_creation(tmp_path: Path) -> None:
    """严格 umask 只会减少创建时权限，返回前应补回所有者读写位。"""
    previous = os.umask(0o777)
    try:
        fd, filename = create_private_tempfile(str(tmp_path))
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
        assert _acl_entries(Path(filename)) == []
        os.write(fd, b"private content")
        assert Path(filename).read_bytes() == b"private content"
    finally:
        os.close(fd)


@pytest.mark.parametrize("failure", ["read_acl", "create", "set_acl", "verify"])
def test_native_failures_do_not_leave_credential_tempfiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """ACL 读取、原子创建或读回验证失败时直接报错，并清理已创建的文件和句柄。"""
    library = macos._load_security_library()
    native_get_acl, native_open, native_chmod = library.acl_get_fd_np, library.openx_np, library.fchmodx_np
    created_fds: list[int] = []

    def failing_acl(fd: int, acl_type: int) -> int | None:
        """模拟文件系统拒绝读取父目录 ACL 或刚创建文件的 ACL。"""
        if failure == "read_acl" or fd in created_fds:
            ctypes.set_errno(errno.EACCES)
            return None
        return native_get_acl(fd, acl_type)

    def tracked_open(path: bytes, flags: int, descriptor: int) -> int:
        """记录真实句柄或模拟不支持原子 ACL 创建的文件系统。"""
        if failure == "create":
            ctypes.set_errno(errno.ENOTSUP)
            return -1
        fd = native_open(path, flags, descriptor)
        if fd >= 0:
            created_fds.append(fd)
        return fd

    def failing_chmod(fd: int, descriptor: int) -> int:
        """模拟无法落实最终权限的文件系统，检查返回句柄前的失败清理。"""
        if failure == "set_acl":
            ctypes.set_errno(errno.EACCES)
            return -1
        return native_chmod(fd, descriptor)

    monkeypatch.setattr(library, "acl_get_fd_np", failing_acl)
    monkeypatch.setattr(library, "openx_np", tracked_open)
    monkeypatch.setattr(library, "fchmodx_np", failing_chmod)
    monkeypatch.setattr(macos, "_load_security_library", lambda: library)
    with pytest.raises(OSError):
        create_private_tempfile(str(tmp_path))
    assert not list(tmp_path.glob(".config.*.tmp"))
    for fd in created_fds:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)


@pytest.mark.parametrize("link", ["hard", "symbolic"])
def test_linked_credentials_are_rejected(tmp_path: Path, link: str) -> None:
    """ACL 保护不能绕过已有的符号链接和硬链接拒绝规则。"""
    original, path = tmp_path / "original", tmp_path / "config.yaml"
    original.write_text("unchanged", encoding="utf-8")
    if link == "hard":
        path.hardlink_to(original)
    else:
        path.symlink_to(original)
    with pytest.raises(PermissionError, match="regular file without links"):
        protect_private_file(path)
    assert original.read_text(encoding="utf-8") == "unchanged"
