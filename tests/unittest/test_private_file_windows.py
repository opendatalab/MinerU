"""在真实 Windows 文件系统上验证 OWNER RIGHTS 目录和凭证文件 DACL。"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

import pytest

from mineru.utils.private_file import create_private_tempfile, protect_private_file

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires Windows DACL support")


def _set_owner_rights_directory(path: Path, extra_ace: str = "") -> None:
    """显式构造 Python mkdir(mode=0o700) 的 DACL，兼容不同 Python 版本。"""
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    security.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    security.SetFileSecurityW.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    sddl = "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)" + extra_ace
    if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not security.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def _read_dacl(path: Path) -> str:
    """通过独立的 Windows SDDL 转换读回磁盘权限。"""
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    security.GetFileSecurityW.restype = wintypes.BOOL
    security.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.c_void_p,
    ]
    security.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    size = wintypes.DWORD()
    security.GetFileSecurityW(str(path), 4, None, 0, ctypes.byref(size))
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(size.value)
    if not security.GetFileSecurityW(str(path), 4, descriptor, size, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    sddl = wintypes.LPWSTR()
    if not security.ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, 1, 4, ctypes.byref(sddl), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return sddl.value
    finally:
        kernel.LocalFree(ctypes.cast(sddl, ctypes.c_void_p))


@pytest.mark.parametrize("create", [False, True], ids=["existing-file", "new-tempfile"])
def test_owner_rights_directory_allows_private_files(tmp_path: Path, create: bool) -> None:
    """安全目录使用 OW 时应成功，文件仍使用显式用户 SID 且不继承目录权限。"""
    _set_owner_rights_directory(tmp_path)
    parent_dacl = _read_dacl(tmp_path)
    assert "(A;OICI;FA;;;OW)" in parent_dacl
    if create:
        fd, filename = create_private_tempfile(str(tmp_path))
        path = Path(filename)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            assert path.read_bytes() == b""
            protected_dacl = _read_dacl(path)
            stream.write("private test content")
    else:
        path = tmp_path / "config.yaml"
        path.write_text("private test content", encoding="utf-8")
        protect_private_file(path)
        protected_dacl = _read_dacl(path)
    assert path.read_text(encoding="utf-8") == "private test content"
    assert protected_dacl.startswith("D:P(A;;FA;;;S-1-")
    assert protected_dacl.endswith("(A;;FA;;;SY)")
    assert protected_dacl.count("(") == 2
    protect_private_file(path)
    assert _read_dacl(path) == protected_dacl
    assert _read_dacl(tmp_path) == parent_dacl


@pytest.mark.parametrize("create", [False, True], ids=["existing-file", "new-tempfile"])
@pytest.mark.parametrize("trustee", ["WD", "AU", "BU"], ids=["everyone", "authenticated-users", "users"])
def test_owner_rights_does_not_allow_other_directory_writers(tmp_path: Path, create: bool, trustee: str) -> None:
    """即使存在 OW，其他普通账户可写的父目录仍必须拒绝。"""
    _set_owner_rights_directory(tmp_path, f"(A;OICI;FA;;;{trustee})")
    path = tmp_path / "config.yaml"
    path.write_text("private test content", encoding="utf-8")
    original_dacl = _read_dacl(path)
    with pytest.raises(PermissionError, match="directory must not be writable by other users"):
        if create:
            fd, filename = create_private_tempfile(str(tmp_path))
            os.close(fd)
            Path(filename).unlink()
        else:
            protect_private_file(path)
    assert _read_dacl(path) == original_dacl
    assert path.read_text(encoding="utf-8") == "private test content"
    assert not list(tmp_path.glob(".config.*.tmp"))
