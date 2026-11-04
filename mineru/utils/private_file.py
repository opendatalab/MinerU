"""为本机凭证配置提供 POSIX 权限、macOS ACL 和 Windows DACL 保护。"""

from __future__ import annotations

import os
import secrets
import stat
import sys
import tempfile
from pathlib import Path


def protect_private_file(path: str | Path) -> None:
    """限制普通账户读取凭证文件；权限无法落实时失败，不降级为公开文件。"""
    path = Path(path).absolute()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("Credential configuration must be a regular file without links.")
    if os.name == "nt":
        _protect_windows_file(path)
        return
    if sys.platform == "darwin":
        from ._private_file_macos import protect_macos_file

        protect_macos_file(path)
        return
    if info.st_uid != os.getuid():
        raise PermissionError("Credential configuration must be owned by the current user.")
    parent = path.parent.stat()
    if parent.st_uid not in {os.getuid(), 0}:
        raise PermissionError("Credential configuration directory must have a trusted owner.")
    if parent.st_mode & 0o022:
        raise PermissionError("Credential configuration directory must not be writable by other users.")
    os.chmod(path, 0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("The filesystem cannot enforce private credential configuration permissions.")


def create_private_tempfile(directory: str) -> tuple[int, str]:
    """从创建时就限制凭证临时文件权限，防止先打开的其他账户句柄读取后写入的正文。"""
    if os.name != "nt" and sys.platform != "darwin":
        fd, path = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=directory)
        try:
            protect_private_file(path)
        except BaseException:
            os.close(fd)
            os.unlink(path)
            raise
        return fd, path
    for _ in range(10):
        path = Path(directory).absolute() / f".config.{secrets.token_hex(12)}.tmp"
        try:
            if sys.platform == "darwin":
                from ._private_file_macos import protect_macos_file

                fd = protect_macos_file(path, create=True)
            else:
                fd = _protect_windows_file(path, create=True)
        except FileExistsError:
            continue
        assert fd is not None
        return fd, str(path)
    raise FileExistsError("Could not allocate a private configuration file.")


def _protect_windows_file(path: Path, *, create: bool = False) -> int | None:
    """按当前账户和 SYSTEM 的受保护 DACL 创建或保护文件，并读回验证实际权限。"""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    pointer = ctypes.c_void_p
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer

    class SecurityAttributes(ctypes.Structure):
        """在 CreateFile 时传入 DACL，避免先创建公开文件再收紧权限的窗口。"""

        _fields_ = [("length", wintypes.DWORD), ("descriptor", pointer), ("inherit", wintypes.BOOL)]

    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    security.OpenProcessToken.restype = wintypes.BOOL
    security.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        pointer,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    security.GetTokenInformation.restype = wintypes.BOOL
    security.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(wintypes.LPWSTR)]
    security.ConvertSidToStringSidW.restype = wintypes.BOOL
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(pointer),
        ctypes.POINTER(wintypes.DWORD),
    ]
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    security.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, pointer]
    security.SetFileSecurityW.restype = wintypes.BOOL
    security.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        pointer,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    security.GetFileSecurityW.restype = wintypes.BOOL
    security.GetSecurityDescriptorDacl.argtypes = [
        pointer,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(pointer),
        ctypes.POINTER(wintypes.BOOL),
    ]
    security.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    security.GetSecurityDescriptorOwner.argtypes = [pointer, ctypes.POINTER(pointer), ctypes.POINTER(wintypes.BOOL)]
    security.GetSecurityDescriptorOwner.restype = wintypes.BOOL
    security.GetSecurityDescriptorControl.argtypes = [pointer, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD)]
    security.GetSecurityDescriptorControl.restype = wintypes.BOOL

    def read_descriptor(target: Path) -> ctypes.Array:
        """读取文件或目录的所有者和 DACL；不支持安全描述符的文件系统直接失败。"""
        size = wintypes.DWORD()
        security.GetFileSecurityW(str(target), 0x00000005, None, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        result = ctypes.create_string_buffer(size.value)
        if not security.GetFileSecurityW(str(target), 0x00000005, result, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def dacl_pointer(sd: object) -> int:
        """取得非空 DACL，避免将空 DACL 指针对应的全员访问误认为私密权限。"""
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), pointer()
        if not security.GetSecurityDescriptorDacl(sd, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not present.value or not dacl.value:
            raise PermissionError("Credential configuration requires an explicit DACL.")
        return dacl.value

    def sid_string(sid: int) -> str:
        """把系统 SID 转成稳定标识，并释放 Windows 分配的字符串。"""
        result = wintypes.LPWSTR()
        if not security.ConvertSidToStringSidW(sid, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return result.value
        finally:
            kernel.LocalFree(ctypes.cast(result, pointer))

    process_token = wintypes.HANDLE()
    sid_text = wintypes.LPWSTR()
    descriptor = pointer()
    created_handle = None
    created = False
    transferred = False
    try:
        # TOKEN_QUERY=0x0008，TokenUser=1；令牌用户的 SID 不依赖本地化用户名。
        if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(process_token)):
            raise ctypes.WinError(ctypes.get_last_error())
        size = wintypes.DWORD()
        security.GetTokenInformation(process_token, 1, None, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        user = ctypes.create_string_buffer(size.value)
        if not security.GetTokenInformation(process_token, 1, user, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        sid = ctypes.cast(user, ctypes.POINTER(pointer)).contents.value
        if not security.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        # 父目录的所有者或写权限也不能让其他普通账户替换凭证文件。
        parent = read_descriptor(path.parent)
        trusted = {sid_text.value, "S-1-5-18", "S-1-5-32-544"}
        owner, defaulted = pointer(), wintypes.BOOL()
        if not security.GetSecurityDescriptorOwner(parent, ctypes.byref(owner), ctypes.byref(defaulted)):
            raise ctypes.WinError(ctypes.get_last_error())
        if sid_string(owner.value) not in trusted:
            raise PermissionError("Credential configuration directory must have a trusted owner.")
        if not create:
            existing = read_descriptor(path)
            if not security.GetSecurityDescriptorOwner(existing, ctypes.byref(owner), ctypes.byref(defaulted)):
                raise ctypes.WinError(ctypes.get_last_error())
            if sid_string(owner.value) not in trusted:
                raise PermissionError("Credential configuration must have a trusted owner.")
        # OWNER RIGHTS 代表已验证可信的目录所有者；Python 的 mkdir(mode=0o700) 会使用该 SID。
        # 仅将它用于目录 ACE 校验，不把它本身当作可信的文件或目录所有者。
        trusted_writers = trusted | {"S-1-3-4"}
        parent_dacl = dacl_pointer(parent)
        count = ctypes.c_ushort.from_address(parent_dacl + 4).value
        ace = parent_dacl + 8
        for _ in range(count):
            ace_type = ctypes.c_ubyte.from_address(ace).value
            flags = ctypes.c_ubyte.from_address(ace + 1).value
            size = ctypes.c_ushort.from_address(ace + 2).value
            # 忽略仅继承的 ACE 和拒绝条目；未知的允许条目按不安全处理。
            if not flags & 0x08 and ace_type != 1:
                mask = ctypes.c_uint32.from_address(ace + 4).value
                if mask & 0x500D0156 and (ace_type != 0 or sid_string(ace + 8) not in trusted_writers):
                    raise PermissionError("Credential configuration directory must not be writable by other users.")
            ace += size
        expected = f"D:P(A;;FA;;;{sid_text.value})(A;;FA;;;SY)"
        if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(expected, 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if create:
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
            # GENERIC_READ | GENERIC_WRITE，CREATE_NEW，FILE_ATTRIBUTE_NORMAL；句柄不向子进程继承。
            created_handle = kernel.CreateFileW(str(path), 0xC0000000, 0x00000007, ctypes.byref(attributes), 1, 0x80, None)
            if created_handle == pointer(-1).value:
                created_handle = None
                raise ctypes.WinError(ctypes.get_last_error())
            created = True
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION 禁止重新继承公开权限。
        elif not security.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = read_descriptor(path)
        control, revision = wintypes.WORD(), wintypes.DWORD()
        if not security.GetSecurityDescriptorControl(actual, ctypes.byref(control), ctypes.byref(revision)):
            raise ctypes.WinError(ctypes.get_last_error())
        expected_dacl, actual_dacl = dacl_pointer(descriptor), dacl_pointer(actual)
        expected_size = ctypes.c_ushort.from_address(expected_dacl + 2).value
        actual_size = ctypes.c_ushort.from_address(actual_dacl + 2).value
        if not control.value & 0x1000 or ctypes.string_at(actual_dacl, actual_size) != ctypes.string_at(
            expected_dacl, expected_size
        ):
            raise PermissionError("The filesystem cannot enforce private credential configuration permissions.")
        if create:
            import msvcrt

            fd = msvcrt.open_osfhandle(created_handle, os.O_RDWR | os.O_BINARY)
            transferred = True
            return fd
        return None
    finally:
        if created_handle is not None and not transferred:
            kernel.CloseHandle(created_handle)
        if created and not transferred:
            path.unlink(missing_ok=True)
        if descriptor:
            kernel.LocalFree(descriptor)
        if sid_text:
            kernel.LocalFree(ctypes.cast(sid_text, pointer))
        if process_token:
            kernel.CloseHandle(process_token)


__all__ = ["create_private_tempfile", "protect_private_file"]
