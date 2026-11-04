"""使用 macOS 原生 ACL 接口保护凭证，避免仅检查 BSD 权限位遗漏额外授权。"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from pathlib import Path

_ACL_TYPE_EXTENDED = 0x100
_ACL_EXTENDED_ALLOW = 1
_ACL_EXTENDED_DENY = 2
_ACL_ENTRY_ONLY_INHERIT = 1 << 8
_ACL_FLAG_NO_INHERIT = 1 << 17
# 目录只允许列举、搜索、读取属性/扩展属性/安全信息及同步；其他或未知权限均按可写处理。
_READ_ONLY_PERMISSIONS = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9) | (1 << 11) | (1 << 20)


def _load_security_library() -> ctypes.CDLL:
    """按需加载系统 ACL API，并显式声明指针宽度，避免 arm64 上地址截断。"""
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    pointer = ctypes.c_void_p
    library.acl_init.argtypes = [ctypes.c_int]
    library.acl_init.restype = pointer
    library.acl_free.argtypes = [pointer]
    library.acl_free.restype = ctypes.c_int
    library.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    library.acl_get_fd_np.restype = pointer
    library.acl_get_entry.argtypes = [pointer, ctypes.c_int, ctypes.POINTER(pointer)]
    library.acl_get_entry.restype = ctypes.c_int
    library.acl_get_tag_type.argtypes = [pointer, ctypes.POINTER(ctypes.c_int)]
    library.acl_get_tag_type.restype = ctypes.c_int
    library.acl_get_permset_mask_np.argtypes = [pointer, ctypes.POINTER(ctypes.c_uint64)]
    library.acl_get_permset_mask_np.restype = ctypes.c_int
    library.acl_get_flagset_np.argtypes = [pointer, ctypes.POINTER(pointer)]
    library.acl_get_flagset_np.restype = ctypes.c_int
    library.acl_get_flag_np.argtypes = [pointer, ctypes.c_uint]
    library.acl_get_flag_np.restype = ctypes.c_int
    library.acl_add_flag_np.argtypes = [pointer, ctypes.c_uint]
    library.acl_add_flag_np.restype = ctypes.c_int
    library.acl_get_qualifier.argtypes = [pointer]
    library.acl_get_qualifier.restype = pointer
    library.mbr_uid_to_uuid.argtypes = [ctypes.c_uint, pointer]
    library.mbr_uid_to_uuid.restype = ctypes.c_int
    library.filesec_init.argtypes = []
    library.filesec_init.restype = pointer
    library.filesec_free.argtypes = [pointer]
    library.filesec_free.restype = None
    library.filesec_set_property.argtypes = [pointer, ctypes.c_int, pointer]
    library.filesec_set_property.restype = ctypes.c_int
    library.openx_np.argtypes = [ctypes.c_char_p, ctypes.c_int, pointer]
    library.openx_np.restype = ctypes.c_int
    library.fchmodx_np.argtypes = [ctypes.c_int, pointer]
    library.fchmodx_np.restype = ctypes.c_int
    return library


def _checked(result: int | None) -> int:
    """保留原生错误码，权限或文件系统能力不足时不得静默降级。"""
    if result is None or result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _check_acl(library: ctypes.CDLL, fd: int, *, directory: bool) -> None:
    """校验已打开对象的 ACL；目录拒绝不可信写权限，凭证文件必须没有扩展授权。"""
    acl = library.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED)
    # Darwin 对有效 fd 上不存在的扩展 ACL 返回 ENOENT；其他错误不能视为没有 ACL。
    if acl is None and ctypes.get_errno() == errno.ENOENT:
        return
    _checked(acl)
    try:
        entry = ctypes.c_void_p()
        entry_id = 0  # ACL_FIRST_ENTRY；Darwin 的 ACL_NEXT_ENTRY 为 -1。
        trusted: set[bytes] = set()
        while True:
            result = library.acl_get_entry(acl, entry_id, ctypes.byref(entry))
            # Darwin 用 -1/EINVAL 表示迭代结束，与 Linux 的返回约定不同。
            if result == -1 and ctypes.get_errno() == errno.EINVAL:
                break
            _checked(result)
            entry_id = -1
            if not directory:
                raise PermissionError("The filesystem cannot enforce private credential configuration ACLs.")
            tag, flags, permissions = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
            _checked(library.acl_get_tag_type(entry, ctypes.byref(tag)))
            _checked(library.acl_get_flagset_np(entry, ctypes.byref(flags)))
            inherit_only = _checked(library.acl_get_flag_np(flags, _ACL_ENTRY_ONLY_INHERIT))
            if tag.value == _ACL_EXTENDED_DENY or inherit_only:
                continue
            _checked(library.acl_get_permset_mask_np(entry, ctypes.byref(permissions)))
            if tag.value == _ACL_EXTENDED_ALLOW and not permissions.value & ~_READ_ONLY_PERMISSIONS:
                continue
            if not trusted:
                for uid in {os.getuid(), 0}:
                    identifier = ctypes.create_string_buffer(16)
                    code = library.mbr_uid_to_uuid(uid, identifier)
                    if code:
                        raise OSError(code, os.strerror(code))
                    trusted.add(identifier.raw)
            qualifier = _checked(library.acl_get_qualifier(entry))
            try:
                if tag.value != _ACL_EXTENDED_ALLOW or ctypes.string_at(qualifier, 16) not in trusted:
                    raise PermissionError("Credential configuration directory must not be writable by other users.")
            finally:
                library.acl_free(qualifier)
    finally:
        library.acl_free(acl)


def _check_file_owner(fd: int) -> None:
    """用实际文件描述符复核类型、硬链接数量和所有者，避免只信任路径预检。"""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("Credential configuration must be a regular file without links.")
    if info.st_uid != os.getuid():
        raise PermissionError("Credential configuration must be owned by the current user.")


def protect_macos_file(path: Path, *, create: bool = False) -> int | None:
    """原子设置 0600 和禁止继承的空 ACL；创建文件时从首个可见时刻即保持私密。"""
    library = _load_security_library()
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    acl, descriptor = None, None
    fd = None
    created = False
    transferred = False
    try:
        parent = os.fstat(parent_fd)
        if parent.st_uid not in {os.getuid(), 0}:
            raise PermissionError("Credential configuration directory must have a trusted owner.")
        if parent.st_mode & 0o022:
            raise PermissionError("Credential configuration directory must not be writable by other users.")
        _check_acl(library, parent_fd, directory=True)
        acl = _checked(library.acl_init(0))
        flags = ctypes.c_void_p()
        _checked(library.acl_get_flagset_np(acl, ctypes.byref(flags)))
        _checked(library.acl_add_flag_np(flags, _ACL_FLAG_NO_INHERIT))
        descriptor = _checked(library.filesec_init())
        mode, acl_pointer = ctypes.c_ushort(0o600), ctypes.c_void_p(acl)
        # FILESEC_MODE=4、FILESEC_ACL=5；属性值分别为 mode_t 和 acl_t 的地址。
        _checked(library.filesec_set_property(descriptor, 4, ctypes.byref(mode)))
        _checked(library.filesec_set_property(descriptor, 5, ctypes.byref(acl_pointer)))
        if create:
            fd = _checked(
                library.openx_np(
                    os.fsencode(path), os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, descriptor
                )
            )
            created = True
        else:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        _check_file_owner(fd)
        # 创建时已禁止 ACL 继承；此处也补回被严格 umask 去掉的所有者读写位。
        _checked(library.fchmodx_np(fd, descriptor))
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
            raise PermissionError("The filesystem cannot enforce private credential configuration permissions.")
        _check_acl(library, fd, directory=False)
        if create:
            transferred = True
            return fd
        return None
    finally:
        if fd is not None and not transferred:
            os.close(fd)
        if created and not transferred:
            path.unlink(missing_ok=True)
        if descriptor is not None:
            library.filesec_free(descriptor)
        if acl is not None:
            library.acl_free(acl)
        os.close(parent_fd)


__all__ = ["protect_macos_file"]
