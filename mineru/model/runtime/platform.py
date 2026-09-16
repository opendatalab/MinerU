# Copyright (c) Opendatalab. All rights reserved.
"""推理运行环境的平台事实与版本门槛。"""

import platform

from packaging import version


def is_windows_environment() -> bool:
    """判断当前操作系统是否为 Windows。"""
    return platform.system() == "Windows"


def is_mac_environment() -> bool:
    """判断当前操作系统是否为 macOS。"""
    return platform.system() == "Darwin"


def is_linux_environment() -> bool:
    """判断当前操作系统是否为 Linux。"""
    return platform.system() == "Linux"


def is_apple_silicon_cpu() -> bool:
    """判断 CPU 是否使用 Apple Silicon 兼容架构。"""
    return platform.machine() in ["arm64", "aarch64"]


def is_mac_os_version_supported(min_version: str = "13.5") -> bool:
    """按调用方要求检查 macOS 版本与 CPU 架构。"""
    if not is_mac_environment() or not is_apple_silicon_cpu():
        return False
    mac_version = platform.mac_ver()[0]
    if not mac_version:
        return False
    # print("Mac OS Version:", mac_version)
    return version.parse(mac_version) >= version.parse(min_version)


__all__ = [
    "is_windows_environment",
    "is_mac_environment",
    "is_linux_environment",
    "is_apple_silicon_cpu",
    "is_mac_os_version_supported",
]
