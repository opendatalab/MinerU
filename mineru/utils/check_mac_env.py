# Copyright (c) Opendatalab. All rights reserved.
import platform

from packaging import version


# Detect if the current environment is a Mac computer
def is_mac_environment() -> bool:
    return platform.system() == "Darwin"


# 检测cpu是否为Apple Silicon架构
def is_apple_silicon_cpu() -> bool:
    return platform.machine() in ["arm64", "aarch64"]


#如果是Mac电脑且为Apple Silicon架构，检测macos版本是否在13.5以上
def is_mac_os_version_supported(min_version: str = "13.5") -> bool:
    if not is_mac_environment() or not is_apple_silicon_cpu():
        return False
    mac_version = platform.mac_ver()[0]
    # print("Mac OS Version:", mac_version)
    return version.parse(mac_version) >= version.parse(min_version)

if __name__ == '__main__':
    print("Is Mac Environment:", is_mac_environment())
    print("Is Apple Silicon CPU:", is_apple_silicon_cpu())
    print("Is Mac OS Version Supported (>=13.5):", is_mac_os_version_supported())