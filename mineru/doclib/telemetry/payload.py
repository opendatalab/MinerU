"""Telemetry payload and environment context builders."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ...version import __version__
from .constants import TELEMETRY_API_VERSION, TELEMETRY_SCHEMA_VERSION


def canonical_dimensions(dimensions: dict[str, str]) -> dict[str, str]:
    return {key: dimensions[key] for key in sorted(dimensions)}


def dimensions_hash(dimensions: dict[str, str]) -> str:
    raw = json.dumps(canonical_dimensions(dimensions), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def utc_iso_from_ms(value_ms: int) -> str:
    return datetime.fromtimestamp(value_ms / 1000, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def compact_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def build_period_payload(
    *,
    batch_id: str,
    installation_id: str,
    period_start: int,
    period_end: int,
    context: dict[str, str],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "api_version": TELEMETRY_API_VERSION,
        "batch_id": batch_id,
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "installation_id": installation_id,
        "period_start": utc_iso_from_ms(period_start),
        "period_end": utc_iso_from_ms(period_end),
        "context": context,
        "metrics": metrics,
    }


def collect_environment_context() -> dict[str, str]:
    return {
        "app": "mineru",
        "app_version": __version__,
        "os": _os_name(),
        "arch": platform.machine().lower() or "unknown",
        "python": _python_version(),
        "install_channel": _install_channel(),
        "cpu_count_bucket": _cpu_count_bucket(os.cpu_count()),
        "gpu_vendor": _gpu_vendor(),
    }


def _os_name() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _cpu_count_bucket(cpu_count: int | None) -> str:
    if cpu_count is None or cpu_count <= 0:
        return "unknown"
    if cpu_count <= 2:
        return "1_2"
    if cpu_count <= 4:
        return "3_4"
    if cpu_count <= 8:
        return "5_8"
    if cpu_count <= 16:
        return "9_16"
    return "gt_16"


def _install_channel() -> str:
    if _running_in_docker():
        return "docker"
    if _looks_like_source_checkout():
        return "source"

    installer = _metadata_value("INSTALLER")
    if installer in {"uv", "pip"}:
        return installer

    direct_url = _metadata_value("direct_url.json")
    if '"editable":true' in direct_url.replace(" ", ""):
        return "source"
    return "unknown"


def _running_in_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "docker" in cgroup or "kubepods" in cgroup or "containerd" in cgroup


def _looks_like_source_checkout() -> bool:
    root = Path(__file__).resolve().parents[3]
    return (root / "pyproject.toml").is_file() and (root / ".git").exists()


def _metadata_value(name: str) -> str:
    for distribution in ("mineru", "mineru-vl-utils"):
        try:
            text = metadata.distribution(distribution).read_text(name)
        except metadata.PackageNotFoundError:
            continue
        if text:
            return text.strip().lower()
    return ""


def _gpu_vendor() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "apple"
    if system == "Linux":
        return _linux_gpu_vendor()
    if system == "Windows":
        return _windows_gpu_vendor()
    return "unknown"


def _linux_gpu_vendor() -> str:
    vendor_ids: list[str] = []
    for vendor_path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            vendor_ids.append(vendor_path.read_text(encoding="utf-8", errors="ignore").strip().lower())
        except OSError:
            continue
    if not vendor_ids:
        for vendor_path in Path("/sys/bus/pci/devices").glob("*/vendor"):
            try:
                vendor_ids.append(vendor_path.read_text(encoding="utf-8", errors="ignore").strip().lower())
            except OSError:
                continue
    if "0x10de" in vendor_ids:
        return "nvidia"
    if "0x1002" in vendor_ids:
        return "amd"
    if "0x8086" in vendor_ids:
        return "intel"
    return "unknown"


def _windows_gpu_vendor() -> str:
    try:
        import winreg
    except Exception:
        return "unknown"

    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
    except OSError:
        return "unknown"

    with root:
        index = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                subkey = winreg.OpenKey(root, subkey_name)
            except OSError:
                continue
            with subkey:
                values = []
                for value_name in ("DriverDesc", "ProviderName", "MatchingDeviceId"):
                    try:
                        values.append(str(winreg.QueryValueEx(subkey, value_name)[0]).lower())
                    except OSError:
                        continue
                joined = " ".join(values)
                if "nvidia" in joined or "ven_10de" in joined:
                    return "nvidia"
                if "amd" in joined or "radeon" in joined or "ven_1002" in joined:
                    return "amd"
                if "intel" in joined or "ven_8086" in joined:
                    return "intel"
    return "unknown"


__all__ = [
    "build_period_payload",
    "canonical_dimensions",
    "collect_environment_context",
    "compact_json_bytes",
    "dimensions_hash",
    "utc_iso_from_ms",
]
