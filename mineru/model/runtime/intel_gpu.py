# Copyright (c) Opendatalab. All rights reserved.
"""Explicit PCI architecture families for the Linux WebGPU product policy."""

from typing import Literal

IntelArchitecture = Literal["legacy", "xe_lp", "xe_hpg", "xe_lpg", "xe2_lpg", "xe2_hpg", "xe3_lpg", "xe3_lpg_xs"]

INTEL_VENDOR_ID = 0x8086

# PCI device IDs are not chronologically ordered. Match the high byte exactly,
# as in https://github.com/google/dawn/blob/main/src/dawn/gpu_info.json (2026-10-21).
# Older Gen6/7/8/9/11 families share the same CPU policy, without conflating
# their actual hardware generations. 0x49 includes the Xe-LP DG1 discrete GPU.
INTEL_PCI_ARCHITECTURES: dict[int, IntelArchitecture] = {
    0x0100: "legacy",
    0x0400: "legacy",
    0x0A00: "legacy",
    0x0C00: "legacy",
    0x0D00: "legacy",
    0x0F00: "legacy",
    0x1600: "legacy",
    0x2200: "legacy",
    0x1900: "legacy",
    0x3100: "legacy",
    0x3E00: "legacy",
    0x5A00: "legacy",
    0x5900: "legacy",
    0x8700: "legacy",
    0x9B00: "legacy",
    0x8A00: "legacy",
    0x4500: "legacy",
    0x4E00: "legacy",
    0x9800: "legacy",
    0x4600: "xe_lp",
    0x4C00: "xe_lp",
    0x4900: "xe_lp",
    0x9A00: "xe_lp",
    0xA700: "xe_lp",
    0x4F00: "xe_hpg",
    0x5600: "xe_hpg",
    0x7D00: "xe_lpg",
    0xB600: "xe_lpg",
    0x6400: "xe2_lpg",
    0xE200: "xe2_hpg",
    0xB000: "xe3_lpg",
    0xFD00: "xe3_lpg_xs",
}

WEBGPU_INTEL_ARCHITECTURES: frozenset[IntelArchitecture] = frozenset(
    {"xe_hpg", "xe_lpg", "xe2_lpg", "xe2_hpg", "xe3_lpg", "xe3_lpg_xs"}
)


def intel_architecture(device_id: int) -> IntelArchitecture | None:
    """Classify a 16-bit Intel GPU PCI ID; unknown families remain unknown."""
    return INTEL_PCI_ARCHITECTURES.get(device_id & 0xFF00) if 0 <= device_id <= 0xFFFF else None


def supports_webgpu_architecture(vendor_id: int, device_id: int) -> bool:
    """Restrict Intel only. This product policy does not claim hardware incompatibility."""
    return vendor_id != INTEL_VENDOR_ID or intel_architecture(device_id) in WEBGPU_INTEL_ARCHITECTURES


__all__ = [
    "INTEL_PCI_ARCHITECTURES",
    "INTEL_VENDOR_ID",
    "IntelArchitecture",
    "WEBGPU_INTEL_ARCHITECTURES",
    "intel_architecture",
    "supports_webgpu_architecture",
]
