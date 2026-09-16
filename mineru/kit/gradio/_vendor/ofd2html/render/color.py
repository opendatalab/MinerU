"""Color helpers: OFD ``CT_Color`` ``Value`` attribute -> CSS color string.

OFD colors are space-separated channel values, with channel space implied by
the parent ``ColorSpace``. For the HTML preview we approximate everything to
sRGB:

* RGB triples: pass through.
* CMYK quadruples: simple naive conversion (sufficient for preview).
* Single-channel gray: replicate to RGB.

Channel values may be ints (0-255) or normalized floats (0.0-1.0).
"""

from __future__ import annotations

from typing import Optional


# 解析 OFD 颜色值。
def parse_color(raw: Optional[str], default: str = "#000000") -> str:
    if not raw:
        return default
    parts = raw.replace(",", " ").split()
    if not parts:
        return default
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return default
    nums = [_to_byte(n) for n in nums]
    if len(nums) == 1:
        v = nums[0]
        return _hex(v, v, v)
    if len(nums) == 3:
        return _hex(nums[0], nums[1], nums[2])
    if len(nums) >= 4:
        # CMYK -> sRGB approximation.
        c, m, y, k = nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0, nums[3] / 255.0
        r = round(255 * (1 - c) * (1 - k))
        g = round(255 * (1 - m) * (1 - k))
        b = round(255 * (1 - y) * (1 - k))
        return _hex(r, g, b)
    return default


# 将颜色分量限制到字节范围。
def _to_byte(n: float) -> int:
    # Heuristic: <=1 means normalized float; otherwise treat as 0-255 byte.
    if 0 <= n <= 1:
        n = n * 255
    return max(0, min(255, int(round(n))))


# 生成十六进制颜色。
def _hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"
