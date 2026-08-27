# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF parser 与 renderer 共享的跨平台字体解析和度量。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import platform

from PIL import ImageFont

from .models import Font


def _font_search_roots() -> tuple[Path, ...]:
    """返回当前平台常见字体目录，不在模块导入时访问文件系统。"""
    system = platform.system()
    roots: list[Path] = []
    if system == "Darwin":
        roots.extend((Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"))
    elif system == "Windows":
        import os

        windows_dir = Path(os.environ.get("WINDIR", "C:/Windows"))
        roots.append(windows_dir / "Fonts")
    else:
        roots.extend(
            (
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".fonts",
                Path.home() / ".local/share/fonts",
            )
        )
    return tuple(root for root in roots if root.is_dir())


@lru_cache(maxsize=1)
def _font_file_index() -> dict[str, str]:
    """惰性建立字体文件名索引，避免 import 时扫描系统目录。"""
    index: dict[str, str] = {}
    for root in _font_search_roots():
        try:
            candidates = root.rglob("*")
            for candidate in candidates:
                if candidate.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                    continue
                index.setdefault(candidate.stem.casefold().replace(" ", ""), str(candidate))
        except OSError:
            continue
    return index


def _font_aliases(face_name: str, charset: int) -> tuple[str, ...]:
    """返回 Windows 字体名在 Linux/macOS 上的固定替代顺序。"""
    normalized = face_name.casefold().replace(" ", "")
    alias_map = {
        "arial": ("Arial", "LiberationSans-Regular", "DejaVuSans"),
        "calibri": ("Calibri", "Carlito-Regular", "Arial", "DejaVuSans"),
        "cambria": ("Cambria", "Caladea-Regular", "DejaVuSerif"),
        "timesnewroman": ("Times New Roman", "LiberationSerif-Regular", "DejaVuSerif"),
        "couriernew": ("Courier New", "LiberationMono-Regular", "DejaVuSansMono"),
        "simsun": ("SimSun", "Songti SC", "NotoSerifCJKsc-Regular", "DejaVuSans"),
        "microsoftyahei": ("Microsoft YaHei", "PingFang SC", "NotoSansCJKsc-Regular", "DejaVuSans"),
    }
    aliases = alias_map.get(normalized, (face_name,))
    if charset in {128, 129, 134, 136}:
        aliases = (*aliases, "PingFang SC", "NotoSansCJKsc-Regular", "NotoSansCJK-Regular", "DejaVuSans")
    return tuple(dict.fromkeys((*aliases, "DejaVuSans")))


@lru_cache(maxsize=256)
def load_font(
    face_name: str,
    size: int,
    weight: int,
    italic: bool,
    charset: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按字体名、别名和文件索引加载 Pillow 字体。"""
    normalized_size = max(1, min(size, 4096))
    index = _font_file_index()
    for alias in _font_aliases(face_name, charset):
        keys = [alias.casefold().replace(" ", "")]
        if weight >= 700:
            keys.insert(0, f"{keys[0]}bold")
        if italic:
            keys.insert(0, f"{keys[0]}italic")
        candidates = [alias, f"{alias}.ttf", *(index[key] for key in keys if key in index)]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, normalized_size)
            except OSError:
                continue
    return ImageFont.load_default(size=normalized_size)


def measure_text_advance(font: Font, text: str) -> float:
    """用 renderer 同款字体回退规则估算无显式 spacing 的逻辑 advance。"""
    if not text:
        return 0.0
    font_size = max(1, round(abs(font.height or -12.0)))
    loaded = load_font(font.face_name, font_size, font.weight, font.italic, font.charset)
    advance = float(loaded.getlength(text))
    if font.width:
        natural_width = max(float(loaded.getlength("0")), 1e-9)
        advance *= abs(font.width) / natural_width
    return max(advance, 0.0)


__all__ = ["load_font", "measure_text_advance"]
