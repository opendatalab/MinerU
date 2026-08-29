# Copyright (c) Opendatalab. All rights reserved.
"""OFD 资源文件索引与作用域合并。"""

from __future__ import annotations

from loguru import logger

from .constants import MAX_DRAW_PARAM_INHERITANCE
from .errors import OfdResourceLimitError
from .geometry import parse_numbers
from .models import CompositeResource, FontResource, MediaResource, ResourceRegistry
from .package import OfdPackage, element_text, first_child, local_name, parse_int


def _bool_attr(value: str | None) -> bool:
    """把 OFD 布尔属性解析为确定值。"""
    return (value or "").strip().casefold() in {"true", "1"}


def _resource_asset_part(package: OfdPackage, resource_part: str, base_loc: str, location: str) -> str | None:
    """把资源 BaseLoc 与资源内相对路径组合成包成员。"""
    base_part = package.resolve_reference(resource_part, base_loc) if base_loc else posix_parent(resource_part)
    if base_part is None:
        return None
    synthetic_base = f"{base_part.rstrip('/')}/_resource.xml" if base_loc else resource_part
    return package.resolve_reference(synthetic_base, location)


def posix_parent(part_name: str) -> str:
    """返回包成员的 POSIX 父目录。"""
    return part_name.rsplit("/", 1)[0] if "/" in part_name else ""


def parse_resource_part(package: OfdPackage, resource_part: str | None) -> ResourceRegistry:
    """解析单个 PublicRes、DocumentRes 或 PageRes。"""
    registry = ResourceRegistry()
    if resource_part is None:
        return registry
    root = package.xml_part(resource_part)
    if root is None:
        logger.warning(f"OFD_OPTIONAL_RESOURCE_MISSING: part={resource_part!r}")
        return registry
    base_loc = (root.get("BaseLoc") or "").strip()
    for element in root.iter():
        name = local_name(element.tag)
        resource_id = parse_int(element.get("ID"))
        if resource_id is None:
            continue
        if name == "Font":
            font_file = element_text(first_child(element, "FontFile"))
            font_part = _resource_asset_part(package, resource_part, base_loc, font_file) if font_file else None
            registry.fonts[resource_id] = FontResource(
                resource_id=resource_id,
                font_name=(element.get("FontName") or "").strip(),
                family_name=(element.get("FamilyName") or "").strip(),
                font_part=font_part,
                bold=_bool_attr(element.get("Bold")),
                italic=_bool_attr(element.get("Italic")),
            )
        elif name == "MultiMedia":
            media_file = element_text(first_child(element, "MediaFile"))
            media_part = _resource_asset_part(package, resource_part, base_loc, media_file) if media_file else None
            if media_part:
                registry.media[resource_id] = MediaResource(
                    resource_id=resource_id,
                    media_type=(element.get("Type") or "").strip(),
                    media_format=(element.get("Format") or "").strip(),
                    media_part=media_part,
                )
        elif name == "CompositeGraphicUnit":
            size = parse_numbers(f"{element.get('Width') or ''} {element.get('Height') or ''}", expected=2)
            registry.composites[resource_id] = CompositeResource(
                resource_id=resource_id,
                width=size[0] if size else 0.0,
                height=size[1] if size else 0.0,
                element=element,
            )
        elif name == "DrawParam":
            registry.draw_params[resource_id] = {str(key): str(value) for key, value in element.attrib.items()}
    return registry


def merge_registries(*registries: ResourceRegistry) -> ResourceRegistry:
    """按 Public→Document→Page 顺序合并资源并记录重复 ID。"""
    merged = ResourceRegistry()
    for registry in registries:
        for field_name in ("fonts", "media", "composites", "draw_params"):
            target = getattr(merged, field_name)
            incoming = getattr(registry, field_name)
            for resource_id, value in incoming.items():
                if resource_id in target:
                    logger.warning(f"OFD_DUPLICATE_RESOURCE_ID: kind={field_name}, id={resource_id}; later scope wins")
                target[resource_id] = value
    return merged


def resolve_draw_param(registry: ResourceRegistry, resource_id: int | None) -> dict[str, str]:
    """迭代解析 DrawParam Relative 继承，并限制深度、检测循环。"""
    if resource_id is None:
        return {}
    inheritance_chain: list[dict[str, str]] = []
    visited: set[int] = set()
    current_id: int | None = resource_id
    while current_id is not None:
        if current_id in visited:
            raise ValueError(f"OFD DrawParam cycle detected at id={current_id}")
        current = registry.draw_params.get(current_id)
        if current is None:
            break
        if len(inheritance_chain) >= MAX_DRAW_PARAM_INHERITANCE:
            raise OfdResourceLimitError(f"OFD resource limit exceeded: max_draw_param_inheritance={MAX_DRAW_PARAM_INHERITANCE}")
        visited.add(current_id)
        inheritance_chain.append(current)
        current_id = parse_int(current.get("Relative"))

    result: dict[str, str] = {}
    for current in reversed(inheritance_chain):
        result.update({key: value for key, value in current.items() if key not in {"ID", "Relative"}})
    return result


__all__ = ["merge_registries", "parse_resource_part", "resolve_draw_param"]
