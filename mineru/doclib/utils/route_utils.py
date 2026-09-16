"""HTTP route metadata shared by doclib client and server implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

RouteMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

F = TypeVar("F", bound=Callable[..., Any])
_global_route_index = 0


@dataclass(frozen=True, slots=True)
class RouteInfo:
    """HTTP route metadata attached to concrete doclib implementation methods."""

    method: RouteMethod
    path: str
    tags: tuple[str, ...] = ()
    index: int = 0

    def __post_init__(self) -> None:
        if self.method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            raise ValueError(f"unsupported route method: {self.method}")
        if not self.path.startswith("/"):
            raise ValueError(f"route path must start with '/': {self.path}")
        if self.path != "/" and self.path.endswith("/"):
            raise ValueError(f"route path must not end with '/': {self.path}")
        if any(not tag for tag in self.tags):
            raise ValueError("route tags must not contain empty values")
        if self.index < 0:
            raise ValueError(f"route index must be non-negative: {self.index}")


def route(method: RouteMethod, path: str, *, tags: Sequence[str] = ()) -> Callable[[F], F]:
    """Attach HTTP route metadata to a concrete client/server method."""

    global _global_route_index
    _global_route_index += 1
    route_info = RouteInfo(method=method, path=path, tags=tuple(tags), index=_global_route_index)

    def decorator(func: F) -> F:
        setattr(func, "_route_info", route_info)
        return func

    return decorator


def get_route_info(func: Callable[..., Any]) -> RouteInfo:
    """Return route metadata attached by ``@route``."""

    target = getattr(func, "__func__", func)
    route_info = getattr(target, "_route_info", None)
    if not isinstance(route_info, RouteInfo):
        name = getattr(target, "__qualname__", repr(target))
        raise ValueError(f"{name} does not have doclib route metadata")
    return route_info


def has_route_info(func: Callable[..., Any]) -> bool:
    """Return whether a callable has doclib route metadata."""

    target = getattr(func, "__func__", func)
    return isinstance(getattr(target, "_route_info", None), RouteInfo)
