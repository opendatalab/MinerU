"""Tier → backend mapping and engine resolution."""

from __future__ import annotations

from mineru.constants import Tier

TIER_BACKEND_MAP: dict[str, str] = {
    Tier.FLASH: "flash",
    Tier.STANDARD: "pipeline",
    Tier.PRO: "hybrid-auto-engine",
}


def resolve_backend(tier: str, backend: str | None = None) -> str:
    """Resolve tier to a concrete backend.  Explicit backend overrides tier."""
    if backend:
        return backend
    return TIER_BACKEND_MAP.get(tier, "pipeline")
