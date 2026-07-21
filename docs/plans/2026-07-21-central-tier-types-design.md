# Central Tier Types and Constants Design

Date: 2026-07-21

## Goal

Make `mineru/types.py` the single source of truth for server, managed-server, and model tier types and their finite value collections. Remove duplicate definitions and replace internal/private names with consistent public names.

## Canonical Definitions

Keep the existing `Tier`, `ServerTier`, and `ManagedServerTier` types in `mineru/types.py`, add `ModelTier`, and define these explicit constants next to their corresponding types:

- `SERVER_TIERS: tuple[ServerTier, ...] = ("flash", "basic", "standard")`
- `MANAGED_SERVER_TIERS: tuple[ManagedServerTier, ...] = ("basic", "standard")`
- `MODEL_TIERS: tuple[ModelTier, ...] = ("basic", "standard")`
- `TIERS: tuple[Tier, ...] = ("flash", "basic", "standard", "advanced")`
- `TIER_ORDER: dict[Tier, int] = {tier: index for index, tier in enumerate(TIERS)}`
- `TIERS_BY_SERVER_TIER: dict[ServerTier, tuple[Tier, ...]]`

The request-tier mapping remains:

- Flash server: Flash requests;
- Basic server: Flash and Basic requests;
- Standard server: Flash, Basic, Standard, and Advanced requests.

Tuple definitions remain explicit rather than being derived dynamically from `Literal` annotations. `TIER_ORDER` is a convenience rank index derived from the canonical `TIERS` tuple, avoiding a second manually maintained tier ordering while preserving readable comparisons, sorting, and tolerant `.get(..., -1)` lookups.

## Migration

Replace the old names everywhere: the managed-server collection becomes `MANAGED_SERVER_TIERS`, both duplicated API-server collections become `SERVER_TIERS`, and the request-tier expansion mapping becomes `TIERS_BY_SERVER_TIER`.

Move `ModelTier` and `MODEL_TIERS` out of `mineru/utils/model_registry.py`. Consumers import them directly from `mineru/types.py`; the registry no longer re-exports them.

No compatibility aliases or old-module re-exports remain. This is an internal source migration with no intended behavior, API payload, CLI option, error-message, or tier-order change.

## Import Boundaries

`mineru/types.py` remains dependency-light and does not import parser, doclib, kit, config, or model-registry modules. Consumers depend on `types.py`, so the move cannot introduce a reverse dependency or import cycle.

## Verification

Tests and static checks verify:

1. All five canonical constants have the expected ordered values and mapping, and `TIER_ORDER` matches the indices of `TIERS`.
2. API-server startup validation and request-tier expansion are unchanged.
3. Managed-server configuration accepts only Basic and Standard.
4. Model commands and model-repository selection still accept only Basic and Standard.
5. The superseded identifiers no longer occur in tracked code or documentation.
6. Ruff, import checks, and the tracked unit-test suite remain green.
