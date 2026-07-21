# Central Tier Types and Constants Design

Date: 2026-07-21

## Goal

Make `mineru/types.py` the single source of truth for request, server, and deployment tier types and their finite value collections. Remove duplicate definitions and replace internal/private names with consistent public names.

## Canonical Definitions

Keep `Tier` and `ServerTier` in `mineru/types.py`, add a shared `DeploymentTier` for managed-server provisioning and model preparation, and define these explicit constants next to their corresponding types:

- `SERVER_TIERS: tuple[ServerTier, ...] = ("flash", "basic", "standard")`
- `DEPLOYMENT_TIERS: tuple[DeploymentTier, ...] = ("basic", "standard")`
- `TIERS: tuple[Tier, ...] = ("flash", "basic", "standard", "advanced")`
- `TIER_ORDER: dict[Tier, int] = {"flash": 0, "basic": 1, "standard": 2, "advanced": 3}`
- `TIERS_BY_SERVER_TIER: dict[ServerTier, tuple[Tier, ...]]`

The request-tier mapping remains:

- Flash server: Flash requests;
- Basic server: Flash and Basic requests;
- Standard server: Flash, Basic, Standard, and Advanced requests.

Tuple and mapping definitions remain explicit rather than being derived dynamically. The small amount of duplicated ordering in `TIER_ORDER` is intentional: it keeps rank semantics visible to static source analysis while preserving readable comparisons, sorting, and tolerant `.get(..., -1)` lookups.

## Migration

The managed-server configuration and model-repository selection layers both use `DeploymentTier` and `DEPLOYMENT_TIERS`. Basic and Standard are deployment profiles: Flash needs no managed deployment or model bundle, while Advanced reuses the Standard deployment.

Remove the redundant identity conversion between managed-server and model tiers. A validated deployment tier can be passed directly to model-repository selection.

No compatibility aliases or old-module re-exports remain. This is an internal source migration with no intended behavior, API payload, CLI option, error-message, or tier-order change.

## Import Boundaries

`mineru/types.py` remains dependency-light and does not import parser, doclib, kit, config, or model-registry modules. Consumers depend on `types.py`, so the move cannot introduce a reverse dependency or import cycle.

## Verification

Tests and static checks verify:

1. All five canonical constants and the explicit `TIER_ORDER` mapping have the expected ordered values.
2. API-server startup validation and request-tier expansion are unchanged.
3. Managed-server configuration accepts only Basic and Standard deployment tiers.
4. Model commands and model-repository selection use the same deployment tiers.
5. The superseded identifiers no longer occur in tracked code or documentation.
6. Ruff, import checks, and the tracked unit-test suite remain green.
