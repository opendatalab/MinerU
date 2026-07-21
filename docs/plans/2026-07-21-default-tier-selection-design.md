# Default Tier Selection Design

Date: 2026-07-21

## Goal

Simplify implicit parse-tier selection to match the new server capability contract. A conforming server that offers Advanced also offers Standard, so Advanced no longer needs to be an implicit fallback.

## Selection Rules

When a user submits a quality parse request without a tier, select:

```text
standard -> basic -> error
```

Advanced remains a valid quality tier but must be requested explicitly. Flash remains an explicit preview tier for ordinary requests.

Background parsing rules may fall back to Flash and therefore select:

```text
standard -> basic -> flash
```

Cached-result selection remains unchanged:

```text
advanced -> standard -> basic -> flash
```

An already-created Advanced result is still the highest-quality reusable cache even though Advanced is not selected implicitly for new work.

## Type Constants

Separate the default selection order from quality-tier membership:

- `DEFAULT_QUALITY_TIER_SELECTION_ORDER` contains Standard and Basic.
- `QUALITY_TIERS` contains Basic, Standard, and Advanced.
- `PARSING_RULE_TIER_SELECTION_ORDER` appends Flash to the default order.
- `CACHED_TIER_SELECTION_ORDER` remains unchanged.

This prevents removing Advanced from file-type validation and explicit request handling when it is removed from the default order.

## Compatibility Boundary

Current first-party API servers never advertise Advanced without Standard. An external or obsolete Advanced-only server is not treated as having an implicit default quality tier. This is intentional and does not affect explicit `--tier advanced` requests to conforming Standard servers.

## Verification

Tests verify:

1. Standard is preferred over Basic.
2. Basic is selected when Standard is unavailable, even if Advanced is advertised.
3. Advanced alone does not become an implicit default.
4. Advanced remains a member of `QUALITY_TIERS`.
5. Parsing rules use Standard, Basic, then Flash.
6. Cached-result selection still prefers Advanced.
7. API-server, Doclib, Gradio, CLI, and tracked unit tests remain green.
