# API Server Startup Tier Design

Date: 2026-07-21

## Goal

Separate the parse server's startup capability tier from the public request tier. A server starts at one capability ceiling and accepts every request tier covered by that ceiling. The public request API continues to expose `flash`, `basic`, `standard`, and `advanced`.

Standard and Advanced have identical installation, model, and hardware requirements. Starting a Standard server therefore also enables Advanced requests. Flash remains available for standalone and Gradio deployments, while Doclib managed parse servers disable it because Doclib executes Flash parsing in-process.

## Startup Configuration

The startup-level tier is a single value:

- `flash`
- `basic`
- `standard`

Advanced is not a startup-level value. It remains a valid request-level tier.

When no startup tier is supplied, the server uses `standard`. The command line exposes one additional flag, `--no-flash`, which defaults to false.

The startup capability expansion is:

| Startup configuration | Advertised request tiers |
|---|---|
| `--tier flash` | `flash` |
| `--tier basic` | `flash`, `basic` |
| `--tier basic --no-flash` | `basic` |
| `--tier standard` | `flash`, `basic`, `standard`, `advanced` |
| `--tier standard --no-flash` | `basic`, `standard`, `advanced` |

`--tier flash --no-flash` is invalid because it would leave the server with no capabilities. Server startup must fail with a clear parameter-conflict error.

Both `python -m mineru.parser.api_server` and `mineru-kit api-server` use these semantics. The startup tier option is no longer repeatable.

## Internal Model

The implementation distinguishes the startup type from the public request type. The startup type accepts only `flash`, `basic`, and `standard`; the request `Tier` continues to accept all four public parsing tiers.

An explicit, statically visible mapping expands the startup tier into request capabilities. `create_app()` receives one startup tier and a `no_flash` flag, then builds:

- the advertised `/v1/tiers` metadata;
- the request-tier-to-runtime mapping;
- the default request tier;
- model metadata for the enabled capabilities.

`app.state.tier` represents the single startup tier. `app.state.tiers` and `app.state.tier_runtime_options` represent the expanded request capabilities. `app.state.default_tier` remains separate.

Dependency preflight checks the startup ceiling rather than each expanded request tier. The Standard optional dependency group includes Basic dependencies, and the Standard model set includes the Basic model set. Standard also covers Advanced because Advanced reuses Standard dependencies and models.

## Request Behavior

`CreateJobRequest.tier`, `MinerUApiParser.tier`, and the public REST payload remain unchanged.

Request routing validates the requested tier against the expanded startup capabilities:

- a Standard server accepts Basic, Standard, and Advanced requests;
- a Basic server rejects Standard and Advanced requests;
- a server started with `--no-flash` rejects explicit Flash requests.

`--no-flash` has strict execution semantics. It does not merely hide Flash from `/v1/tiers`; it prevents the process from running the Flash backend. Inputs such as Office or HTML files that require Flash normalization are rejected by a no-Flash server instead of being silently converted to Flash execution.

Without `--no-flash`, the existing file-type normalization behavior remains.

Default request behavior remains:

- a Standard server defaults to Standard;
- a Basic server defaults to Basic;
- a Flash-only server retains the existing explicit-Flash requirement for PDF and image requests.

## Deployment Modes

The standalone API server defaults to Standard with Flash enabled, so it advertises all four request tiers.

Gradio's embedded API server uses Standard with Flash enabled. Its tier discovery and dropdown therefore continue to expose all four public tiers.

Doclib managed parse servers accept only `basic` or `standard` as `parse_server.local.managed_tier`. Their child process command always includes `--no-flash`:

- managed Basic advertises `basic`;
- managed Standard advertises `basic`, `standard`, and `advanced`.

Existing database values outside `basic` and `standard`, including the former `advanced` value, are not migrated, deleted, or rewritten. They are ignored at runtime and the effective managed startup tier falls back to the default `standard`. New configuration writes reject values outside `basic` and `standard`.

## Errors

Startup rejects:

- an unsupported startup tier, including `advanced`;
- the conflicting combination `--tier flash --no-flash`.

Request handling rejects:

- a request tier outside the server's expanded capabilities;
- explicit Flash requests when Flash is disabled;
- inputs that require Flash execution when Flash is disabled.

Errors use the existing API error envelope and identify the relevant `tier` or input parameter. No request schema changes are introduced.

## Documentation

Normative documentation must describe startup tier as a single capability ceiling and request tier as the four-value parsing choice. Update the API server CLI documentation, architecture and tier documentation, relevant ADR text, the root `README.md`, and `skills/mineru/SKILL.md`.

Advanced setup instructions must say to install the Standard extra, download and verify the Standard model set, start or configure a Standard server, and select Advanced at request time. Documentation must not instruct users to start an Advanced server or set `parse_server.local.managed_tier advanced`.

## Verification

Tests cover:

1. Capability expansion for Flash, Basic, and Standard startup tiers.
2. Strict Flash removal with `--no-flash`.
3. Rejection of `--tier flash --no-flash`.
4. Standard accepting Advanced requests and Basic rejecting them.
5. Matching semantics in both API server CLI entry points.
6. Managed child processes receiving `--no-flash`.
7. Managed configuration accepting only Basic and Standard.
8. Invalid stored managed tiers falling back to Standard without database writes.
9. Gradio continuing to discover all four public request tiers.
10. Documentation residue checks for repeatable startup `--tier`, Advanced startup commands, and `parse_server.local.managed_tier advanced`.
