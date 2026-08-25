# Model Management Tier Scope Design

Status: Approved
Date: 2026-07-21

## Goal

Keep the four public parsing tiers (`flash`, `basic`, `standard`, and `advanced`) while reducing model installation and management to the two distinct model sets that actually exist: Basic and Standard.

## Packaging

The project exposes only `basic` and `standard` model-runtime extras. The `advanced` extra is removed because Advanced uses the same dependencies as Standard. Test dependencies use `mineru[standard]`.

Users preparing Advanced parsing install `mineru[standard]`; no separate Advanced compatibility extra is retained.

## Model Registry

`REPOS_FOR_TIER` and `model_repos_for_tier()` cover only:

- `basic`: `PDF-Extract-Kit-1.0`
- `standard`: `PDF-Extract-Kit-1.0` and `MinerU2.5-Pro-2605-1.2B`

Flash never enters model lookup because it does not use models. Advanced callers explicitly reuse the Standard model set before calling the registry. Passing `flash`, `advanced`, or an unrelated value directly to `model_repos_for_tier()` is an error.

## `mineru-kit models`

The `download`, `verify`, and `show` subcommands handle only Basic and Standard model tiers.

- `download --tier` accepts only `basic` and `standard`.
- `verify --tier` accepts only `basic` and `standard`.
- `show` lists only the Basic and Standard model sets.
- Passing Flash or Advanced returns an `invalid_request` error whose supported values are `basic, standard`.

Direct model repository arguments remain supported and unchanged.

## Managed Parse Server

Managed parse server model readiness remains available for Basic, Standard, and Advanced parsing tiers:

- Basic checks the Basic model set.
- Standard checks the Standard model set.
- Advanced explicitly checks the Standard model set.

When Advanced is missing model files, remediation points to:

```bash
mineru-kit models download --tier standard
```

Flash does not use the managed quality-tier model readiness path.

## Documentation

Model installation and management documentation describes only the Basic and Standard model sets. Advanced documentation explains that it reuses the Standard extra and model preparation commands. Parsing examples continue to expose all four public parsing tiers.

## Verification

Tests cover:

1. The absence of the `advanced` extra and use of `mineru[standard]` by test dependencies.
2. Basic and Standard model registry mappings.
3. Registry rejection of Flash and Advanced.
4. `models download` and `models verify` rejection of Flash and Advanced.
5. `models show` output containing only Basic and Standard tier rows.
6. Advanced managed-server readiness checking the Standard model set and recommending the Standard download command.
7. Documentation and tracked-file residue scans for removed commands and extras.
