# MinerU Next Dev Daily Release Design

## Goal

Add a new GitHub Actions workflow in the personal repository `johnking0099/MinerU-Repo` to publish a daily development package for the `next` branch.

The published package is separate from the official package:

- package name: `mineru-next-dev`
- version scheme: `4.0.0.devYYYYMMDD`

The workflow should run once per day. It should publish only when the `next` branch source commit changed since the last successful publish.

## Scope

This design covers:

- workflow trigger strategy
- package naming and versioning
- source-change detection
- release state persistence
- PyPI publishing flow

This design does not cover:

- changing the official `mineru` release workflow
- changing upstream `opendatalab/MinerU` workflows
- supporting multiple scheduled publishes per day

## Requirements

1. The workflow runs in the personal repository, not the upstream repository.
2. The workflow checks out the `next` branch as the release source.
3. The workflow runs once per day on a schedule.
4. If `next` has no source change since the last successful publish, the workflow exits without publishing.
5. If `next` changed, the workflow publishes a package to the real PyPI index.
6. The published package name is `mineru-next-dev`.
7. The published version is `4.0.0.devYYYYMMDD`.
8. The workflow must support manual execution through `workflow_dispatch`.
9. Release state must not mutate the `next` branch itself.

## Chosen Approach

Use two branches with distinct roles:

- `next`: source branch used to build the package
- `release-state`: state-only branch used to record the last successful published commit

The workflow will:

1. Checkout `next`
2. Resolve the current `next` HEAD commit SHA
3. Read the last published SHA from a state file on `release-state`
4. Compare the two SHAs
5. Skip publish when they match
6. Patch package metadata locally in the workflow workspace
7. Build and upload the distribution to PyPI
8. Update the state file on `release-state` after a successful publish

This avoids the main failure mode of writing state back to `next`, which would otherwise change `HEAD` and force false-positive daily releases.

## State File

State is stored in the repository on branch `release-state`.

Suggested path:

`/.github/release-state/mineru-next-dev.json`

Suggested format:

```json
{
  "package": "mineru-next-dev",
  "version": "4.0.0.dev20260624",
  "source_branch": "next",
  "source_commit": "abc123def456",
  "published_at": "2026-06-24T02:00:00Z"
}
```

## Versioning Rules

The package version is always generated from the current UTC date:

- `4.0.0.devYYYYMMDD`

Examples:

- `4.0.0.dev20260624`
- `4.0.0.dev20260625`

Daily scheduled execution means the normal path produces at most one publish attempt per day. Manual reruns are expected to be for recovery or inspection, not for additional same-day release numbering.

## Package Metadata Strategy

The workflow should patch package metadata only in the transient workflow workspace.

Recommended local modifications before build:

1. patch `[project].name` in `pyproject.toml` from `mineru` to `mineru-next-dev`
2. patch `mineru/version.py` to the generated `4.0.0.devYYYYMMDD`

These changes should not be committed back to `next`.

This keeps the release package separate from the official package without creating repository noise.

## Workflow Trigger Strategy

The new workflow should support:

- `schedule`
- `workflow_dispatch`

The schedule should run once per day at a fixed UTC time. The exact cron expression can be tuned later, but the design assumes one scheduled run per day.

## Publish Decision Logic

The workflow should publish only when:

- current `next` HEAD SHA differs from the stored `source_commit` on `release-state`

If the state file does not exist yet, the workflow should treat the run as the initial publish and continue.

If the publish fails, the state file must not be updated.

## Authentication and Secrets

Required repository secrets:

- `PYPI_TOKEN`: used to upload `mineru-next-dev` to PyPI

Required workflow permissions:

- `contents: write`

`contents: write` is needed only for updating the `release-state` branch after a successful publish.

## Failure Handling

1. State write must happen only after PyPI upload succeeds.
2. If build or upload fails, the workflow should leave `release-state` unchanged.
3. If `release-state` is missing, the workflow should initialize it.
4. If the package version already exists on PyPI for the day, manual rerun behavior should fail clearly instead of mutating state incorrectly.

## Alternatives Considered

### Alternative 1: store state on `next`

Rejected.

Reason:

- writing state to `next` changes the branch `HEAD`
- future runs would always see a changed commit
- this creates false-positive daily publishes

### Alternative 2: query PyPI for the last published version and infer commit

Rejected.

Reason:

- harder to reconstruct source commit accurately
- introduces external-state coupling
- more complex than repository-controlled state

### Alternative 3: use GitHub Releases or artifacts as state

Rejected.

Reason:

- possible, but less transparent and harder to inspect manually
- repository file state is simpler and easier to repair

## Implementation Plan

Implementation should add:

1. a new workflow under `.github/workflows/`
2. a small helper script to:
   - compute the release version
   - read/write the release-state JSON
   - patch package metadata in the workspace
3. initial bootstrap handling for the `release-state` branch

## Open Decisions Already Resolved

- publish target: real PyPI
- source branch: `next`
- package name: `mineru-next-dev`
- version scheme: `4.0.0.devYYYYMMDD`
- state storage: repository branch `release-state`
- scheduled frequency: once per day

