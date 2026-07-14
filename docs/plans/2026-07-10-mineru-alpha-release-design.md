# MinerU Alpha Release Design

## Goal

Publish PEP 440 Alpha versions such as `4.0.0a1` to the existing `mineru` PyPI project from an immutable Git tag, without changing the current stable or `mineru-next-dev` release flows.

## Release contract

- Alpha releases use annotated tags named `vX.Y.ZaN`, for example `v4.0.0a1`.
- The tagged commit must be contained in the remote `next` branch.
- `mineru/version.py` at the tagged commit must contain the same version as the tag without the leading `v`.
- The workflow checks out and builds the tagged commit, never the latest moving branch head.
- The built wheel metadata must contain the expected project name and version.
- A successful PyPI upload is followed by a GitHub Release marked as a pre-release.

## Workflow structure

The independent `.github/workflows/alpha-release.yml` workflow is triggered only by tags matching the broad Alpha tag glob. A Python validation helper then enforces the exact canonical tag format.

The build job:

1. checks out the tagged commit with full history;
2. fetches `origin/next` and verifies that the tagged commit is its ancestor;
3. validates the tag and `mineru/version.py`;
4. builds a wheel and runs package metadata checks;
5. installs the wheel without dependencies in a clean virtual environment and verifies its installed distribution version;
6. uploads the validated wheel as a workflow artifact.

The publishing job downloads exactly that artifact and publishes it with Twine, using the same PyPI API token authentication as the stable release workflow. The token is read from the `PYPI_TOKEN` GitHub secret through Twine's `TWINE_PASSWORD` environment variable. The job remains associated with the protected `pypi` GitHub environment.

The final job creates a GitHub pre-release and attaches the same wheel. It runs only after the PyPI publishing job succeeds.

## Security and failure behavior

- Third-party actions are pinned to immutable commit SHAs where practical.
- Default workflow permissions are read-only; write permissions are scoped to the jobs that need them.
- PyPI authentication uses the existing `PYPI_TOKEN` GitHub secret. The token should be project-scoped to `mineru` and rotated according to the release credential policy.
- Concurrency is scoped to the tag so duplicate runs for one tag cannot publish concurrently.
- Existing PyPI versions are not silently skipped. A duplicate upload fails visibly because published files are immutable.
- A validation, build, or smoke-test failure stops before the publishing job.

## Operator flow

1. Update `mineru/version.py` on `next`, for example to `4.0.0a1`.
2. Commit the version change and let normal CI pass.
3. Create and push the annotated tag `v4.0.0a1`.
4. Ensure the repository or `pypi` environment provides the `PYPI_TOKEN` secret.
5. Approve the protected `pypi` environment deployment if required.

The workflow implementation can be tested locally without creating or pushing a tag. Merely adding the workflow does not publish anything.
