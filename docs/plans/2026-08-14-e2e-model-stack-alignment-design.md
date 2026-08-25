# E2E Model Stack Alignment Design

Date: 2026-08-14

## Goal

Align the CLI end-to-end test document with the current README model-stack contract while keeping the suite deterministic and avoiding duplicate coverage unrelated to local parsing.

## README Corrections

The startup configuration default is `model.stack=auto`, not `light`. Its effective value is selected from the detected device:

```text
CPU -> light
accelerator -> full
```

`model.stack` is startup configuration and is not a supported key for `mineru config set`. The README must document `MINERU_MODEL_STACK=<stack>` and `config.yaml` as the supported overrides. Model download and verification examples must pass `--stack <stack>` explicitly when preparing a managed parse server.

## E2E Profiles

The full E2E suite runs with `MINERU_MODEL_STACK=light`. This profile is deterministic across hardware and exercises all CLI behavior, including local Basic, Standard, and Advanced PDF parsing.

A focused mandatory `full` profile then exercises only local PDF parsing for Basic, Standard, and Advanced. It does not duplicate remote parsing, watch, scan, CRUD, telemetry, or maintenance cases because those behaviors do not depend on the model runtime stack.

The `full` profile must:

1. Stop the current server.
2. Start a replacement server with `MINERU_MODEL_STACK=full` in its environment.
3. Wait until the managed parse server is healthy.
4. Force new Basic, Standard, and Advanced parses and verify `privacy=local`, `via=local`, the requested tier, completed status, and non-null content.
5. Stop the server and restore the `light` profile before later cases run.

## Installation And Models

The E2E environment installs `.[full]` because the mandatory profile matrix includes the optional full runtime. The base package already contains the light runtime.

Setup downloads and verifies both Standard deployment model sets explicitly:

```text
standard/light
standard/full
```

Each Standard model set also covers the corresponding Basic models and supports Advanced requests through a Standard managed server. Model preparation remains setup work and may use `mineru-kit`; formal CLI cases continue to use `mineru` only.

## Tier Selection

Implicit quality-tier selection is:

```text
standard -> basic -> error
```

Advanced remains available only when requested explicitly. An Advanced-only remote server does not provide an implicit default tier. Cached-result selection remains:

```text
advanced -> standard -> basic -> flash
```

## Verification

Documentation verification must confirm:

1. No E2E setup step refers to the removed `standard` extra.
2. Every setup model download and verification command specifies a stack.
3. The main suite and isolated missing-model cases inherit the deterministic light profile.
4. The mandatory full profile restores light before subsequent cases.
5. Default-tier expectations contain Standard and Basic only.
6. README examples use supported startup-configuration mechanisms.
7. Markdown has no whitespace errors and all referenced CLI options exist.
