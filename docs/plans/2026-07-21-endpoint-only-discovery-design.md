# Endpoint-only Doclib discovery

## Context

The default `DoclibClient` resolver reads the endpoint file under the current
`MINERU_HOME`, but currently derives UDS or TCP transports from startup
configuration when that file is missing or invalid. A derived TCP endpoint can
connect to a server owned by another home.

## Decision

Default discovery treats `$MINERU_HOME/doclib.endpoint.json` as the sole source
of runtime transports. If the file is absent, invalid, or contains no usable
transport, the client has no connection candidates and reports that the server
is not running when a request is made.

Transports recorded in a valid endpoint file remain ordered with UDS before
TCP. Explicit `socket_path` and `base_url` constructor arguments remain direct
endpoint overrides and do not read the discovery file.

## Server identity validation

Each Doclib Server process generates a random `server_id`. Endpoint version 2
stores that identifier alongside the PID and transports, and
`GET /server/status` returns the same value. A default-discovery client lazily
probes each candidate transport before its first business request and only uses
a candidate whose reported `server_id` matches the endpoint.

This binds the endpoint file selected through `MINERU_HOME` to the exact server
process that wrote it without comparing runtime-specific path strings. A stale
endpoint whose TCP port or UDS path has been reused is rejected with
`server_instance_mismatch`. Version 1 endpoints and malformed version 2
endpoints are invalid and provide no connection candidates.

Explicit `socket_path` and `base_url` values represent a caller-selected server
and bypass endpoint discovery and instance validation. PID and `mineru_home`
remain diagnostic fields and are not identity checks.

## Verification

Unit tests cover endpoint schema round trips, valid endpoint ordering, missing
and invalid endpoint files, explicit UDS/TCP overrides, matching identities,
mismatched identities, and fallback to another transport after an identity
mismatch. Existing doclib and CLI server tests must remain green.
