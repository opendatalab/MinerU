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

This change does not add server home validation. That is a separate follow-up
needed to reject stale endpoint files that resolve to a live server owned by a
different home.

## Verification

Unit tests cover valid endpoint ordering, missing and invalid endpoint files,
and explicit UDS/TCP overrides. Existing doclib and CLI server tests must remain
green.
