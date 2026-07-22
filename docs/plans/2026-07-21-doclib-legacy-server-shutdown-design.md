# Doclib legacy server shutdown and stale discovery files

## Context

Endpoint schema version 2 binds discovery to a per-process `server_id`. After
an upgrade, however, an already-running version 1 server still owns the same
MinerU home and exposes an endpoint without that identifier. Treating the
endpoint as invalid can let the new CLI start a second server against the same
SQLite database and background queues.

Server shutdown also currently removes the endpoint and UDS path
unconditionally. An older process that exits after a replacement starts can
therefore remove files owned by the replacement.

## Decision

Endpoint version 1 is accepted only as a legacy identity format. Its PID is
compared with the PID returned by `GET /server/status`; matching values identify
the legacy process. Endpoint version 2 continues to use only `server_id` for
identity. PID remains diagnostic for version 2.

Server process shutdown and `mineru server stop` no longer unlink endpoint or
UDS files. Stop sends the shutdown request and waits until the identified
server is no longer reachable. Restart starts a replacement only after that
wait succeeds.

Stale discovery files are removed immediately before startup, after probing has
confirmed that no server identified by the endpoint is running. The new server
then binds its socket and atomically writes a fresh version 2 endpoint.

## Error handling

A version 1 endpoint whose PID differs from server status is rejected as
`server_instance_mismatch`. A shutdown that does not complete within the
15-second wait period fails with `service_unavailable` and does not start a
replacement.

## Verification

Tests cover version 1 parsing, PID match and mismatch, version 2 PID
independence, retained shutdown files, stale-file cleanup before startup, and
restart waiting. A real-process upgrade simulation verifies that stopping and
restarting never leaves two servers attached to one MinerU home.
