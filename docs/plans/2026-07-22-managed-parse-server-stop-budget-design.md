# Managed Parse Server Stop Budget Design

## Problem

`mineru server stop` waits up to 15 seconds for doclib to stop and release its HOME ownership lock. Managed parse-server shutdown currently applies `parse_server_stop_timeout_sec` independently to stdin EOF, terminate, and kill. With the default value of 10 seconds, shutdown can consume up to 30 seconds before doclib continues its remaining cleanup.

During model preload, the parse-server process can remain inside model initialization after its stdin watcher sets `server.should_exit`. In the observed E2E failure, it ignored stdin EOF for 10 seconds and terminate for another 10 seconds. The CLI returned after 15 seconds while doclib still held the ownership lock.

## Decision

Treat `parse_server_stop_timeout_sec` as a total managed parse-server stop budget.

For the default 10-second budget:

- Healthy or established process: wait up to 5 seconds after stdin EOF.
- Startup or preload process: wait no more than 2 seconds after stdin EOF.
- Wait up to 3 seconds after terminate.
- Use the remaining budget after kill to reap the process.

Every stage derives its timeout from one monotonic deadline. Time used by an earlier stage is not added back to the total budget.

The existing 15-second CLI stop timeout remains unchanged. This leaves approximately 5 seconds for the remaining background tasks, database closure, endpoint cleanup, and ownership lock release.

## Scope

The shutdown policy remains centralized in `stop_managed_parse_server`.

Callers pass whether the managed process is still starting from the existing `ParseServerHealth.local_starting` state. No new configuration keys or timeout logic are added to individual background tasks.

The parse-server stdin watcher and model preload implementation remain unchanged. Making model initialization cooperatively cancellable would require broader model-runtime changes and is not required to bound doclib shutdown.

## Failure Handling

- Closing stdin remains the first shutdown action.
- If the graceful stage exceeds its allocation, send terminate.
- If terminate exceeds its allocation, send kill.
- If kill or reap fails, log the failure and let doclib continue shutdown instead of extending the configured budget.
- A process that exits at any stage returns immediately.

## Tests

- Verify EOF, terminate, and kill use portions of one total deadline instead of receiving the full timeout independently.
- Verify startup/preload state receives a shorter graceful interval.
- Verify a process ignoring EOF and terminate is killed and reaped within the configured total budget.
- Verify restart and doclib shutdown pass the existing startup state to the centralized helper.
- Re-run CLI server lifecycle unit tests and the focused `JSONERR-004` to `JSONERR-005` E2E sequence.
