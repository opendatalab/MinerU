# Doclib home ownership lock

## Context

`doclib.start.lock` serializes the CLI launch window, but it is released after
startup and can be bypassed by invoking `python -m mineru.doclib.app` directly.
The endpoint file is discovery metadata rather than an ownership primitive, and
SQLite WAL permits multiple processes to open the same database. Consequently,
two doclib servers can operate on one `MINERU_HOME` concurrently.

## Decision

Add `$MINERU_HOME/doclib.lock` as the authoritative ownership lock for all
mutable doclib state in that home. The doclib server process itself acquires a
non-blocking `filelock.FileLock` before removing stale discovery files,
initializing SQLite, or starting the managed parse server. It retains the lock
until all server shutdown work has completed.

`doclib.start.lock` remains a short-lived CLI launch coordinator. It improves
concurrent CLI behavior but does not establish ownership. The server-side
`doclib.lock` is authoritative and also covers direct app invocation.

The lock path is always derived from `MINERU_HOME` and is not independently
configurable. The lock file may remain on disk permanently; its existence does
not indicate ownership and it must not contain required metadata. PID,
`server_id`, and transports remain in `doclib.endpoint.json`.

## Lifecycle

Startup proceeds in this order:

1. Create `MINERU_HOME` if necessary.
2. Acquire `doclib.lock` without waiting.
3. Remove stale endpoint and UDS files.
4. Create the app, initialize doclib state, and start managed services.
5. Write the endpoint only after transports bind successfully.

Shutdown stops the managed parse server and background workers, closes doclib
resources, removes endpoint and UDS discovery files while ownership is still
held, and finally releases `doclib.lock`. Abrupt process exit releases the OS
lock automatically; the next owner removes stale discovery files after it
acquires the lock.

Failure to acquire the lock exits startup with a concise message identifying
the home and lock path. It does not remove endpoint, UDS, database, or lock
files. A live endpoint may still be queried for diagnostics, but endpoint state
never overrides ownership.

The guarantee applies between versions that implement `doclib.lock`. Existing
endpoint probing remains useful during upgrades, but an older process that does
not participate in the ownership protocol cannot be excluded solely by the new
lock.

## Portability

`filelock.FileLock` provides native process-held locks on Windows, macOS, and
Linux. This design guarantees one owner per home when all participants observe
the same native lock domain. It does not claim reliable mutual exclusion across
Windows and WSL, or on filesystems whose lock semantics are not coherent.

## Verification

Tests cover lock-path derivation, direct double startup, concurrent CLI startup,
lock release after graceful and abrupt exit, startup cleanup only after lock
acquisition, and preservation of discovery files when acquisition fails. A
real-process test confirms that only one server can initialize a shared home and
that a replacement can start after the owner exits.
