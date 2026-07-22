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
configurable. Lock-file retention is platform-specific; its existence does not
indicate ownership and it must not contain required metadata. PID, `server_id`,
and transports remain in `doclib.endpoint.json`.

No separate owner metadata file is introduced. While `doclib.lock` remains the
ownership authority, `doclib.endpoint.json` also acts as a cooperative
diagnostic record for the current owner. PID is diagnostic only: clients must
not use it to establish ownership, automatically terminate a process, or decide
that a stale lock can be removed.

## Lifecycle

Startup proceeds in this order:

1. Create `MINERU_HOME` if necessary.
2. Acquire `doclib.lock` without waiting.
3. Remove stale endpoint and UDS files.
4. Create the app identity and atomically write an endpoint containing the
   current PID, `server_id`, and an empty transport list.
5. Bind transports and atomically update the endpoint with the usable transport
   list.
6. Initialize doclib state and start managed services through app lifespan.

Writing the empty-transport endpoint closes the normal startup diagnostic gap:
when ownership is held but no transport responds yet, CLI errors can report the
current owner's PID. If endpoint metadata is absent or invalid, the CLI omits
the PID rather than guessing. It never falls back to platform-specific lock
inspection or automatic process cleanup. A PID may belong to another runtime
namespace when a home is shared, so the message presents it only as reported
diagnostic information.

Shutdown stops the managed parse server and background workers, closes doclib
resources, removes endpoint and UDS discovery files while ownership is still
held, and finally releases `doclib.lock`. Abrupt process exit releases the OS
lock automatically; the next owner removes stale discovery files after it
acquires the lock.

CLI stop and restart wait for both endpoint unavailability and ownership lock
release. After acquiring the released lock, the CLI performs fallback endpoint
and UDS cleanup while still holding ownership. Endpoint disappearance alone does
not prove that shutdown cleanup has finished or that a replacement can safely
start.

Failure to acquire the lock exits startup with a concise message identifying
the owned home, but the user-facing message does not expose the lock path. The
exception carries no home or lock-path state because both are deterministically
derived from the active `MINERU_HOME`. This avoids duplicating state and
suggesting that deleting the lock file is a valid recovery action.
`DoclibLockUnavailable` is an internal control-flow marker and does not format a
user-facing message; CLI and direct process entrypoints render the ownership
message at their respective error boundaries.

Direct app startup and CLI `start`, `stop`, and `status` use the same ownership
message: `MinerU home [<home>] is currently owned by another doclib server
process`, with a parenthetical `reported PID <pid>` when valid current endpoint
metadata is available. The message does not claim that an endpoint is not
responding because discovery may instead be absent, incomplete during startup,
invalid, or temporarily unreachable. It does not remove endpoint, UDS,
database, or lock files. Endpoint state never overrides ownership.

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
that a replacement can start after the owner exits. Endpoint tests also cover
the early empty-transport diagnostic record and CLI error rendering with and
without an available PID.
