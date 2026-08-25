# PDF Render Pool Diagnostics Design

## Context

On Windows, managed local parsing can time out while waiting for PDF image rendering even though direct `pypdfium2` rendering and a standalone call to the same rendering helper complete quickly. The current timeout only reports the page range and duration, so it cannot distinguish task submission problems, worker startup failures, worker-side stalls, or a worker result that never reaches the parent process.

## Decision

Add diagnostics at the existing process-pool boundary without changing rendering behavior:

- Log task submission in the parent process at debug level, including the parent PID, thread, executor identity, and page range.
- Log worker entry, completion, and exception through Loguru, including the worker PID, page range, elapsed time, and output count.
- When a timeout occurs, log one warning containing each future's state and each known worker's PID, liveness, and exit code.
- Log successful completion at debug level with total elapsed time.
- Do not log file names, paths, PDF contents, or image contents.

The existing timeout, process count, task scheduling, executor recycling, and raised exception remain unchanged.

## Failure Isolation

The combined logs identify the failing stage:

- No worker-entry log indicates that the submitted task did not begin executing.
- Worker entry without completion or exception indicates a worker-side render stall.
- Worker completion without a completed future indicates result serialization or IPC delivery trouble.
- A dead worker with an exit code indicates process termination before normal task completion.

Diagnostic collection must be best effort. Inspecting executor internals must never replace or hide the original render timeout.

## Testing

Unit tests cover deterministic future and worker state snapshots, worker lifecycle logging, and timeout logging with a fake executor. Existing PDF rendering tests continue to cover behavior; the new tests must not start real subprocesses or wait for the production timeout.
