# Remote Transport Retry Design

Issue: #99

## Goal

Make transient transport failures in the v1 API client recoverable when replay is safe, and expose exhausted failures as structured transport errors instead of `parse_failed`.

## Retry Boundary

The client retries requests only when replay cannot create duplicate parse jobs:

- Retry `GET` requests for health, upload state, job polling, and output downloads.
- Retry the upload content `PUT` with the same bytes and headers.
- After a transport failure from upload completion, query the upload. Return its file when completed, or retry completion while it remains pending.
- Do not automatically retry `POST /v1/parse/jobs`. The API does not currently provide an idempotency contract, so a lost response may already represent a created and billable job.

Retries use a small bounded exponential delay. HTTP application errors are not transport retries; they continue through the existing structured response handling.

## Error Model

Classification uses exception types rather than message matching:

- `httpx.TimeoutException` becomes `remote_timeout` for a remote target.
- Other exhausted `httpx.TransportError` instances become `remote_unreachable` for a remote target.
- Local API targets retain parse-server-oriented error semantics.

The internal transport exception carries the request stage so logs can distinguish health discovery, upload creation, byte upload, upload completion, job submission, polling, and output download. It must not expose credentials or signed upload URLs.

If a remote job itself returns `parse_failed`, the client preserves that structured server result. It does not infer a transport failure from text in the server message.

## Implementation Shape

`mineru/parser/api_client.py` owns matching synchronous and asynchronous request helpers. The helpers apply the retry policy and raise a typed internal transport error after exhaustion. Upload completion adds state recovery through `GET /v1/uploads/{upload_id}`.

`ParseService` maps typed transport failures according to the selected API target and records the resulting error code on the parse task. Existing `_V1APIError` handling remains responsible for structured API and job errors.

Job submission idempotency is a separate API change. Once the server supports an idempotency key, the client can safely retry `POST /v1/parse/jobs` with a stable key.

## Tests

Tests cover synchronous and asynchronous behavior:

- transient GET and PUT failures recover within the retry budget;
- upload completion recovers a completed upload after a lost response;
- job creation is attempted once after a transport failure;
- exhausted timeout and transport failures map to the documented remote codes;
- HTTP errors and remote job `parse_failed` responses are not retried or reclassified;
- retry logs and exceptions identify the stage without exposing request secrets.
