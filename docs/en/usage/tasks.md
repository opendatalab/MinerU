# Tasks API

This page describes the **4.1.0** contract. `/v1/tasks` is the shared asynchronous parsing API for API Server, Router, and compatible SaaS: submit a JSON manifest, upload missing files, poll, and download results per file. Existing clients should read the [4.1 migration guide](../reference/migration_4_1.md). For synchronous single-file requests, see the [file-parse API](file_parse.md).

## Startup and authentication

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier auto
```

The service exposes deployment-specific definitions at `/docs` and `/openapi.json`. When an API key is configured, send `Authorization: Bearer <API_KEY>` on API requests. Local services without a key do not require that header. The API base may include a reverse-proxy path prefix.

| Method and path | Success response | Purpose |
|---|---|---|
| `POST /v1/tasks` | 202, `CreateTaskResponse` | Submit a JSON manifest. |
| `PUT files[].upload.url` | Local/Router: 200, `UploadFileResponse` | Upload one file's raw bytes; accept successful 2xx responses from object storage. |
| `GET /v1/tasks/{task_id}` | 200, `TaskDetailResponse` | Read a complete status snapshot. |
| `POST /v1/tasks/{task_id}/cancel` | 200, `CancelTaskResponse` | Cancel unfinished files; no request body required. |

Local upload paths are `/v1/tasks/{task_id}/uploads/{file_id}`; clients should use the returned full URL. There are no task/file lists, public DELETE operations, or batch-result endpoint. The main API Server and Router do not expose `/v1/file_parse`.

## Creation request

`POST /v1/tasks` accepts only `application/json`. Put all business parameters in the JSON body. Query parameters, unknown fields, and invalid values return 400; other content types return 415.

| Field | Type | Default or requirement |
|---|---|---|
| `files` | object[] | Required, 1–100 entries; not a JSON-encoded string. |
| `tier` | string or null | Optional; `flash/basic/standard/advanced`, as described below. |
| `ocr_mode` | string | `auto`; accepts `auto/txt/ocr`. |
| `page_range` | string | Empty means the whole document; PDF ranges include `1-5`, `r1`, and `all`. |
| `output_formats` | string[] | `["default"]`; nonempty, values from `default/docx/html/latex`, deduplicated in first-occurrence order. |

Each `files[]` entry:

| Field | Type | Requirement |
|---|---|---|
| `file_name` | string | Required logical filename, 1–1024 characters, retaining its extension. Duplicate names are allowed. |
| `bytes` | integer or null | Required positive integer for uploads. Optional for URLs; when supplied, must match the actual size. |
| `url` | string or null | HTTP(S) download URL; omitted or null means a client upload. |
| `sha256sum` | string or null | Optional 64-character lowercase hexadecimal SHA-256 for validation and, for uploads, cache preflight. |
| `page_range` | string or null | Optional per-file PDF range overriding the global range. Non-PDF native documents still parse as a whole. |

Requests must omit `file_id/client_file_id`, including null values. The server allocates public `file_id` values in input order; duplicate names or content still receive different IDs. Use IDs to correlate uploads, queries, cancellation, and downloads. Local/Router result Files IDs match these IDs; Files lookup returns 404 before the result exists. Source-cache IDs remain private.

### Tier selection

- For PDFs/images, an omitted or null tier selects the first supported value in **standard → basic → flash → advanced** order. Advanced is selected automatically only when it is the sole available tier.
- An explicitly unsupported PDF/image tier rejects the entire creation request with `400 invalid_request`, `error.param="tier"`. No work is accepted and no tier substitution occurs.
- Every normally available compatible service supports existing native document formats. Office, HTML, and other native documents always use Flash, regardless of `/v1/tiers` or a valid requested tier. Empty strings and unknown enum values are still invalid.
- Mixed batches select tiers per file. On a Standard-capable service, a PDF + DOCX request without a tier uses Standard and Flash respectively. Selection is fixed at admission, not repeated during queuing or forwarding. Successful files report the actual value in `parse.tier`.

### Example request and creation receipt

Sizes, IDs, URLs, and timestamps below are illustrative; use the actual byte count for a real upload.

```json
{
  "files": [{"file_name": "report.pdf", "bytes": 12345}],
  "tier": "standard",
  "ocr_mode": "auto",
  "output_formats": ["default", "docx"]
}
```

202 creation receipt:

```json
{
  "task_id": "job_example",
  "status": "waiting-file",
  "files": [{
    "file_id": "file-example",
    "file_name": "report.pdf",
    "status": "waiting-file",
    "upload": {
      "url": "http://127.0.0.1:8000/v1/tasks/job_example/uploads/file-example",
      "headers": {},
      "expires_at": 1800003600
    }
  }]
}
```

202 means accepted. Creation receipts omit input sizes, parsing statistics, and result URLs; use GET for details even if a file has already completed.

URL inputs use the same creation endpoint, for example:

```json
{
  "files": [{"file_name": "report.pdf", "url": "https://example.com/report.pdf"}],
  "output_formats": ["html"]
}
```

The server downloads URL inputs without a client PUT; replace the example with a real download URL allowed by the deployment.

## Response fields

Success responses are direct objects without a `data/task` envelope. Below, `?` marks an optional field and is not part of its name.

| Response | Top-level fields | files[] fields |
|---|---|---|
| `CreateTaskResponse` | `task_id`, `status`, `files` | `file_id`, `file_name`, `status`, `upload?` |
| `TaskDetailResponse` | `task_id`, `status`, `created_at`, `started_at?`, `finished_at?`, `output_formats`, `progress`, `files`, `error?` | `file_id`, `file_name`, `status`, `bytes?`, `url?`, `upload?`, `parse?`, `result_url?`, `error?` |
| `CancelTaskResponse` | `task_id`, `status` (always canceled), `finished_at`, `files` | `file_id`, `file_name`, `status`, `parse?`, `result_url?`, `error?` |
| `UploadFileResponse` | `task_id`, `file_id`, `status` (the uploaded file's state) | None |

`task_id/file_id/file_name/status` are strings, `files` is an array, `bytes` is an integer, `url/result_url` are strings, and `output_formats` is a string array. Time fields are ISO-8601 UTC strings.

| Nested object | Fields and meaning |
|---|---|
| `upload` | `url: string`, `headers: object` (string pairs, possibly empty), `expires_at: integer` (Unix seconds). Present only when an upload is currently needed. |
| `parse` | `tier: string`, `duration_ms?: integer`, `parser_version?: string`. Successful files only; version identifies the executing worker. Duration includes source reading/downloading, validation, parsing, and export, excluding waits for upload, preparation slots, or parsing queues. |
| `progress` | Required integers `completed/failed/total`. Canceled files count as neither completed nor failed; total is the number of inputs. |
| `error` | Required nonempty `code: string`, `message: string`, and `param?: string`. May appear at the GET top level, in GET files, or in failed files of cancellation receipts; no type field. |

Tasks success responses recursively omit nulls, preserving `0/false/[]/{}`. Each response is a complete snapshot: an absent field has no current value, so do not retain stale upload links. Clients should tolerate future extra fields. Public HTTP errors also omit absent param values; Files and file-parse use separate null rules.

Example completed GET response (200):

```json
{
  "task_id": "job_example",
  "status": "completed",
  "created_at": "2026-10-28T00:00:00Z",
  "started_at": "2026-10-28T00:00:02Z",
  "finished_at": "2026-10-28T00:00:10Z",
  "output_formats": ["default", "docx"],
  "progress": {"completed": 1, "failed": 0, "total": 1},
  "files": [{
    "file_id": "file-example",
    "file_name": "report.pdf",
    "status": "completed",
    "bytes": 12345,
    "parse": {"tier": "standard", "duration_ms": 8000, "parser_version": "4.1.0"},
    "result_url": "http://127.0.0.1:8000/v1/files/file-example/content"
  }]
}
```

Example local PUT receipt: `{"task_id":"job_example","file_id":"file-example","status":"pending"}`. Successful object-storage PUT responses need not contain this JSON; query Tasks afterwards to confirm readiness.

## Uploads, URLs, and caching

PUT raw bytes using the returned `upload.url/headers`; no completion call is required. Upload only when a file is `waiting-file` and has an `upload` object. Otherwise poll. Each validated file queues independently without waiting for other files; waiting uploads do not reserve parsing capacity.

- Valid hash-based cache hits skip uploads. Only source bytes are reused; each task parses with its own parameters. Concurrent first uploads may transfer duplicate bytes, but source storage is deduplicated.
- URL inputs are downloaded by the server without an upload action, HEAD requirement, or Content-Length requirement. `file_name` determines the format; actual bytes are populated after download and supplied sizes/hashes are checked. URLs are downloaded each time, without hash-based download skipping.
- Local URL sources default to HTTPS only. HTTP requires `--allow-http-source`; defaults are a 60-second download timeout and a 200 MiB size limit. Direct client uploads have no fixed size or PDF page limit, though deployments may impose gateway limits.
- Upload links expire 3600 seconds after creation and are not renewed by polling. Uploads admitted before the deadline may finish; files not yet admitted become `failed/upload_expired`. Processing of successfully uploaded files is not limited by this deadline.
- Do not repeat successful PUTs. Query after a lost response: skip upload for `pending/running/completed`, poll `waiting-file` without an upload object, and retry within bounds only after failure when a valid upload action remains. Expired uploads need a new task. Chunk-level resumable uploads are not supported.

Local source caches default to 24-hour retention with collection every five minutes; active tasks and transfers protect source bytes. API/Router task records and indexes are in memory and are not guaranteed to survive restarts.

## Dynamic Router upstreams and cascading

A running Router can register, inspect, and safely deregister remote upstreams. An upstream can be a V1 API Server or another Router. Management affects only the receiving Router; it does not deregister downstream workers or stop remote processes. Locally managed workers are excluded.

Set an independent management key to enable the endpoints. Without a key, these paths return 404. `--admin-api-key` overrides `MINERU_ROUTER_ADMIN_API_KEY` and also works with `--reload`. This key is separate from parsing credentials and is never forwarded downstream; parsing authentication is unchanged.

```bash
# An empty pool can start: health returns 503 until a healthy upstream is added,
# while the management endpoints remain available.
export MINERU_ROUTER_ADMIN_API_KEY='replace-with-a-management-secret'
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus none

# Set the same key in another terminal before running these requests.
curl -X POST http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY" \
  -H 'Content-Type: application/json' -d '{"url":"http://worker-a:8000"}'
curl http://127.0.0.1:8002/v1/router/workers \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
# Use the worker_id returned by registration or listing.
curl -i -X DELETE http://127.0.0.1:8002/v1/router/workers/remote-1 \
  -H "Authorization: Bearer $MINERU_ROUTER_ADMIN_API_KEY"
```

| Endpoint | Response and behavior |
|---|---|
| `GET /v1/router/workers` | `200`, with `{"object":"list","data":[...]}` containing both startup-configured and dynamically registered remote nodes. |
| `POST /v1/router/workers` | Accepts only `url`. Returns `201` for a new URL or `200` for an existing normalized URL. Re-registration preserves its ID and references and resumes a draining node. |
| `DELETE /v1/router/workers/{worker_id}` | Returns `204` after removal if no references remain, or `202` with a draining snapshot otherwise. Unknown and locally managed nodes return `404 worker_not_found`. |

Each node snapshot contains `worker_id`, `url`, `state` (`active/draining`), `healthy`, `tiers`, `active_jobs`, and `retained_routes`. URLs must be HTTP/HTTPS base URLs; path prefixes are supported, while embedded credentials, query strings, and fragments are rejected. Invalid input returns `400 invalid_request`; missing or incorrect management credentials return `401 invalid_api_key`. Errors use `error.code/message` and optional `param`.

Registration performs an initial health and capability probe. An unreachable node or one returning invalid tier data remains registered with `healthy=false` and is retried by the existing periodic checks. Invalid capabilities do not replace the last valid snapshot or break management listing. Only healthy, active nodes participate in new scheduling and health/models/tiers aggregation. A successful health check does not undo draining.

Deregistration stops new reservations immediately. Existing reservations, upload handoffs, queries, cancellation, result leases, and downloads continue. Removal waits for execution reservations, requests/streams, execution and file routes, and leases to be released; `active_jobs=0` alone does not mean the remote service can stop. Results default to 24-hour retention, so a node may remain draining during that period. The five-minute collection loop removes eligible nodes; repeating DELETE also checks eligibility. With `--retention-seconds 0`, permanent result references can prevent removal indefinitely. There is no force-delete operation.

Files that have not yet been dispatched use the remaining eligible nodes. If none qualify, they fail with the existing `quality_tier_unavailable` error; no implicit tier downgrade or new waiting queue is introduced. Removing the last node is allowed. Absence from this Router's list only means this Router has released its references; the downstream service may still serve other callers.

For a two-level deployment, configure the outer Router with inner Router URLs:

```bash
mineru-kit router --local-gpus none \
  --upstream-url http://router-a:8002 \
  --upstream-url http://router-b:8002
```

The same POST endpoint can add Router C at runtime. Use independent worker pools below each inner Router and an acyclic topology. Global load accounting and cross-Router task takeover are not provided.

**Dynamic membership lasts only for the current process.** Registration and deregistration do not modify startup configuration. Restarting restores the `--upstream-url` list; dynamic nodes must be registered again, and worker IDs are not stable across restarts. The outer Router remains a single point of failure, and existing tasks are not guaranteed to recover after any Router restarts.

## States and cancellation

| File state | Meaning |
|---|---|
| `waiting-file` | Waiting for/receiving uploads, URL downloads, validation, or object-storage confirmation. |
| `pending` | Input ready, waiting for parsing resources. |
| `running` | Parsing, exporting, or saving results. |
| `completed` | All requested formats succeeded; the result is downloadable. |
| `failed` | Input, parsing, or export failed; inspect error. |
| `canceled` | An unfinished file was canceled. |

Parent aggregation prioritizes running, then pending, then waiting-file. All successful files yield completed; terminal success/failure mixtures yield partial; no successes yield failed. Cancellation sets the parent to canceled. A parent may move from running back to waiting-file while completed files remain downloadable.

Cancellation preserves completed/failed files and marks the others canceled. Receipt file states are therefore limited to completed/failed/canceled. Non-interruptible computation or cleanup may still finish internally, but late results cannot overwrite published cancellation states. Canceling an already terminal task returns `409 task_already_terminal`.

Example 200 receipt for canceling a single-file task that is still processing:

```json
{
  "task_id": "job_example",
  "status": "canceled",
  "finished_at": "2026-10-28T00:00:05Z",
  "files": [{"file_id": "file-example", "file_name": "report.pdf", "status": "canceled"}]
}
```

## Outputs and downloads

| output_formats | Result per input |
|---|---|
| `["default"]` | ZIP with `markdown.md`, `middle_json.json`, `structured_content.json`, and `images/`. |
| `["docx"]` | A DOCX with embedded images. |
| `["html"]` | An HTML file with embedded images; formula scripts retain renderer behavior. |
| `["latex"]` | ZIP with `document.tex` and `images/`. |
| Multiple formats | One flat ZIP, without nested ZIPs; DOCX/HTML names are `document.docx/document.html`. |

Each input is parsed once. One `files[].result_url` is published after all requested formats succeed. If any format fails, that file becomes `failed/result_packaging_failed` without partial artifacts; other files continue. There is no combined batch ZIP or top-level result_url. Completed files can be downloaded before the parent finishes.

ModelJson is internal. `include_model_output` is removed and returns 400 even when false; results do not contain `model_output.json`.

Local services and Router return Files URLs; SaaS may return signed object-storage URLs. Use returned result URLs without constructing storage paths. Attach the API key only to same-origin API requests, reevaluating every redirect; never forward it to object storage. During result retention, query again to renew an expired download signature. Polling and downloads do not render again, and unfinished files do not cause completed results to be collected early.

## Errors

Request-level errors use a non-2xx status and `{"error": ...}`, for example:

```json
{
  "error": {
    "code": "task_already_terminal",
    "message": "Task is canceled"
  }
}
```

| HTTP | Common code | Scenario |
|---|---|---|
| 400 | `invalid_request`, `page_range_invalid` | Invalid fields, tiers, ranges, or old parameters. |
| 400 | `upload_size_mismatch`, `file_hash_mismatch`, `upload_interrupted` | Upload validation or transfer failure. |
| 401 | `invalid_api_key` | Invalid credentials on an authenticated service. |
| 404 | `task_not_found`, `upload_not_found` | Task unavailable/collected, or no such upload file ID. |
| 409 | `task_already_terminal`, `file_already_terminal` | Canceling a terminal task or uploading to a file no longer accepting bytes. |
| 410 | `upload_expired` | Upload window expired. |
| 415 | `invalid_request` | Creation request is not JSON. |
| 503 | `quality_tier_unavailable`, `server_shutting_down`, or model preload codes | The service cannot accept the request. |

Asynchronous download, parsing, and export failures usually appear in **files[].error inside a 200 query snapshot**; HTTP 200 does not mean parsing succeeded. Public errors contain required nonempty code, message, and optional param; type is neither emitted nor accepted. Absent param values are omitted, although receivers also accept null. Branch on HTTP status and code rather than parsing message text. Inspect every file's state: an absent top-level error does not establish that all files succeeded.

## Runnable client examples

Run these Bash examples from the repository root. The standard-library Python client calculates real sizes and hashes, uploads files, and downloads per-file results:

```bash
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf notes.docx

OUTPUT_FORMATS=default,docx RESULT_DIR=results \
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf

# Resume with the original file list and order, without creating again.
TASK_ID=job_example MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh report.pdf
```

Set `MINERU_API_KEY` for authentication, `MINERU_TIER/OCR_MODE/PAGE_RANGE` for parsing options, and `SEND_HASH=0` to disable hash preflight. `MAX_POLLS/POLL_INTERVAL` default to 300/1 and bound polling; `CONNECT_TIMEOUT/REQUEST_TIMEOUT` bound individual transfers. Creation is not automatically retried. Exit codes: 0 all successful, 2 partial success, 1 failure/cancellation, 124 polling exhausted. Completed downloads remain on disk.

For direct HTTP calls, append `/v1/tasks/{URL-encoded task_id}` to the configured API base for queries, and append `/cancel` to that route for POST cancellation. Save creation IDs for recovery; Tasks does not return a status_url.
