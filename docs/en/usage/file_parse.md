# file-parse API: Synchronous Single-File Parsing

This page describes the **4.1.0** interface. `mineru-kit file-parse-server` is a standalone HTTP wrapper: it accepts one file, creates and uploads a [Task](tasks.md), waits, and returns a ZIP, DOCX, or HTML attachment. It connects to a local API, Router, or compatible SaaS without loading parsing models.

## Startup

Start the upstream parser, then start the wrapper in another terminal:

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier auto
```

```bash
mineru-kit file-parse-server --api-url http://127.0.0.1:8000
```

| Option | Default | Meaning |
|---|---|---|
| `--api-url` | `MINERU_API_URL`, otherwise required | Upstream API base, optionally with a proxy prefix; do not append `/v1/tasks`. |
| `--api-key` | `MINERU_API_KEY`, otherwise empty | Bearer token for upstream requests only. |
| `--host`, `--port` | `127.0.0.1`, `8003` | Wrapper listen address. |
| `--state-dir` | `${MINERU_HOME:-~/.mineru}/file-parse` | Recovery database and temporary transfer directory. |
| `--retention-seconds` | `86400` | Record retention after first observing a terminal state; 0 disables collection. |

The wrapper has no API-key authentication of its own and is intended for local or trusted-network use. Add authentication at a reverse proxy when exposing it externally. Incoming Authorization is not forwarded upstream; requests cannot override the upstream address or credentials.

The wrapper has its own `/docs` and `/openapi.json`. The routes below exist only on this service, not on the main API Server or Router.

## Request fields

`POST /v1/file_parse` accepts only `multipart/form-data`. Query parameters, unknown fields, multiple files, and repeated scalar fields are rejected; `output_formats` may be repeated.

| Field | Default | Meaning |
|---|---|---|
| `file` | Required | One nonempty file, retaining its extension. |
| `tier` | Upstream default | Optional `flash/basic/standard/advanced`, affecting PDFs/images only. |
| `ocr_mode` | `auto` | `auto/txt/ocr`. |
| `page_range` | Whole document | PDF ranges include `1-5`, `r1`, and `all`; the upstream validates format-specific support. |
| `output_formats` | `default` | `default/docx/html/latex`; repeat the field for a list, deduplicated in first-occurrence order. |
| `wait_timeout` | `3600` | Integer seconds, 0–3600. |
| `request_id` | Generated | 1–64 letters, digits, underscores, or hyphens; supply and save it before submission to recover after a disconnect. |

Omitting tier lets the upstream choose Standard → Basic → Flash → Advanced for PDFs/images; Advanced is selected automatically only when it is the sole available tier. An explicitly unsupported PDF/image tier returns an error without substitution. Existing native documents always use Flash, even when a valid requested tier is not advertised upstream. Empty form strings and the literal string `"null"` are invalid tiers, not omissions.

`output_formats` is neither a JSON string nor a comma-separated value: use repeated form fields. `include_model_output` is removed and returns 400 when supplied.

## Upload and attachment responses

These are Bash/curl examples; in Windows PowerShell use `curl.exe` and adjust line continuations. The service calculates the size and hash from the actual uploaded file.

```bash
# response.body can be an attachment or JSON; check status and headers too.
curl --max-time 7200 -D response.headers -o response.body \
  http://127.0.0.1:8003/v1/file_parse \
  -F 'file=@report.pdf' \
  -F 'request_id=report-example' \
  -F 'tier=standard' \
  -F 'output_formats=default' -F 'output_formats=docx'
```

| Endpoint | HTTP | Body |
|---|---|---|
| POST, synchronous success | 200 | Raw attachment bytes, without a JSON/Base64 envelope. |
| POST, wait expires or `wait_timeout=0` | 202 | Complete StatusResponse. |
| `GET /v1/file_parse/{request_id}`, successful query | 200 | Complete StatusResponse, including known failure, cancellation, or unknown submission outcomes. |
| `GET /v1/file_parse/{request_id}/result`, successful download | 200 | The same attachment as synchronous POST. |

For an input named `report.pdf`:

| output_formats | Filename | Content-Type / content |
|---|---|---|
| `default` | `report.zip` | `application/zip`; Markdown, Middle JSON, Structured Content, images/. |
| `docx` | `report.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document`; embedded images. |
| `html` | `report.html` | `text/html; charset=utf-8`; embedded images. |
| `latex` | `report.zip` | `application/zip`; document.tex, images/. |
| Multiple formats | `report.zip` | A flat ZIP matching [Tasks outputs](tasks.md#outputs-and-downloads). |

Successful attachments include `Content-Disposition`, `X-Request-ID`, and `X-Task-ID`. A 202 includes `Location` (the status URL), `Retry-After: 2`, and `X-Request-ID`, plus `X-Task-ID` when confirmed. Status queries and error responses do not guarantee these headers; read recovery information from the JSON body.

## Status queries

All seven StatusResponse fields are always present. Nullable fields use null rather than the Tasks omission rule:

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Wrapper request ID. |
| `task_id` | string or null | Confirmed upstream task ID, or null before confirmation. |
| `status` | string | Single-file state, listed below. |
| `status_url` | string | Full query URL on the wrapper itself. |
| `result_url` | string or null | Wrapper result endpoint when completed, otherwise null. |
| `output_formats` | string[] | Normalized output format list. |
| `error` | object or null | Latest stored error or upstream file error, otherwise null. |

| State | Meaning |
|---|---|
| `prepared` | Source and recovery record saved; not submitted yet. |
| `submitting` | Creating the upstream task without a confirmed task_id yet. |
| `waiting-file` | Upstream input preparation; an absent upload does not establish readiness. |
| `pending`, `running` | Queued, parsing, or exporting upstream. |
| `completed` | Upstream artifact ready; downloading may still fail due to expiry or transfer errors. |
| `failed`, `canceled` | Known failure or upstream cancellation. |
| `submission_unknown` | Creation may have succeeded, but no valid task ID was received; creation is not replayed automatically. |

The wrapper processes one input and has no partial state. It exposes neither Tasks files/progress nor upstream signed URLs. Example 202 or GET status body:

```json
{
  "request_id": "report-example",
  "task_id": "job_example",
  "status": "running",
  "status_url": "http://127.0.0.1:8003/v1/file_parse/report-example",
  "result_url": null,
  "output_formats": ["default", "docx"],
  "error": null
}
```

```bash
curl http://127.0.0.1:8003/v1/file_parse/report-example
curl --fail-with-body -D result.headers -o report.zip \
  http://127.0.0.1:8003/v1/file_parse/report-example/result
```

Downloads return `409 result_not_ready` before completion, or potentially `409 task_not_created` without a confirmed upstream task. Retry GET after an interrupted download. For an expired signature, the wrapper queries the original task and tries up to three times; unrecoverable expired results do not trigger reparsing.

## Waiting, idempotency, and recovery

`wait_timeout` starts after the upstream file is ready and covers queuing, parsing, and export. Receiving the source, upstream upload/download/validation, and result download use separate budgets, so this is not an overall HTTP request timeout. Normal polling is every two seconds; connection, metadata-request, and transfer-idle timeouts are 10, 30, and 600 seconds. Zero wait can be used to obtain a status-query entrypoint first.

The service takes responsibility only after receiving and persisting the entire source. Earlier disconnects do not create a task. Afterwards, client disconnects, wait timeouts, and wrapper shutdown do not automatically cancel upstream work.

- The same request_id, file content, logical filename, and normalized parsing options reuse the original task; different input returns `409 request_id_conflict`. wait_timeout is not part of request identity.
- SQLite records are isolated by upstream address and authentication without storing the API key. Only one wrapper process may use a state directory. Restarts resume known tasks and still-valid pending uploads.
- Creation task_id/file_id values are persisted before fetching details. A failed first query or a restart does not repeat creation. Unknown creation outcomes retain submission_unknown instead of being blindly retried.
- Sources are removed after upstream readiness or terminal failure. Attachments are downloaded temporarily and removed after sending or interruption. Recovery records extend neither upstream upload links nor result retention; lost upstream tasks are not recreated.
- The new protocol namespace uses `tasks-v5-`; old namespaces are neither loaded, migrated, nor automatically deleted. See [4.1 migration](../reference/migration_4_1.md).

There is no separate wrapper cancellation endpoint. With a confirmed task_id and upstream authorization, POST `/v1/tasks/{task_id}/cancel` to the **configured upstream API base**, then query the wrapper to observe canceled. Do not use the wrapper base or confuse request_id with task_id. Without a saved ID or a received response, a caller may be unable to recover its request.

## Error responses

Errors directly generated by the wrapper contain `code: string`, `message: string`, and `upstream_status: integer|null`. Without a recovery record, only error is returned:

```json
{
  "error": {
    "code": "unsupported_media_type",
    "message": "Expected multipart/form-data",
    "upstream_status": null
  }
}
```

With a record, the complete seven-field StatusResponse is returned with error replaced by the current operation's error. For example, an unfinished download returns HTTP 409 while status remains running. Stored errors in successful queries may also contain http_status or preserve upstream code/message and optional param. Historical recovery records may retain old type or other fields; tolerate these extensions without treating them as supported by the new Tasks protocol.

| HTTP | Common code | Meaning |
|---|---|---|
| 400 | `invalid_request`, `upload_interrupted` | Invalid parameters, multiple/empty files, or disconnect before acceptance. |
| 404 | `request_not_found` | Record absent or collected. |
| 408 | `upload_receive_timeout` | Idle timeout while receiving the request body. |
| 409 | `request_id_conflict`, `result_not_ready`, `task_not_created`, or cancellation codes | Conflicting input, result not ready, no task yet, or upstream cancellation. |
| 410 | `upload_expired` | Upstream upload window expired. |
| 415 | `unsupported_media_type` | Not multipart/form-data. |
| 500 | `storage_error` | Local storage failure. |
| 502 | `submission_unknown`, `upstream_unreachable`, `upstream_protocol_error`, `invalid_result`, or upstream codes | Upstream request, transfer, parsing, or result-validation failure. |

Upstream authentication failures return wrapper HTTP 502 with `upstream_status=401`. Explicit tier rejection also returns 502, preserving upstream 400/503; these are failed, not submission_unknown. Other creation-stage 5xx, 408, network failures, or malformed responses may leave acceptance unknown and become submission_unknown. A shared-schema `503 quality_tier_unavailable` is a confirmed rejection. Neither case automatically replays creation; the same ID keeps the existing record.

## Python example client

The repository script requires httpx, prints the request ID before submission, and saves successful attachments atomically. On 202 it prints recovery information and exits with 124. Run from the repository root:

```bash
python scripts/http_file_parse_example.py report.pdf --output report.docx \
  --base-url http://127.0.0.1:8003 --output-format docx --request-id report-docx

python scripts/http_file_parse_example.py report.pdf --wait-timeout 0 \
  --base-url http://127.0.0.1:8003 --request-id report-background
```
