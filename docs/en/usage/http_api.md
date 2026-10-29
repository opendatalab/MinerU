# V1 HTTP API

In 4.1, Tasks is the single asynchronous parsing API. API Server, Router, the SDK and WebUI use a JSON `files` array, upload only missing files, poll task metadata and download each file's result. See the [Tasks contract](tasks.md) and [Python SDK](sdk_api.md).

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
OUTPUT_FORMATS=default,docx MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_api_example.sh a.pdf
```

## Create and upload

Send application/json to POST `/v1/tasks`. Each files entry requires file_name and, for uploads, bytes; sha256sum and page_range are optional. URL entries may omit bytes. Requests must not include client_file_id or file_id, even as null. The server generates a unique file_id for every entry and returns entries in input order. This ID remains stable through upload, query, cancellation and result download. Options are tier, ocr_mode, page_range and output_formats.

The response contains task_id and an upload object on each file that still needs uploading. files[].upload is omitted otherwise; its fields are url, headers and expires_at. Uploads always use PUT; upload_method is not returned. Follow each returned PUT URL and its headers; no completion request is needed. Each ready file enters pending independently. waiting-file covers upload, server-side download and validation; only a present upload object asks the client to send bytes. Input preparation consumes no parsing slot.

Loopback requests follow the same protocol. The SDK computes SHA-256 before submission; completed, unexpired cached inputs are not uploaded again. Concurrent first uploads may transfer duplicate bytes, but source storage is deduplicated. Default retention is 24 hours with collection every 5 minutes; active task and transfer references protect their bytes.

Upload links expire one hour after task creation. Queries do not renew them. Uploads accepted before the deadline may finish. After an uncertain upload, query first. pending/running/completed need no upload; waiting-file with no upload field requires polling only. Retry only when an upload target is available, within the transfer retry budget. A successful PUT is not repeated while waiting for confirmation.

## Status and results

Poll GET `/v1/tasks/{task_id}` using the configured API base, preserving its path prefix and URL-encoding the task ID. Bind the creation receipt to inputs in order, then match files by file_id. Parent states are waiting-file, pending, running, completed, partial, failed and canceled. Files use the same states except partial; upload_status has been removed. Each completed file has one files[].result_url; there is no top-level result_url or combined batch ZIP. Completed files can be downloaded while other files are still pending.

Outputs are default (Markdown, Middle JSON, Structured Content and images in a ZIP), docx, html, or latex (TEX and images in a ZIP). Multiple formats share one flat ZIP. DOCX and HTML embed images. ModelJson stays inside parsing and postprocessing; API, SDK, CLI and Gradio results do not include model output. The removed include_model_output option is rejected with HTTP 400, including false.

Task and file listing endpoints have been removed. POST `/v1/tasks/{task_id}/cancel` cancels unfinished files. Files metadata and output download remain available; task-record and artifact deletion endpoints are not provided. Before completion, files[].parse is omitted. Successful files report their actual tier, duration_ms and parser_version; there is no response-level tier. Responses no longer echo per-file sha256sum, page_range, private source IDs or duplicate name fields. Request parameters remain available.

Use returned result URLs directly. Local servers and Router return Files URLs; SaaS may return signed storage URLs. Attach the API key only to same-origin API requests and reevaluate credentials at every redirect. Query the task again when a result signature expires.

## Client and migration

Both shell examples delegate to the standard-library Python Tasks client. RESULT_DIR selects the per-file destination. OUTPUT_FORMATS defaults to default. SEND_HASH=0 disables the optional preflight hash. Resume with TASK_ID and the same input order.

Creation is not automatically retried. MAX_POLLS/POLL_INTERVAL bound polling; CONNECT_TIMEOUT/REQUEST_TIMEOUT bound individual requests. Exit codes are 0 (success), 2 (partial success), 1 (failure or cancellation) and 124 (poll budget exhausted). Completed downloads remain on disk.

Tasks and Files JSON use `file_name`; creation accepts `files`. Old `manifest`/`filename` fields and mixed old/new fields are rejected with 400. The standalone upload and Jobs APIs have been removed. Coordinate server, Router and SDK upgrades; see [4.1 migration](../reference/migration_4_1.md). The separate [file-parse wrapper](file_parse.md) provides recoverable synchronous single-file calls.

POST returns a creation receipt, GET returns task details, POST cancel returns every file’s final state and successful result links, and local PUT returns task_id/file_id/status. Tasks success responses omit null fields but preserve zero values and empty collections. POST cancel cancels unfinished files while preserving completed results.
