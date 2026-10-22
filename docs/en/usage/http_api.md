# V1 HTTP API

Tasks is the single asynchronous parsing API. API Server, Router, the SDK and WebUI use a JSON manifest, upload only missing files, poll task metadata and download each file's result. See the [Tasks contract](../../next/api/tasks.md) and [Python SDK](sdk_api.md).

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_task_example.sh a.pdf b.pdf
OUTPUT_FORMATS=default,docx MINERU_API_URL=http://127.0.0.1:8000 bash scripts/http_api_example.sh a.pdf
```

## Create and upload

Send application/json to POST `/v1/tasks`. Each manifest entry contains a unique client_file_id, filename, bytes for uploaded files, and optional sha256sum/page_range. URL entries may omit bytes. Options are tier, ocr_mode, page_range, output_formats and include_model_output.

The response contains task_id, status_url and upload links for missing files. Follow each returned PUT URL and its headers; no completion request is needed. Each ready file queues independently. Waiting for uploads consumes no parsing slot.

Loopback requests follow the same protocol. The SDK computes SHA-256 before submission; completed, unexpired cached inputs are not uploaded again. Concurrent first uploads may transfer duplicate bytes, but source storage is deduplicated. Default retention is 24 hours with collection every 5 minutes; active task and transfer references protect their bytes.

Upload links expire one hour after task creation. Queries do not renew them. Uploads accepted before the deadline may finish. After an uncertain upload, query first and skip ready files; retry pending uploads from the beginning while their links remain valid.

## Status and results

Poll status_url and match files by client_file_id. Parent states are waiting_upload, queued, running, completed, partial, failed and canceled. Each completed file has one files[].result_url; there is no top-level result_url or combined batch ZIP. Completed files can be downloaded while other files are still pending.

Outputs are default (Markdown, Middle JSON, Structured Content and images in a ZIP), docx, html, or latex (TEX and images in a ZIP). Multiple formats share one flat ZIP. DOCX and HTML embed images. include_model_output defaults to false; enabling it requires default and adds model_output.json when raw output exists.

GET `/v1/tasks` lists parent tasks with status filters and cursor pagination. DELETE `/v1/tasks/{task_id}` cancels unfinished files. Files metadata, listing, output download and deletion remain available.

Use returned result URLs directly. Local servers and Router return Files URLs; SaaS may return signed storage URLs. Attach the API key only to same-origin API requests and reevaluate credentials at every redirect. Query the task again when a result signature expires.

## Client and migration

Both shell examples delegate to the standard-library Python Tasks client. RESULT_DIR selects the per-file destination. OUTPUT_FORMATS defaults to default; INCLUDE_MODEL_OUTPUT=1 requests raw model output. SEND_HASH=0 disables the optional preflight hash. Resume with TASK_ID and the same input order.

Creation is not automatically retried. MAX_POLLS/POLL_INTERVAL bound polling; CONNECT_TIMEOUT/REQUEST_TIMEOUT bound individual requests. Exit codes are 0 (success), 2 (partial success), 1 (failure or cancellation) and 124 (poll budget exhausted). Completed downloads remain on disk.

The standalone upload and Jobs APIs have been removed. Coordinate server, Router and SDK upgrades; see [migration notes](../../next/api/changes.md). The separate [file_parse wrapper](../../next/api/file-parse.md) provides recoverable synchronous single-file calls.
