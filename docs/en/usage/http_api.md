# V1 HTTP API

HTTP clients can use the tasks form-manifest and download-link protocol. This page first covers that workflow, then the existing Uploads/Jobs resource workflow used by the Python SDK and WebUI. For the Python client, see [Python SDK](sdk_api.md).

Start a local service; for SaaS, use a deployment implementing the same tasks contract:

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

## Tasks: manifests, per-file uploads, and ZIP links

API servers, Routers, and SaaS deployments implementing this contract use the same client workflow: submit a manifest, upload missing files, poll status, and download the returned `result_url`.
SaaS can return an OSS URL; a local service returns its Files URL. Clients use the returned links and do not require SaaS to implement the local Files API.
See the [complete tasks contract](../../next/api/tasks.md).

```bash
# One or more files: preflight first and upload only cache misses.
MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# Hashes are optional.
SEND_HASH=0 MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# Continue the existing task using the original file list and order.
TASK_ID=job_... MINERU_API_URL=http://127.0.0.1:8000 \
  bash scripts/http_task_example.sh a.pdf b.pdf

# Local multipart accepts one file and still returns an asynchronous task.
curl --fail-with-body 'http://127.0.0.1:8000/v1/tasks' \
  -F 'tier=flash' -F 'ocr_mode=txt' -F 'files=@document.pdf'
```

All `POST /v1/tasks` parameters belong in the form body. `manifest` is a text field containing a JSON array: each item has a unique `client_file_id`, `filename`, actual `bytes`, and optional `sha256sum` and per-file `page_range`.
Multi-file requests submit metadata only. Single-file direct uploads may include a one-item manifest to declare a hash.
Parsing options remain `tier`, `ocr_mode`, `page_range`, and repeated `output_formats`; a final ZIP is always generated.
POST query parameters return `400`; JSON and raw-byte task creation return `415`. Existing Uploads/Jobs JSON requests remain unchanged.

Creation immediately returns a stable `task_id` and `status_url`. `uploads` contains only missing files, with their upload method, URL, headers, and expiration.
Parsing starts automatically after every file passes validation; no client completion request is needed. Without a supplied hash, the service receives the file and still computes a hash for storage deduplication.
Each fully uploaded file becomes `ready` independently. If another upload is interrupted, retry only that file on the same task. This is file-level retry, not resumable chunk upload within one file.

Poll only `status_url`. States are `waiting_upload/queued/running/completed/partial/failed/canceled`.
`completed/partial` and `result_url` are published only after the ZIP is saved. Status never includes inline document contents; SaaS clients download directly from object storage.
Batch ZIPs contain numbered directories and a per-file manifest. Packaging failures produce a failed task, not an unusable download link.

The complete client is [`scripts/http_task_example.sh`](https://github.com/opendatalab/MinerU/blob/master/scripts/http_task_example.sh) and its adjacent Python file, requiring only Python 3.10+ standard library.
`UPLOAD_MODE=direct` accepts one file. `RESULT_FILE` defaults to `result.zip`; `MAX_POLLS/POLL_INTERVAL` control polling, and `CONNECT_TIMEOUT/REQUEST_TIMEOUT` control transfer budgets.
Each file gets at most three upload attempts. Creation is never automatically retried. After an uncertain upload response, query status before retransmitting; refresh expired links by querying the original task.
The API Key is attached only to same-origin requests, never to OSS, external downloads, or cross-origin redirects.
Exit codes are `0` for success, `2` for partial success with a saved ZIP, `1` for failure or cancellation, and `124` when the polling budget is exhausted without canceling the task.

Local services impose no fixed uploaded-file size or PDF page-count limit; the limit remains 100 files per task. Waiting uploads consume no parsing slot or Router worker reservation.
Router selects by load and reserves before upstream submission; a cache hit cannot pin work to a busy worker. One batch stays on one worker, which produces the ZIP; Router maps resources and streams downloads.

## Existing Uploads / Jobs resource example

For independent Upload, File, and Job resources, continue using the following script. Its protocol and exit codes remain unchanged and are maintained separately from the tasks client above:

> [`scripts/http_api_example.sh`](https://github.com/opendatalab/MinerU/blob/master/scripts/http_api_example.sh)

```bash
export MINERU_API_URL=http://127.0.0.1:8000
export MINERU_API_KEY=secret-key        # omit for anonymous local access
./scripts/http_api_example.sh document.pdf
```

The script's behavior is verified by `tests/unittest/test_http_api_example_script.py` against a real Flash service and a scripted mock, without model downloads. Exit codes are stable for automation:

| Code | Meaning |
| --- | --- |
| `0` | Job completed; every requested artifact was saved |
| `1` | Script, transport, protocol, download, or local write failure |
| `2` | Job partial: artifacts of completed files were saved first |
| `3` | Job failed |
| `4` | Job canceled |
| `124` | Polling budget exhausted; `job_id` printed so polling can resume |

## Resource API authentication and byte uploads

Requests to `/v1/*` carry `Authorization: Bearer $MINERU_API_KEY` when the service is started with `--api-key`; omit the header for anonymous local access.

The byte upload uses exactly the method, URL, and headers returned by `POST /v1/uploads`:

- A self-hosted service returns a same-origin URL such as `/v1/uploads/{id}/content`. Same-origin means scheme + host + effective port; it is checked by the client, not assumed. Same-origin uploads must carry the API authentication when the service requires it.
- The official API returns a pre-signed object-storage URL that carries its own authorization. Do **not** attach the MinerU API key to a different-origin URL — that would leak the credential. Headers returned by the service (`upload_headers`) are always kept as-is.

The example script implements the same rule as the Python SDK (`api_client._same_origin_upload_headers`): resolve relative URLs against the API base, compare scheme + host + effective port, attach the MinerU key only for same-origin uploads. Relative URLs resolve to same-origin; the same host on a different port counts as a different origin.

Downloads go through `GET /v1/files/{file_id}/content`, which may answer with a `302` redirect. The example follows redirects without `--location-trusted`, so credentials are never re-sent to a different origin (including the same host on a different port; requires curl >= 7.83.0).

## Resource-script request-cycle details

- **Upload lifecycle.** Pass `sha256sum` when creating the upload for integrity checking and instant reuse of an already-known file; in that case the response is `completed` and embeds the `file` object directly, so byte upload and completion are skipped.
- **A 200 response is not automatically valid.** The script validates JSON structure at every step: IDs must be non-empty strings and statuses must belong to the supported sets; anything else fails immediately instead of entering the polling loop with a `null` ID.
- **Bounded polling.** `MAX_POLLS` limits the number of polling requests (each also spends up to the request timeout and the poll interval); when the budget is exhausted the script exits `124` and prints the `job_id` — the job was not canceled and polling can simply resume.
- **Terminal states.** `completed`, `partial`, `failed`, and `canceled` are all terminal. `partial` means some files succeeded: the script saves the artifacts of the completed files, prints per-file results including errors, and only then exits `2`. `failed` exits `3`, `canceled` exits `4`.
- **Atomic downloads.** Artifacts are written to `*.part` first and moved into place on success; a `completed` job with a failing or missing artifact download does not exit `0`.
- **Outputs.** Every artifact is a `purpose:"parse_output"` file; content always comes from `GET /v1/files/{file_id}/content` using the ids under `files[].output_files`. Treat `GET /v1/health` (`features.sources`, `features.output_formats`) as the source of truth for what the current deployment supports; renderer formats do not imply API support.

## Production notes

- Bind `--host` consciously. The default `127.0.0.1` keeps the service on loopback; `0.0.0.0` exposes it to the network, so combine that with `--api-key` or an authenticating reverse proxy.
- Storage and restart boundary: file bytes live under `--upload-dir` (a temporary directory that is removed on clean shutdown when not configured), but the upload/file/job resource index is in-process state. Restarting the service does **not** preserve old `upload_id`/`file_id`/job ids, even when the directory is kept. This API is not a recoverable persistent job service; use the document library for persistent reading, indexing, and caching — it is not a drop-in persistent replacement for this upload/job interface.
- `--concurrency` caps concurrent parse jobs; queue growth beyond it shows up as longer `queued` times, not failures.
- `GET /v1/usage` reports consumption. Use `/v1/health` and the OpenAPI docs at `/docs` (when enabled) to separate service-side failures from client errors; check server logs before retrying a `failed` job.

The convenience workflow uses `/v1/tasks` and its returned links, with no synchronous-wait or inline-result route. Legacy root routes remain unavailable; see [migration](../reference/migration_4.md) for older clients.
