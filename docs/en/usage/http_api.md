# V1 HTTP API Walkthrough

The self-hosted V1 API is the same interface the Python SDK and WebUI use. This page explains the request cycle — create an upload, upload the bytes, submit a parse job, poll to a terminal state, download the outputs — and points to a tested example script. For the Python client, see [Python SDK](sdk_api.md).

Start a local service, or point the examples at any V1 deployment including the official cloud service:

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

## Tested example script

The complete cycle — including timeouts, response validation, bounded polling, exit codes, and credential isolation — is maintained as a single tested script in the repository:

> [`scripts/http_api_example.sh`](https://github.com/opendatalab/MinerU/blob/next/scripts/http_api_example.sh)

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

## Authentication and the byte upload

Requests to `/v1/*` carry `Authorization: Bearer $MINERU_API_KEY` when the service is started with `--api-key`; omit the header for anonymous local access.

The byte upload uses exactly the method, URL, and headers returned by `POST /v1/uploads`:

- A self-hosted service returns a same-origin URL such as `/v1/uploads/{id}/content`. Same-origin means scheme + host + effective port; it is checked by the client, not assumed. Same-origin uploads must carry the API authentication when the service requires it.
- The official API returns a pre-signed object-storage URL that carries its own authorization. Do **not** attach the MinerU API key to a different-origin URL — that would leak the credential. Headers returned by the service (`upload_headers`) are always kept as-is.

The example script implements the same rule as the Python SDK (`api_client._same_origin_upload_headers`): resolve relative URLs against the API base, compare scheme + host + effective port, attach the MinerU key only for same-origin uploads. Relative URLs resolve to same-origin; the same host on a different port counts as a different origin.

Downloads go through `GET /v1/files/{file_id}/content`, which may answer with a `302` redirect. The example follows redirects without `--location-trusted`, so credentials are never re-sent to a different origin (including the same host on a different port; requires curl >= 7.83.0).

## Request-cycle details the script deliberately covers

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

The V1 service does not provide legacy `/file_parse` or `/tasks` routes; see [migration](../reference/migration_4.md) for older clients.
