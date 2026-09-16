# V1 HTTP API Walkthrough

The self-hosted V1 API is the same interface the Python SDK and WebUI use. This page walks through one complete request cycle with plain HTTP calls: create an upload, upload the bytes, submit a parse job, poll to a terminal state, and download the outputs. For the Python client, see [Python SDK](sdk_api.md).

Start a local service, or point the examples at any V1 deployment including the official cloud service:

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

Requests to `/v1/*` carry `Authorization: Bearer $MINERU_API_KEY` when the service is started with `--api-key`; omit the header for anonymous local access. The byte upload in step 2 does **not** go through `/v1`: it uses exactly the URL, method, and headers returned in step 1.

## Complete example (curl + jq)

```bash
BASE=http://127.0.0.1:8000
AUTH="Authorization: Bearer $MINERU_API_KEY"    # omit when the service allows anonymous access
FILE=document.pdf
SIZE=$(stat -c%s "$FILE")                       # macOS: stat -f%z "$FILE"

# 1. Create an upload session
RESP=$(curl -s -X POST "$BASE/v1/uploads" -H "$AUTH" -H "Content-Type: application/json" -d "{
  \"filename\": \"$FILE\", \"bytes\": $SIZE, \"mime_type\": \"application/pdf\", \"purpose\": \"parse\"
}")

# 2. Upload the bytes using the returned method/url/headers (pending uploads only)
if [ "$(echo "$RESP" | jq -r .status)" = "pending" ]; then
  curl -s -X PUT "$(echo "$RESP" | jq -r .upload_url)" \
       -H "Content-Type: application/pdf" --data-binary "@$FILE"
  # 3. Complete the upload to obtain the file id
  RESP=$(curl -s -X POST "$BASE/v1/uploads/$(echo "$RESP" | jq -r .id)/complete" -H "$AUTH")
fi
FILE_ID=$(echo "$RESP" | jq -r .file.id)

# 4. Submit a parse job
JOB=$(curl -s -X POST "$BASE/v1/parse/jobs" -H "$AUTH" -H "Content-Type: application/json" -d "{
  \"files\": [{\"source\": {\"type\": \"file_id\", \"file_id\": \"$FILE_ID\"}, \"page_range\": \"1-10\"}],
  \"tier\": \"standard\",
  \"output_formats\": [\"markdown\", \"zip\"]
}")
JOB_ID=$(echo "$JOB" | jq -r .job_id)

# 5. Poll until a terminal state (bounded loop; add your own attempt limit)
while : ; do
  JOB=$(curl -s "$BASE/v1/parse/jobs/$JOB_ID" -H "$AUTH")
  STATUS=$(echo "$JOB" | jq -r .status)
  case "$STATUS" in completed|partial|failed|canceled) break ;; esac
  sleep 3
done
echo "$JOB" | jq '{status, progress, files: [.files[] | {name, status, error}]}'

# 6. Download the artifacts of each completed file
for FID in $(echo "$JOB" | jq -r ".files[].output_files.markdown.file_id? // empty"); do
  curl -s "$BASE/v1/files/$FID/content" -H "$AUTH" -o document.md
done
```

Details the example deliberately shows:

- **Upload lifecycle.** Pass `sha256sum` when creating the upload for integrity checking and instant reuse of an already-known file; in that case the response is `completed` and embeds the `file` object directly, so steps 2–3 are skipped.
- **Byte upload.** Always use the returned `upload_method`, `upload_url`, and `upload_headers` verbatim. A self-hosted server returns a same-origin loopback URL; the official API returns a pre-signed object-storage URL that carries its own authorization — do not add the Bearer header there.
- **Terminal states.** `completed`, `partial`, `failed`, and `canceled` are all terminal. `partial` means some files succeeded: read `output_files` from the per-file entries whose `status` is `completed`, and inspect `error` on the rest; partial success is not full completion.
- **Client timeout is not job failure.** If your polling budget runs out while the job is still `queued` or `running`, resume polling `GET /v1/parse/jobs/{job_id}` later; the job is not canceled because your client stopped asking.
- **Outputs.** Every artifact is a `purpose:"parse_output"` file; content always comes from `GET /v1/files/{file_id}/content` using the ids under `files[].output_files`. Treat `GET /v1/health` (`features.sources`, `features.output_formats`) as the source of truth for what the current deployment supports; renderer formats do not imply API support.

## Production notes

- Bind `--host` consciously. The default `127.0.0.1` keeps the service on loopback; `0.0.0.0` exposes it to the network, so combine that with `--api-key` or an authenticating reverse proxy.
- Uploads and outputs are stored on the service host; `--upload-dir` controls the upload directory. Plan disk usage and retention for high-throughput deployments — output files carry `expires_at` where the deployment configures it.
- `--concurrency` caps concurrent parse jobs; queue growth beyond it shows up as longer `queued` times, not failures.
- `GET /v1/usage` reports consumption. Use `/v1/health` and the OpenAPI docs at `/docs` (when enabled) to separate service-side failures from client errors; check server logs before retrying a `failed` job.

The V1 service does not provide legacy `/file_parse` or `/tasks` routes; see [migration](../reference/migration_4.md) for older clients.
