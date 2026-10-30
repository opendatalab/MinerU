# Migrating from 4.0.x to 4.1

This guide covers API and runtime changes since 4.0.12. HTTP paths still use `/v1`, but this is a breaking protocol update: old clients and new servers cannot be mixed directly.

## Python and installation

Supported Python versions are `>=3.11,<3.15`; Python 3.10 is no longer supported. Create a new environment, preferably with Python 3.12, reinstall the required dependencies, and update interpreter paths in service startup configuration.

Stop the old services and drain their tasks before running:

```bash
uv venv --python 3.12 .mineru-4.1
# Linux/macOS
source .mineru-4.1/bin/activate
# Windows PowerShell: .\.mineru-4.1\Scripts\Activate.ps1
uv pip install -U "mineru>=4.1,<5"
mineru version --json
```

Use the [torch/full extras](../quick_start/extension_modules.md) when you need Torch or higher-throughput VLM serving. Optional engines retain their own Python wheel, driver, and platform constraints.

## Service tiers and native documents

API startup uses one ceiling: `--tier auto|flash|basic|standard|advanced`. `--tiers` is removed; `--no-flash` and `--no-advanced` are not supported. The ceiling includes itself and lower tiers; `--tier standard` no longer also enables Advanced.

```bash
mineru-kit api-server --tier auto
mineru-kit hardware detect --json
mineru-kit hardware info --json
```

- `auto` (also the startup default) recommends a ceiling from the effective small-model backend and available devices. ONNX performs applicable selection and calibration on Windows/Linux; the first run may prepare or download the OCR recognition model without `--preload-models`. Torch checks device availability without ONNX calibration.
- `hardware detect` uses the same recommendation logic. `hardware info` reads hardware, runtime, and existing calibration information without benchmarking, downloading weights, or writing configuration.
- `/v1/tiers` describes PDF/image capabilities only. Every normally available compatible service supports existing native document formats. Office, HTML, and other native documents use Flash even when Flash is absent from `/v1/tiers`.
- An omitted or JSON `null` request tier selects the first supported PDF/image tier in the order **Standard → Basic → Flash → Advanced**; Advanced is selected automatically only when it is the sole available tier. An explicitly unsupported PDF/image tier returns 400 without changing tiers. Native-only requests ignore a valid requested tier. Mixed batches run each file at its effective tier, reported in `files[].parse.tier`.
- Python `create_app(tiers=...)` retains exact PDF/image tier sets. WebUI and Doclib managed services follow the new startup ceiling; Gradio no longer starts a supplementary local native parser just because a remote service does not advertise Flash.

See [Tiers and runtimes](../usage/tiers.md) for device configuration and support boundaries. The service ceiling and the default tier for an individual request are separate settings.

## HTTP API replacement

All standalone `/v1/uploads` routes and `/v1/parse/jobs` creation, listing, query, and cancellation routes are removed and return 404. There are no aliases or automatic fallbacks.

| Previous contract | 4.1 contract |
|---|---|
| Create Upload → PUT → complete → create Job | `POST /v1/tasks` with a JSON manifest; PUT only missing files, which queue automatically after validation. |
| Query or cancel a Job | `GET /v1/tasks/{task_id}` and `POST /v1/tasks/{task_id}/cancel`. |
| Form or multipart Tasks creation | `application/json`; use the standalone [file-parse service](../usage/file_parse.md) for single-file multipart requests. |
| `manifest`, `filename` | `files` array (1–100 entries), `file_name`; old or mixed fields return 400. |
| `client_file_id`, caller-supplied `file_id` | The server generates a public `file_id` for every input, returning files in input order. Requests must omit both fields, including `null` values. |
| `local/inline/file_id` sources | Uploads or HTTP(S) URLs; loopback clients also use hash preflight and upload cache misses. |
| `uploads/pending_uploads` | Per-file `files[].upload` with `url/headers/expires_at`. Upload paths may still contain `/uploads/`; these are not the old standalone API. |
| `upload_method`, upload completion request | PUT raw bytes to the returned URL with its headers; no completion call. |
| `upload_status`, `waiting_upload/queued` | Unified `files[].status` with `waiting-file/pending`; poll when a waiting file has no `upload`. |
| Tasks `status_url` | Build the query route from the configured API base and URL-encoded `task_id`, preserving any proxy prefix. |
| Shared task response | Separate `CreateTaskResponse`, `TaskDetailResponse`, `CancelTaskResponse`, and `UploadFileResponse` for creation, detail, cancellation, and local upload. |
| Success-response `null` placeholders | Tasks recursively omit absent values, preserving zeros and empty collections. file-parse status and HTTP errors keep their own rules. |
| Public error `type`, nullable `code` | Required nonempty `code`, `message`, and optional `param`. type is neither emitted nor accepted; absent param is omitted. HTTP failures and task errors share this structure. |
| Top-level `tier`, `files[].parse.model_used` | Successful files report actual `tier` and available `duration_ms/parser_version` in `files[].parse`. |
| Response `sha256sum/page_range/name`, source-cache IDs | No longer echoed; request hashes and page ranges remain supported. The public `file_id` stays stable through result delivery. |
| `files[].output_files`, combined batch ZIP | One `files[].result_url` per successful input; no parent-level result URL. |
| `markdown/middle_json/structured_content/zip` output options | `default` returns those general results and images in a ZIP; `docx/html/latex` are also available and can be combined. |
| `include_model_output`, `model_output.json` | The option returns 400. ModelJson remains internal and is absent from API, SDK, CLI, and Gradio results. |
| Health `features.output_formats`, `task_model_output` | Use `features.task_output_formats`, with `upload/url` sources; `task_model_output` is removed. |
| Task/file lists, DELETE cancellation | No listing or public deletion endpoints; with valid authentication, `GET /v1/tasks` returns 405 and `GET /v1/files` returns 404. Cancellation uses POST. |

`--allow-local-source` and `--max-inline-bytes` are also removed. Files lookup/download by ID, Health, Tiers, Models, and Usage remain available. Files lookup or download for a reserved `file_id` returns 404 until its result exists.

See the [Tasks API](../usage/tasks.md) for complete fields, states, upload expiry, and errors. The SDK keeps `MinerUApiParser` methods and return types, uses `ApiTaskStatus`, and retains the Python alias `ApiJobStatus`; that alias does not restore the old HTTP protocol.

The public error update also requires coordinated worker, Router, SDK, file-parse, and SaaS upgrades. Clients handle failures by HTTP status and code. Internal CLI/Doclib error categories and historical file-parse recovery records are not rewritten by this change.

## file-parse and recovery boundaries

`mineru-kit file-parse-server` is a standalone synchronous single-file wrapper over Tasks, connecting to a local API, Router, or compatible SaaS. The main API Server and Router do not expose `/v1/file_parse`.

A 202 after the wait ends does not cancel processing. Save `request_id/status_url` to query or download later. The wrapper resumes known upstream tasks; it does not automatically resubmit an unknown creation outcome or recover tasks lost after an upstream restart or expiry.

New recovery namespaces use the `tasks-v5-` prefix. Existing `tasks-v4-`, `tasks-v3-`, `tasks-v2-`, and unprefixed directories are retained but neither loaded nor migrated. Request IDs from those namespaces do not resume; idempotency and wrapper-restart recovery continue within the new namespace. See the [file-parse API](../usage/file_parse.md) for status and attachment examples.

## Upgrade order and verification

1. Stop accepting old requests, drain in-flight tasks, and save required results. API/Router task and cache indexes are in memory and are not guaranteed to survive restarts.
2. Prepare Python 3.11–3.14 environments. Upgrade workers first, then Router, SDK, WebUI, the file-parse wrapper, and applications. SaaS deployments must align with the same contract; do not rely on old-worker fallback.
3. Update request fields, query/cancel routes, and per-file downloads. Save task and file IDs from creation; do not blindly retry creation requests.
4. Check a mixed PDF/native-document batch, tier selection, cached-upload reuse, and per-file results. Exercise failures, cancellation, upload expiry, and polling after a synchronous wait timeout. Verify Markdown, structured JSON, and assets before reopening traffic.

Users migrating from earlier versions should also read [3.x → 4.0 migration](migration_4.md).
