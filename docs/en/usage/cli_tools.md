# Command Line Tools

MinerU exposes two primary command trees. `mineru` is the document-library client for interactive and agent workflows; `mineru-kit` contains stateless parsing, service, model, router, and WebUI tools.

## Document-library CLI

Use `mineru --help` to list all document-library commands. The most common parsing flow is:

```bash
mineru parse report.pdf --pages all -o report.md
```

When `-o/--output` is omitted, rendered content is written to stdout. PDF input defaults to the first 10 pages; pass `--pages all` for the complete document. The document library manages ingestion, caching, background parsing, reading, search, and cleanup.

Manage the local document-library service with:

```bash
mineru server start
mineru server status
mineru server stop
```

Run `mineru <command> --help` for the authoritative options of each command.

## Stateless and service tools

Use `mineru-kit --help` to list the available tools.

### Batch parsing

```bash
mineru-kit parse report.pdf -o report.md --tier standard
mineru-kit parse ./documents -o ./output --format zip
```

`mineru-kit parse` does not use the document-library database or cache. It supports local parsing and explicit V1 remote parsing; see `mineru-kit parse --help` for tier, backend, page-range, and output options.

### V1 API server

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI documentation. The supported API is `/v1/*`; the removed legacy `/file_parse` and `/tasks` routes are not available.

### Gradio WebUI

```bash
mineru-kit gradio --server-name 127.0.0.1 --server-port 7860
```

Without `--api-url`, Gradio manages a loopback `mineru-kit api-server`. With `--api-url`, it connects only to that V1 service. `mineru-gradio` is retained as a command-name alias and accepts the same modern options; it does not restore legacy Gradio options or HTTP routes.

### Router and VLM server

```bash
mineru-kit router --host 127.0.0.1 --port 8002 --local-gpus auto
mineru-kit vlm-server --engine auto --port 30000
```

`mineru-router` remains a command-name alias for `mineru-kit router`. Router exposes only the V1 API and accepts only its documented worker options.

## Environment variables

- `MINERU_HOME`: root for MinerU configuration, cache, and document-library state.
- `MINERU_CONFIG`: explicit `config.yaml` path.
- `MINERU_MODEL_SOURCE`: model source, such as `huggingface`, `modelscope`, or `local`.
- `MINERU_API_URL` / `MINERU_API_KEY`: default V1 API URL and bearer key for API clients.
- `MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS`: startup timeout for the Gradio-managed local V1 server; default `300` seconds.
- `MINERU_API_ENABLE_FASTAPI_DOCS`: enable `/docs`, `/openapi.json`, and `/redoc` on the V1 API server; default `true`.
- `MINERU_PDF_RENDER_TIMEOUT` / `MINERU_PDF_RENDER_THREADS`: PDF rendering timeout and worker count.
- `MINERU_PROCESSING_WINDOW_SIZE`: processing window size used for large documents.
- `MINERU_INTRA_OP_NUM_THREADS` / `MINERU_INTER_OP_NUM_THREADS`: ONNX operator thread settings.

Prefer each command's `--help` output and [model source documentation](./model_source.md) for current defaults.

## PDF page selection

Use `--pages "1-5,8,r3-r1"`: page numbers start at 1, ranges include both endpoints,
and `r1` means the last page. Use `all` for every page. Results are sorted and deduplicated;
partially out-of-bounds ranges select their valid intersection. Reversed or empty selections
fail with `page_range_invalid`. Without `--pages`, `mineru parse` starts with the first 10 pages;
`mineru-kit parse`, Python and Gradio select all pages. New requests use the current syntax. Historical positive result ranges using ASCII `~`
remain readable without rebuilding Doclib caches; result responses and new cache entries use `-`.
Fullwidth `～` and negative page-number notation are not supported.
See [page-range syntax and historical result compatibility](../../next/page-ranges.md).
