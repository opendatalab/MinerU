# Python SDK and V1 API

## Local Python SDK

`mineru.parser` exports stateless `parse`, `parse_async`, `MinerUParser`, and a shared `ParseResult`; it does not create a document library. PDF page ranges default to all pages; `page_range="1-3"` uses one-based page numbers.

```python
from pathlib import Path
from mineru.parser import parse, ParseResult

result = parse("document.pdf", tier="flash", ocr_mode="txt", page_range="1-3")
Path("document.md").write_text(result.markdown(), encoding="utf-8")
Path("document.json").write_text(result.to_json(), encoding="utf-8")
restored = ParseResult.from_json(result.to_json())
assert restored.to_dict() == result.to_dict()
```

For native documents, use `parse("report.docx", tier="flash")` without a PDF page range. For higher-quality PDF/image parsing, select `tier="standard"` or `tier="advanced"` and prepare the relevant runtime and models.

The asynchronous entrypoint accepts the same parsing options:

```python
import asyncio
from mineru.parser import parse_async

result = asyncio.run(parse_async("report.docx", tier="flash"))
print(result.markdown())
```

Inside an existing event loop, use `await parse_async(...)`. An async interface does not imply identical execution mechanisms across engines.

## Connect to a self-hosted V1 API

Start the service in its inference environment:

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

The Python client handles discovery, file submission, job polling, and result downloads:

```python
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(
    api_url="http://127.0.0.1:8000",
    api_key="",
    tier="standard",
    include_images=True,
)
result = parser.parse("document.pdf", page_range="1-3")
print(result.markdown())
```

`api_url` is the service root before `/v1`. Use `MINERU_API_KEY` for a configured key instead of embedding credentials in source. Connecting to another machine may upload files to that service. The WebUI can use the same address:

```bash
mineru-kit webui --api-url http://127.0.0.1:8000
```

## HTTP workflow

Discover actual service capabilities first:

```bash
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/tiers
```

1. Create an upload with `POST /v1/uploads`. Follow its returned URL, HTTP method, and headers, then call `/v1/uploads/{id}/complete` when required to obtain `file.id`. A completed deduplicated upload can return the file reference directly.
2. Submit a job with `POST /v1/parse/jobs`, using a body such as the following.
3. Poll `GET /v1/parse/jobs/{job_id}`. `completed`, `partial`, `failed`, and `canceled` are all terminal. Inspect per-file errors; partial success is not full completion.
4. Download outputs through `GET /v1/files/{file_id}/content` using the artifact references in the job response.

```json
{
  "files": [{"source": {"type": "file_id", "file_id": "file-id-from-upload"}, "page_range": "1-3"}],
  "tier": "standard",
  "ocr_mode": "auto",
  "output_formats": ["markdown", "middle_json", "structured_content", "zip"]
}
```

Omit `page_range` for non-PDF files. The current self-hosted server provides the four output types above. Consult the server capabilities and OpenAPI at `/docs`; renderer formats do not imply API support.

The 4.0 V1 service does not provide legacy `/file_parse` or `/tasks` routes. See [migration](../reference/migration_4.md) for older clients and [output formats](../reference/output_files.md) for rendering and saving Python results.
