# Python SDK

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

### Batching and instance reuse

Reuse one `MinerUApiParser` instance for batch work instead of constructing a parser per file; each call opens and closes its own HTTP session, so there is nothing to release explicitly:

```python
from pathlib import Path
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(api_url="http://127.0.0.1:8000", tier="standard", include_images=True)
for pdf in Path("./documents").glob("*.pdf"):
    result = parser.parse(str(pdf))
    (Path("out") / f"{pdf.stem}.md").write_text(result.markdown(), encoding="utf-8")
```

Failures on the service side surface as per-file job errors rather than client exceptions; check the service logs and `GET /v1/usage` when throughput matters. See [Tiers and Runtimes](tiers.md) for choosing `tier`.

## HTTP API without the SDK

The same upload → job → poll → download cycle can be driven with plain HTTP calls. The [V1 HTTP API walkthrough](http_api.md) provides a complete curl example covering upload completion, terminal states (including `partial`), resuming after a client timeout, and artifact downloads.

The 4.0 V1 service does not provide legacy `/file_parse` or `/tasks` routes. See [migration](../reference/migration_4.md) for older clients and [output formats](../reference/output_files.md) for rendering and saving Python results.
