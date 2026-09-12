# Quick Start

This guide targets the **MinerU 4.0 stable release**. Existing AMD and vendor accelerator adaptations remain on `<4`; use the [legacy platform guides](../usage/compatibility.md).

## Install MinerU

The MinerU package supports Python `>=3.10,<3.15`. Python 3.12 is a practical starting point for a new environment. Optional engines such as vLLM, LMDeploy, and Torch have additional wheel, OS, and driver constraints; the package's Python range does not guarantee every engine supports that entire range.

```bash
uv venv --python 3.12 .venv
```

Activate on Linux / macOS:

```bash
source .venv/bin/activate
```

Activate in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the base package:

```bash
uv pip install -U "mineru>=4.0,<5"
```

Alternatively, run `python -m pip install -U "mineru>=4.0,<5"` in the activated environment. A package mirror can be selected with the installer's `-i` option.

The base package includes the WebUI, ONNX small-model runtime, and llama.cpp VLM. Apple Silicon automatically includes Torch dependencies. For Torch small models or higher-throughput VLM serving, see [extension modules](extension_modules.md) for the `torch` and `full` extras. Do not use 3.x extras.

## First conversion

Parse a single file without a document library and write Markdown to a file:

```bash
mineru-kit parse document.pdf -o document.md --tier standard
```

The first model-backed parse may download weights. For native PDF text extraction without inference models, explicitly select:

```bash
mineru-kit parse document.pdf -o document.md --tier flash --ocr-mode txt
```

This reads the PDF text layer and does not add an OCR fallback for scanned pages. Use `--ocr-mode ocr` and prepare the relevant models for scanned PDFs.

Native documents and batch conversion:

```bash
mineru-kit parse report.docx -o report.md --tier flash
mineru-kit parse ./documents -o ./output --format zip
```

For directories or multiple inputs, `-o` must be a directory. For single-file Markdown output, use a file path. `mineru-kit parse` defaults to all PDF pages.

## Document library and agents

```bash
mineru parse document.pdf --json
mineru parse document.pdf --pages all -o document.md
mineru search "keyword" --json
```

`mineru` uses the document library and cache. `mineru parse` defaults to the first 10 PDF pages and may return a continuation request when the output reaches its length budget. Follow the returned locator and `next_request`. This differs from `mineru-kit parse`, which directly writes complete conversion outputs; see [Quick Usage](../usage/quick_usage.md).

## WebUI and API

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
```

Open the [local WebUI](http://127.0.0.1:7860). `mineru-webui` is the standalone command for the same entrypoint. Without `--api-url`, the WebUI manages a local V1 API server. See [SDK and API](../usage/sdk_api.md) to connect an existing service.

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

Inspect the running service through its [OpenAPI documentation](http://127.0.0.1:8000/docs).

## Install from source

Run from a repository checkout containing the 4.0 source:

```bash
uv pip install -e .
mineru version --json
```

Confirm a 4.x version; a development checkout may report a prerelease. For NVIDIA containers, see [Docker deployment](docker_deployment.md).

## Next steps

- [Tiers and runtimes](../usage/tiers.md)
- [Model downloads and configuration](../usage/model_source.md)
- [3.x → 4.0 migration](../reference/migration_4.md)
- [Official application](https://mineru.net/) and [online demos](../demo/index.md)
