# Quick Start

Install MinerU 4.x, verify local parsing with a small file, and open the resulting Markdown.

## Install MinerU

The MinerU package supports Python `>=3.11,<3.15`. Python 3.12 is a practical starting point for a new environment. Optional engines such as vLLM, LMDeploy, and Torch have additional wheel, OS, and driver constraints; the package's Python range does not guarantee every engine supports that entire range.

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

Install the latest published 4.x base package:

```bash
uv pip install -U "mineru>=4.0,<5"
```

Alternatively, run `python -m pip install -U "mineru>=4.0,<5"` in the activated environment. A package mirror can be selected with the installer's `-i` option.

The base package includes the WebUI, ONNX small-model runtime, and llama.cpp VLM. Apple Silicon automatically includes Torch dependencies. For Torch small models or higher-throughput VLM serving, see [extension modules](extension_modules.md) for the `torch` and `full` extras. Do not use 3.x extras.

To identify an accelerator or prepare inference models, see [deployment](../deployment/index.md).

## First conversion

In an empty directory, create a small document and convert it to Markdown with the stateless CLI. The native HTML parser runs without a model download.

```bash
python -c "from pathlib import Path; Path('report.html').write_text('<h1>MinerU</h1><p>First conversion.</p>', encoding='utf-8')"
mineru-kit parse report.html -o report.md --tier flash
```

Open `report.md` in the same directory. It contains the heading `MinerU` and the text `First conversion.` To convert your own PDF, supply its path and the desired tier.

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

To keep results, search them, or continue reading, use the same `report.html` in the [document library guide](../usage/quick_usage.md) to start the library service and add your first file. The library command `mineru parse` requests the first 10 PDF pages by default.

### Document-library telemetry

Doclib includes usage and diagnostic telemetry, initially enabled. Inspect it with `mineru telemetry status` or `preview`, and use `mineru telemetry disable` to stop collection and sending. Set `MINERU_TELEMETRY_DISABLED=1` before server startup when needed. Document contents, file names, and paths are excluded; see [telemetry details](../../next/telemetry-minimum.md) for scope and controls.

## WebUI and API

```bash
mineru-kit webui --server-name 127.0.0.1 --server-port 7860
```

Open the [local WebUI](http://127.0.0.1:7860) and follow the [WebUI guide](../usage/webui.md) to upload a file, check its source, and download results. To call a parsing service from an application, start the API separately.

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

Inspect the running service through its [OpenAPI documentation](http://127.0.0.1:8000/docs). Send your [first HTTP request](../usage/http_api.md) to download an example result. Use [Tasks](../usage/tasks.md) for asynchronous batches; start the separate [file-parse service](../usage/file_parse.md) to upload one file and receive an attachment directly.

## Install from source

Run from a checkout of the corresponding version:

```bash
uv pip install -e .
mineru version --json
```

Confirm a 4.x version; a development checkout may report a prerelease. For NVIDIA containers, see [Docker deployment](docker_deployment.md).

## Next steps

- [Tiers and runtimes](../usage/tiers.md)
- [Model downloads and configuration](../usage/model_source.md)
- [3.x → 4.0 migration](../reference/migration_4.md)
- [4.0.x → 4.1 migration](../reference/migration_4_1.md)
- [Official application](https://mineru.net/) and [online demos](https://huggingface.co/spaces/opendatalab/MinerU)
