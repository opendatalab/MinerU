# MinerU

MinerU is a document parsing toolkit for converting:

- `PDF`
- images
- `DOCX`
- `PPTX`
- `XLSX`

into structured outputs such as:

- Markdown
- JSON
- content lists
- extracted images

This repository contains the source code, local runtime wrappers, backend flows, model loading logic, and project documentation.

## What This Repo Is Good For

Use this repo if you want to:

- parse documents locally
- run `pipeline`, `vlm`, or `hybrid` flows directly
- inspect MinerU internals
- customize model loading or backend behavior
- integrate MinerU into another Python workflow

## Supported Main Flows

### `pipeline`

Traditional OCR + layout parsing.

- best compatibility
- supports pure CPU
- stable and practical
- good default when GPU is unavailable

### `vlm`

Vision-language-model-based parsing.

- better layout understanding
- higher hardware requirements
- local or remote server mode

### `hybrid`

VLM + OCR/formula refinement.

- strongest overall PDF parsing path in this repo
- local `auto-engine` mode requires stronger hardware
- `http-client` mode can be used with a remote compatible server

## Backend Comparison

| Backend | Runs on CPU | Needs local heavy model | Notes |
|---|---:|---:|---|
| `pipeline` | Yes | Yes | Best fallback and most compatible |
| `vlm-auto-engine` | No | Yes | Local VLM |
| `hybrid-auto-engine` | No | Yes | Local VLM + OCR refinement |
| `vlm-http-client` | Yes | No | Remote VLM server |
| `hybrid-http-client` | Yes | Partial | Remote VLM server + local pipeline deps |

## Repository Layout

```text
MinerU/
├── mineru/
│   ├── cli/
│   ├── backend/
│   │   ├── pipeline/
│   │   ├── vlm/
│   │   ├── hybrid/
│   │   └── office/
│   ├── model/
│   ├── data/
│   ├── utils/
│   └── direct_flow_processor.py
├── docs/
├── demo/
├── setup.sh
├── run.sh
└── pyproject.toml
```

Important parts:

- `mineru/cli`: CLI, FastAPI, router, orchestration
- `mineru/backend`: the actual parsing flows
- `mineru/model`: model wrappers and format-specific converters
- `mineru/utils`: PDF, OCR, config, and model helper utilities
- `mineru/direct_flow_processor.py`: direct Python wrapper for `pipeline`, `vlm`, and `hybrid` without going through the API layer

## Quick Start

### 1. Setup local environment

```bash
./setup.sh
```

This script:

- creates `.venv`
- installs MinerU in editable mode
- creates `mineru.json` in the repo
- prepares model directories inside the repo

If you want to download models during setup:

```bash
MINERU_DOWNLOAD_MODELS=1 ./setup.sh
```

If you prefer ModelScope:

```bash
MINERU_DOWNLOAD_MODELS=1 MINERU_DOWNLOAD_SOURCE=modelscope ./setup.sh
```

### 2. Parse a document

```bash
./run.sh parse demo/pdfs/demo1.pdf
```

With explicit output dir:

```bash
./run.sh parse demo/pdfs/demo1.pdf ./output
```

Run `pipeline` on CPU:

```bash
MINERU_DEVICE_MODE=cpu ./run.sh parse demo/pdfs/demo1.pdf ./output -b pipeline
```

### 3. Start the local API

```bash
./run.sh api --host 127.0.0.1 --port 8000
```

### 4. Start Gradio

```bash
./run.sh gradio --server-name 0.0.0.0 --server-port 7860
```

## Local Model Location

This repo has been adjusted so local model config and downloaded model cache can live inside the repository.

Default locations:

- config: `./mineru.json`
- pipeline model root: `./.mineru/models/pipeline`
- vlm model root: `./.mineru/models/vlm`

Important environment variables:

- `MINERU_MODEL_SOURCE`
- `MINERU_TOOLS_CONFIG_JSON`
- `MINERU_DEVICE_MODE`

Typical values:

```bash
export MINERU_MODEL_SOURCE=local
export MINERU_DEVICE_MODE=cpu
```

## Direct Python Usage

If you want to bypass the API layer completely, use:

- `mineru.direct_flow_processor.DirectFlowProcessor`

Example:

```python
from mineru.direct_flow_processor import DirectFlowProcessor


def main():
    processor = DirectFlowProcessor(output_root="./output")
    result, saved = processor.run_pipeline(
        "demo/pdfs/demo1.pdf",
        parse_method="auto",
        language="en",
        formula_enable=True,
        table_enable=True,
        draw_layout=True,
        draw_span=True,
    )
    print(result.flow)
    print(saved)


if __name__ == "__main__":
    main()
```

Note:

- for `pipeline`, `vlm`, and `hybrid`, this wrapper calls backend code directly
- it does not go through `mineru-api`
- if your script triggers multiprocessing PDF rendering, keep your entrypoint under `if __name__ == "__main__":`

## Main Processing Flow

At a high level, the repo works like this:

1. read input bytes
2. normalize PDF/image input
3. choose backend
4. analyze pages
5. build `middle_json`
6. render Markdown / JSON / images
7. save output files

For detailed flow documentation, see:

- [Processing Flow](docs/en/reference/processing_flow.md)
- [Module Overview](docs/en/reference/module_overview.md)

## Documentation Map

Useful docs in this repo:

- [Quick Usage](docs/en/usage/quick_usage.md)
- [CLI Tools](docs/en/usage/cli_tools.md)
- [Model Source](docs/en/usage/model_source.md)
- [Output Files](docs/en/reference/output_files.md)
- [Processing Flow](docs/en/reference/processing_flow.md)
- [Module Overview](docs/en/reference/module_overview.md)
- [Vietnamese Reference in `docs/en/reference`](docs/en/reference/reference_vi.md)
- [Vietnamese Overview](docs/docs-vn.md)

## Common Development Notes

### Running on CPU

Use:

```bash
MINERU_DEVICE_MODE=cpu ./run.sh parse <input> <output> -b pipeline
```

If you need `hybrid` on a CPU-only machine, use:

- `hybrid-http-client`

with a remote compatible VLM server.

### Using a custom model API

You can replace the VLM-serving side with your own API, but you must adapt its output into the block schema expected by MinerU.

Relevant code:

- `mineru/backend/vlm/vlm_analyze.py`
- `mineru/backend/vlm/model_output_to_middle_json.py`
- `mineru/backend/vlm/vlm_magic_model.py`

### Multiprocessing note

When calling direct parsing code from your own Python script, always use:

```python
if __name__ == "__main__":
    ...
```

especially for `pipeline`, because PDF rendering uses multiprocessing.

## Suggested Reading Order

If you are new to the codebase:

1. `mineru/cli/fast_api.py`
2. `mineru/cli/common.py`
3. `mineru/backend/hybrid/hybrid_analyze.py`
4. `mineru/backend/pipeline/pipeline_analyze.py`
5. `mineru/backend/vlm/vlm_analyze.py`
6. `mineru/direct_flow_processor.py`
7. `mineru/utils/models_download_utils.py`
8. `mineru/utils/config_reader.py`

## License

This repository is licensed under the [MinerU Open Source License](LICENSE.md).
