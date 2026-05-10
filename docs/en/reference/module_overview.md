# MinerU Module Overview

## Overview

This document summarizes the responsibility of each major module in the `mineru` package so contributors can quickly locate the right code area.

The goal is not to describe every function in detail. Instead, it explains what each module group is responsible for and when you should read it.

## Top-Level Package Layout

The `mineru` package is organized around five major responsibilities:

- `cli`: request entry, API server, client tools, router, and runtime orchestration
- `backend`: document parsing pipelines and output rendering
- `model`: model wrappers and document converters
- `data`: storage, IO abstraction, and path/schema helpers
- `utils`: shared low-level helpers used across the project

## Package Map

```text
mineru/
├── cli/
├── backend/
│   ├── pipeline/
│   ├── vlm/
│   ├── hybrid/
│   ├── office/
│   └── utils/
├── model/
│   ├── docx/
│   ├── pptx/
│   ├── xlsx/
│   ├── layout/
│   ├── ocr/
│   ├── mfr/
│   ├── table/
│   ├── ori_cls/
│   └── vlm/
├── data/
│   ├── data_reader_writer/
│   ├── io/
│   └── utils/
└── utils/
```

## `mineru.cli`

This package is the runtime entry layer of MinerU.

### Main responsibilities

- expose FastAPI endpoints
- provide local and remote client flows
- normalize uploaded files
- dispatch tasks into parsing backends
- manage worker routing and local API startup

### Important modules

- `fast_api.py`
  - Main HTTP service entry.
  - Validates requests, stores uploads, builds parse jobs, and returns results.
- `common.py`
  - Core orchestration shared by CLI and API paths.
  - Reads file bytes, dispatches by backend, and writes final output files.
- `api_client.py`
  - Client-side helpers for talking to the MinerU API.
  - Handles task submission, polling, ZIP download, and local temporary server startup.
- `router.py`
  - Multi-worker routing layer for task distribution and managed local worker processes.
- `client.py`
  - Command-line client wrapper around API-based parsing workflows.
- `gradio_app.py`
  - Gradio web UI entry for interactive usage.
- `vlm_server.py`
  - Server process entry for VLM-compatible model serving workflows.
- `vlm_preload.py`
  - Preload logic for VLM workers to reduce cold-start overhead.
- `models_download.py`
  - CLI entry for model download and setup operations.
- `output_paths.py`
  - Normalizes output directory and file layout behavior.
- `api_protocol.py`
  - Shared protocol constants and version metadata.
- `public_http_client_policy.py`
  - Safety guardrails for public-bind HTTP client usage.

## `mineru.backend`

This package contains the actual document parsing backends and output rendering logic.

### `backend.pipeline`

The traditional OCR and layout pipeline.

#### Responsibility

- perform OCR-oriented document analysis
- batch page images
- convert model results into internal `middle_json`
- render final Markdown and structured outputs

#### Important modules

- `pipeline_analyze.py`
  - Main pipeline execution flow.
- `batch_analyze.py`
  - Batched image inference execution.
- `model_init.py`
  - Initializes pipeline models and caches model instances.
- `model_json_to_middle_json.py`
  - Converts raw model results into MinerU middle format.
- `pipeline_middle_json_mkcontent.py`
  - Renders pipeline `middle_json` into Markdown and content lists.
- `para_split.py`
  - Splits recognized content into paragraph-level structures.
- `pipeline_magic_model.py`, `model_list.py`
  - Supporting structures and mapping logic for pipeline inference.

### `backend.vlm`

The VLM-only document parsing path.

#### Responsibility

- run visual-language model extraction on PDF page images
- merge per-page VLM output into `middle_json`
- render VLM-style output artifacts

#### Important modules

- `vlm_analyze.py`
  - Main VLM backend execution path.
- `model_output_to_middle_json.py`
  - Converts VLM output to MinerU internal format.
- `vlm_middle_json_mkcontent.py`
  - Renders Markdown and content lists from VLM `middle_json`.
- `vlm_magic_model.py`, `utils.py`
  - VLM-specific helper logic and block handling.

### `backend.hybrid`

The combined VLM + OCR + formula refinement path.

#### Responsibility

- decide when OCR is needed
- combine VLM layout understanding with OCR enhancement
- enrich formulas and hard OCR regions
- output unified `middle_json`

#### Important modules

- `hybrid_analyze.py`
  - Main hybrid execution flow and OCR/VLM decision logic.
- `hybrid_model_output_to_middle_json.py`
  - Converts hybrid inference results into MinerU middle format.
- `hybrid_magic_model.py`
  - Hybrid-specific helper structures.

### `backend.office`

The office document parsing path for `docx`, `pptx`, and `xlsx`.

#### Responsibility

- analyze Office documents directly without going through PDF layout inference
- convert office parser results into `middle_json`
- render office outputs as Markdown and structured data

#### Important modules

- `docx_analyze.py`
  - DOCX parsing entry.
- `pptx_analyze.py`
  - PPTX parsing entry.
- `xlsx_analyze.py`
  - XLSX parsing entry.
- `model_output_to_middle_json.py`
  - Converts office parser output into `middle_json`.
- `office_middle_json_mkcontent.py`
  - Renders office results into Markdown and content lists.
- `office_magic_model.py`
  - Office-specific helper structures.

### `backend.utils`

Shared backend helpers used by multiple parsing paths.

#### Responsibility

- markdown rendering helpers
- OCR detection helpers
- runtime timing and progress helpers
- office image/chart handling
- paragraph block post-processing

#### Important modules

- `markdown_utils.py`
- `ocr_det_utils.py`
- `runtime_utils.py`
- `office_chart.py`
- `office_image.py`
- `html_image_utils.py`
- `para_block_utils.py`

## `mineru.model`

This package contains model-facing wrappers and format-specific converters.

It is the lower-level layer below `backend`.

### Office converters

- `model/docx/`
  - DOCX parsing and conversion to intermediate office structures.
- `model/pptx/`
  - PPTX parsing, normalization, and block ordering helpers.
- `model/xlsx/`
  - XLSX parsing and workbook-to-structured-output conversion.
- `office_stream.py`
  - Shared office stream handling helpers.

### ML model wrappers

- `model/layout/`
  - Layout analysis model wrappers such as document layout detection.
- `model/ocr/`
  - OCR model wrappers and OCR-related preprocessing utilities.
- `model/mfr/`
  - Formula recognition support.
- `model/table/`
  - Table parsing and recognition support.
- `model/ori_cls/`
  - Orientation classification.
- `model/vlm/`
  - VLM serving support such as `vllm` and `lmdeploy` server adapters.

### `model/utils`

Shared model-side utilities and OCR/tooling helpers used by the model wrappers.

## `mineru.data`

This package abstracts storage, IO, and path/schema concerns.

### `data_reader_writer`

Read/write abstraction for different storage targets.

#### Important modules

- `base.py`
  - Base reader/writer interfaces.
- `filebase.py`
  - Local filesystem implementation.
- `s3.py`
  - S3-backed implementation.
- `multi_bucket_s3.py`
  - Multi-bucket S3 support.
- `dummy.py`
  - Placeholder or no-op implementation.

### `io`

Transport-oriented IO helpers.

#### Important modules

- `base.py`
- `http.py`
- `s3.py`

These modules support reading or transferring data through external IO channels.

### `data.utils`

Supporting helpers for paths, schemas, and exceptions.

#### Important modules

- `path_utils.py`
- `schemas.py`
- `exceptions.py`

## `mineru.utils`

This package provides the cross-cutting helper layer used throughout the repo.

### Main responsibility areas

- PDF handling and page/image conversion
- file suffix and language guessing
- engine selection and environment configuration
- OCR and bbox utilities
- visualization and debug helpers
- model download and system environment checks

### Important modules

- `pdf_image_tools.py`
  - Converts PDF pages to images and image inputs to PDF bytes.
- `pdfium_guard.py`
  - Safe PDFium open/close and page-count helpers.
- `pdf_classify.py`
  - Decides OCR vs text extraction mode.
- `guess_suffix_or_lang.py`
  - Detects file suffix or language hints.
- `engine_utils.py`
  - Chooses runtime inference engines.
- `config_reader.py`
  - Reads runtime configuration from environment or config sources.
- `draw_bbox.py`
  - Generates layout/span visualization PDFs.
- `ocr_utils.py`
  - OCR result manipulation and normalization helpers.
- `model_utils.py`
  - Shared model-side runtime utilities such as crop and memory helpers.
- `models_download_utils.py`
  - Model download and path resolution helpers.
- `check_sys_env.py`
  - Environment compatibility checks.
- `enum_class.py`
  - Shared enums used across parsing and rendering layers.
- `bbox_utils.py`, `boxbase.py`
  - Bounding-box manipulation helpers.
- `table_merge.py`
  - Cross-page table merge helpers.
- `pdf_reader.py`, `pdf_text_tool.py`, `pdf_page_id.py`
  - Additional PDF parsing and indexing helpers.

## Recommended Reading Order

If you are new to the repo, this reading order is usually the fastest:

1. `mineru/cli/fast_api.py`
2. `mineru/cli/common.py`
3. One backend entry:
   - `mineru/backend/hybrid/hybrid_analyze.py`
   - or `mineru/backend/pipeline/pipeline_analyze.py`
4. Matching `*_middle_json_mkcontent.py`
5. Shared helpers in `mineru/utils/`

If you need office parsing specifically, start from:

1. `mineru/cli/common.py`
2. `mineru/backend/office/docx_analyze.py`
3. `mineru/model/docx/main.py`

## Quick Navigation Guide

- Need to understand request handling: `cli/`
- Need to understand file processing flow: `backend/`
- Need to understand model wrappers or converters: `model/`
- Need to change storage or IO targets: `data/`
- Need utility helpers or PDF/OCR support code: `utils/`
