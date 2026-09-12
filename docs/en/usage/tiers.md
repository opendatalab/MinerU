# Tiers and Runtimes

Tiers express parsing quality and cost. Small-model backends and VLM engines are configured independently; they are not tier names.

| Tier | Purpose | Local execution |
| --- | --- | --- |
| `flash` | Fast previews, indexing, native documents | Native PDF text / document parsing; Flash OCR for scans and images |
| `basic` | OCR, formulas, and tables | ONNX or Torch small models; no local VLM required |
| `standard` | Complex layouts and high-quality parsing | Small models + VLM |
| `advanced` | More demanding quality requirements | Standard models and runtime, with more inference computation |

PDF and images support all four tiers. Office, OpenDocument, RTF, EPUB, OFD, HTML, and CSV/TSV use local Flash and whole-document parsing. Pass `tier="flash"` explicitly for these formats in the local Python SDK. Plain text is not parsed, but can be indexed and read by the document library.

## Default selection

- `mineru-kit parse` and local Python `parse()` default to Standard for PDFs.
- After discovering service capabilities, the document library prefers Standard, then Basic for PDF/images. If neither is available, it returns `quality_tier_unavailable` rather than silently choosing Flash or uploading to the official service.
- Advanced is explicit. Prepare models and managed service capacity as Standard; there is no separate Advanced model download.
- Library `read` consumes cached results without starting a new parse. Without an explicit tier, it prefers cached Advanced, Standard, then Basic results.

## Backends and engines

| Environment | Automatic small models | Automatic VLM |
| --- | --- | --- |
| Apple Silicon / MPS | Torch / MPS | llama.cpp |
| Linux / Windows, base package | ONNX / CPU when no Torch accelerator runtime is available | llama.cpp |
| Linux, accelerator and `full` | Torch | vLLM, otherwise installed LMDeploy |
| Windows, accelerator and `full` | Torch | LMDeploy |
| CPU environment | ONNX / CPU | llama.cpp |

Automatic selection uses installed dependencies and available devices. An explicit choice with missing dependencies fails instead of silently selecting another backend. ONNX small models run on CPU; VLM device selection is independent.

```yaml
model:
  small_backend: auto
  vlm:
    engine: auto
```

`small_backend` accepts `auto/onnx/torch`; `engine` accepts `auto/llama-cpp/vllm/lmdeploy/mlx`. macOS does not select MLX automatically: install and configure it explicitly.

## Resources and platform boundaries

Native text parsing needs no inference models. Basic can run on CPU. Standard / Advanced speed and memory depend on the VLM engine, input size, and concurrency. Plan for at least 16 GB of system memory for higher-throughput local deployment. NVIDIA deployments also need an engine-supported GPU, driver, and sufficient free VRAM; 8 GB is only a planning starting point, not a guarantee for every workload.

For Apple Silicon, at least 16 GB of unified memory and a direct macOS installation are recommended. Existing AMD and vendor adaptations remain on [MinerU <4](compatibility.md); this page does not extend their compatibility claims to 4.0.

See [extension modules](../quick_start/extension_modules.md) for installation and [Model Source](model_source.md) for downloads and configuration.
