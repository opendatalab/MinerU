# Tiers and Runtimes

Tiers express parsing quality and cost. Small-model backends and VLM engines are configured independently; they are not tier names.

This page includes 4.1 device-selection and service-tier behavior. Existing deployments should follow the [4.1 migration guide](../reference/migration_4_1.md).

| Tier | Purpose | Local execution |
| --- | --- | --- |
| `flash` | Fast previews, indexing, native documents | Native PDF text / document parsing; Flash OCR for scans and images |
| `basic` | OCR, formulas, and tables | ONNX or Torch small models; no local VLM required |
| `standard` | Complex layouts and high-quality parsing | Small models + VLM |
| `advanced` | More demanding quality requirements | Standard models and runtime, with more inference computation |

PDF and images support all four tiers. Office, OpenDocument, RTF, EPUB, OFD, HTML/MHTML, and CSV/TSV use local Flash and whole-document parsing. Pass `tier="flash"` explicitly for these formats in the local Python SDK. MHTML web archives (`.mhtml`, `.mht`) are parsed as whole documents. Plain text is not parsed, but can be indexed and read by the document library.

For a saved web page archive:

```bash
mineru-kit parse saved-page.mhtml -o saved-page.md --tier flash
```

## Default selection

- `mineru-kit parse` and local Python `parse()` default to Standard for PDFs.
- After discovering service capabilities, the document library prefers Standard, then Basic for PDF/images. If neither is available, it returns `quality_tier_unavailable` rather than silently choosing Flash or uploading to the official service.
- Tasks and Gradio share the Standard → Basic → Flash → Advanced preference, selecting Advanced automatically only when it is the sole available tier. Advanced shares models with Standard rather than downloading separate Advanced weights; the service must enable Advanced to accept corresponding PDF/image requests.
- Library `read` consumes cached results without starting a new parse. Without an explicit tier, it prefers cached Advanced, Standard, then Basic results.

The API startup `--tier` is a ceiling including itself and lower tiers; `--tier standard` does not enable Advanced. Omission or `--tier auto` recommends a ceiling from the effective backend and hardware, potentially calibrating ONNX at startup. HTTP requests without a tier choose Standard → Basic → Flash → Advanced. Native documents always use Flash, independently of the enabled PDF/image tiers. See [Tasks](tasks.md) for the full rules.

## Backends and engines

| Environment | Automatic small models | Automatic VLM |
| --- | --- | --- |
| Apple Silicon / MPS | Torch / MPS | llama.cpp / Metal |
| Windows x64, base package | ONNX / DirectML or CPU when no Torch accelerator runtime is available | llama.cpp |
| Linux x86_64, base package | ONNX / WebGPU or CPU when no Torch accelerator runtime is available | llama.cpp |
| Linux, accelerator and `full` | Torch | vLLM, otherwise installed LMDeploy |
| Windows, accelerator and `full` | Torch | LMDeploy |
| CPU environment | ONNX / CPU | llama.cpp |

Automatic selection uses installed dependencies and available devices. An explicit choice with missing dependencies fails instead of silently selecting another backend. `model.onnx_device=auto|cpu|gpu` controls ONNX devices independently of the VLM. `MINERU_MODEL_ONNX_DEVICE` overrides file configuration without persisting the override.

The 4.1 base package installs DirectML on Windows x64 and the Vulkan-backed WebGPU plugin on Linux x86_64. Compatible, confirmed discrete GPUs have priority. In auto mode, integrated or unclassified GPUs undergo applicable calibration; insufficient gains or unreliable execution select CPU. Linux uses CPU for older Intel/Xe-LP and unidentified Intel architectures, even with explicit gpu selection. Other systems, unavailable GPUs, or initialization failures log CPU fallback reasons.

Valid automatic choices can be saved to configuration. Set `model.onnx_device` back to auto and restart to recalibrate. Hardware changes invalidate old performance evidence; explicit or saved cpu choices are not overwritten when a discrete GPU is later connected. First calibration may download the recognition model; `hardware info` does not trigger it.

ONNX layout, OCR recognition, MFR, and UNet share device selection. Ordinary OCR detection uses CPU on Windows and may use WebGPU on Linux; seal detection, table classification, and SLANet Plus use CPU. GPU MFR handles full batches of eight, with remaining tails on CPU. The old `MINERU_TABLE_DEVICE` does not override ONNX UNet; Torch table models retain their CPU/CUDA policy. This ONNX path does not require manually replacing the runtime with `onnxruntime-gpu`.

Intel XPU is excluded from automatic LMDeploy selection. Linux prefers an installed XPU-compatible vLLM and otherwise falls back to llama.cpp; Windows uses llama.cpp.

```yaml
model:
  small_backend: auto
  onnx_device: auto
  vlm:
    engine: auto
```

`small_backend` accepts `auto/onnx/torch`; `engine` accepts `auto/llama-cpp/vllm/lmdeploy`. macOS defaults to llama.cpp.

```bash
mineru-kit hardware detect --json
mineru-kit hardware info --json
```

detect shares API startup recommendations and may perform applicable ONNX calibration. info reads hardware, installed runtimes, effective configuration, and calibration records without benchmarking, downloading weights, or writing configuration.

## Resources and platform boundaries

Native text parsing needs no inference models. Basic can run on CPU. Standard / Advanced speed and memory depend on the VLM engine, input size, and concurrency. Plan for at least 16 GB of system memory for higher-throughput local deployment. NVIDIA deployments also need an engine-supported GPU, driver, and sufficient free VRAM; 8 GB is only a planning starting point, not a guarantee for every workload.

For Apple Silicon, at least 16 GB of unified memory and a direct macOS installation are recommended. The 4.1 Windows/Linux ONNX path supports compatible Intel, AMD, and NVIDIA GPUs; legacy vendor-accelerator guides remain scoped to [MinerU <4](compatibility.md).

See [extension modules](../quick_start/extension_modules.md) for installation and [Model Source](model_source.md) for downloads and configuration.
