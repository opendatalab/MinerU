# MinerU-Diffusion Benchmark Harness

This directory contains a small benchmark harness for MinerU-Diffusion style
OpenAI-compatible OCR endpoints. It is intentionally separated from model
runtime code so it can be used with vLLM, HF remote-code servers, or other
compatible services.

The harness covers four use cases:

- single-image text/table/formula/layout requests;
- end-to-end two-step page parsing, layout first and content extraction second;
- PDF-page suite rendering and batch throughput measurement;
- output quality comparison, including markdown similarity and layout box F1.

## Single-Image Benchmark

Start a compatible server, then run the default four tasks:

```bash
python -m benchmarks.mineru_diffusion.bench_hf_server \
  --endpoint http://127.0.0.1:18083/v1/chat/completions \
  --image file:///path/to/page.png \
  --output-dir benchmark_results/mineru_diffusion/single_image \
  --timeout 600
```

To cap individual task budgets:

```bash
python -m benchmarks.mineru_diffusion.bench_hf_server \
  --endpoint http://127.0.0.1:18083/v1/chat/completions \
  --image file:///path/to/page.png \
  --output-dir benchmark_results/mineru_diffusion/single_image_budgeted \
  --task-max-tokens text=1024,table=1024,formula=768,layout=128
```

Results are written as JSONL plus `latest_summary.json` under the selected
output directory.

## Native HF Remote-Code Helper

For local HF remote-code smoke tests, the helper below starts the bundled
compatibility server, waits for `/health`, runs the client, and shuts it down:

```bash
python -m benchmarks.mineru_diffusion.run_native_benchmark \
  --cuda-visible-devices 0 \
  --model-path /path/to/MinerU-Diffusion-V1-0320-2.5B \
  --image file:///path/to/page.png \
  --output-dir benchmark_results/mineru_diffusion/native_direct
```

The lower-level server can also be started manually:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.mineru_diffusion.native_server \
  --model-path /path/to/MinerU-Diffusion-V1-0320-2.5B \
  --host 127.0.0.1 \
  --port 18083 \
  --device cuda:0 \
  --dtype bfloat16
```

`hf_server.py` provides the same OpenAI-compatible surface for an HF
remote-code reference path.

## PDF-Page Suite

Render a manifest from local PDFs:

```bash
python -m benchmarks.mineru_diffusion.end2end_suite \
  render \
  --output-dir benchmark_results/mineru_diffusion/pdf_suite_coverage
```

Run a two-step parse against an OpenAI-compatible endpoint:

```bash
python -m benchmarks.mineru_diffusion.end2end_suite \
  run \
  --manifest benchmark_results/mineru_diffusion/pdf_suite_coverage/manifest.json \
  --endpoint http://127.0.0.1:18083/v1/chat/completions \
  --output-dir benchmark_results/mineru_diffusion/pdf_suite_dllm \
  --layout-concurrency 4 \
  --content-concurrency 4 \
  --dynamic-threshold 0.90
```

The suite records page-level latency, layout output, extracted blocks, markdown,
and aggregate summary metrics.

## Quality Comparison

Compare a candidate result file against a baseline:

```bash
python -m benchmarks.mineru_diffusion.compare_results \
  --baseline benchmark_results/mineru_diffusion/baseline/latest_results.jsonl \
  --candidate benchmark_results/mineru_diffusion/candidate/latest_results.jsonl \
  --output benchmark_results/mineru_diffusion/compare.json \
  --similarity-cases text,table \
  --min-similarity 0.95 \
  --max-control-repeat 8
```

The comparison report includes character-level similarity, output length ratio,
control-token ratio, longest repeated control-token run, and layout box
precision/recall/F1 for layout cases.

## Layout Sampling Experiment

`layout_sampling_experiment.py` compares default sampling against MinerU-style
deterministic layout sampling for the layout stage:

```bash
python -m benchmarks.mineru_diffusion.layout_sampling_experiment \
  --manifest benchmark_results/mineru_diffusion/pdf_suite_coverage/manifest.json \
  --endpoint http://127.0.0.1:18083/v1/chat/completions \
  --baseline-results benchmark_results/mineru_diffusion/baseline/latest_results.jsonl \
  --output-dir benchmark_results/mineru_diffusion/layout_sampling_experiment \
  --layout-concurrency 4 \
  --dynamic-threshold 0.90
```

The experiment writes per-variant JSONL files and a markdown summary.
