# Model Downloads and Configuration

MinerU 4.0 configures small-model backends independently from VLM engines. Configuration defaults to `$MINERU_HOME/config.yaml` (`MINERU_HOME` defaults to `~/.mineru`); `MINERU_CONFIG` selects another file.

```yaml
model:
  source: auto
  base_dir: ~/.mineru/models
  small_backend: auto
  vlm:
    engine: auto
```

## Model sources

`model.source` accepts `auto`, `huggingface`, `modelscope`, and `local`. `auto` probes Hugging Face first and selects ModelScope when it is unavailable. Automatic selection from a default or config value may be written back to configuration. Environment variables override file settings:

```bash
export MINERU_MODEL_SOURCE=modelscope
mineru-kit parse document.pdf -o document.md --tier standard
```

Windows PowerShell:

```powershell
$env:MINERU_MODEL_SOURCE = "modelscope"
```

The small-model bundles `MinerU-4_models_torch` and `MinerU-4_models_onnx` are registered for both Hugging Face and ModelScope. Original VLM weights and llama.cpp GGUF/mmproj use different repositories selected by the engine. Basic needs the selected small-model bundle; Standard also needs VLM models and serves Advanced requests.

## Download, verify, and run offline

Download and verify a Standard deployment using the current configuration:

```bash
mineru-kit models download --tier standard --source huggingface
mineru-kit models verify --tier standard
mineru-kit models show
```

An explicit CPU small-model and llama.cpp combination:

```bash
mineru-kit models download --tier standard --small-backend onnx --vlm-engine llama-cpp --source huggingface
mineru-kit models verify --tier standard --small-backend onnx --vlm-engine llama-cpp
```

A NVIDIA / vLLM deployment can download its models on a build machine without a GPU. Specify the target backends instead of relying on that machine's automatic selection:

```bash
mineru-kit models download --tier standard --small-backend torch --vlm-engine vllm --source huggingface
mineru-kit models verify --tier standard --small-backend torch --vlm-engine vllm
```

These command options apply only to the current operation and do not change persistent configuration. Use the same combination for inference, for example:

```bash
export MINERU_MODEL_SMALL_BACKEND=torch
export MINERU_MODEL_VLM_ENGINE=vllm
export MINERU_MODEL_SOURCE=local
mineru-kit parse document.pdf -o document.md --tier standard
```

Use `basic` as the download tier when no VLM is needed. Advanced has no separate deployment download tier. Set `model.base_dir` before downloading to change the model root. Completion markers and required files determine readiness. Repeated downloads reuse provider caches; do not create completion markers manually.

`local` uses ready local models and fails on missing assets without downloading. An explicit `models download` is a download operation: even with `local` configured, it temporarily resolves a remote source automatically.

## VLM servers and manual MLX

An existing `model.vlm.server_url` takes priority and removes the local VLM weight requirement. This is a model inference endpoint, not the MinerU V1 document parsing API. Small-model requirements still depend on the tier.

MLX requires manual installation of `mlx-vlm>=0.7.0,<0.8.0` and explicit `model.vlm.engine: mlx`. `model.stack`, `MINERU_MODEL_STACK`, and `--stack` have been removed. Restart relevant services after configuration changes.

## Download troubleshooting

Hugging Face uses `hf_xet` by default. If your network cannot reach Xet CAS, set `HF_HUB_DISABLE_XET=1` before downloading to use regular HTTP, or switch to ModelScope. For offline deployment, download and verify before selecting `local`. See [extension modules](../quick_start/extension_modules.md) for installation combinations.
