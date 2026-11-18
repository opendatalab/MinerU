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

## Download, verify, and run offline {#offline-models}

<span id="download-verify-and-run-offline"></span>

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

Explicit `mlx` is also supported for local VLM inference. It is never selected by auto mode and is not a MinerU extra. See [extension modules](../quick_start/extension_modules.md) for its separate installation.

## Remote VLM service

An existing `model.vlm.server_url` takes priority and removes the local VLM weight requirement. This is a model inference endpoint, not the MinerU V1 document parsing API. Small-model requirements still depend on the tier. The full field set:

| Field | Default | Purpose |
| --- | --- | --- |
| `model.vlm.engine` | `auto` | Local engine selection: `auto/llama-cpp/vllm/lmdeploy/mlx`; not used while a remote `server_url` is set |
| `model.vlm.server_url` | (unset) | Remote VLM inference endpoint. Must be HTTP(S) without credentials, query, or fragment; a trailing `/v1` is stripped and the URL is kept ending with `/` (reverse-proxy path prefixes are preserved) |
| `model.vlm.api_key` | (unset) | Bearer key for the remote VLM service; environment variable `MINERU_MODEL_VLM_API_KEY` |
| `model.vlm.model` | (unset) | Model name requested from the remote service; environment variable `MINERU_MODEL_VLM_MODEL` |
| `model.vlm.http_timeout` | `600` | Per-request timeout in seconds |
| `model.vlm.max_concurrency` | `100` | Maximum concurrent VLM requests |

Legacy `MINERU_VL_API_KEY` / `MINERU_VL_MODEL_NAME` are rejected when they conflict with these fields; see the [migration guide](../reference/migration_4.md).

`model.stack`, `MINERU_MODEL_STACK`, and `--stack` have been removed. Restart relevant services after configuration changes.

## Three remote connections at a glance

MinerU talks to three different kinds of remote services. They are not interchangeable — do not reuse one service's address, key, or model name for another:

| Connection | Purpose | Where it is configured |
| --- | --- | --- |
| V1 document parsing service | Upload documents, submit parse jobs, download results (self-hosted `mineru-kit api-server`, router, or the official cloud service) | Per client: `MINERU_API_URL` / `MINERU_API_KEY`, or SDK `MinerUApiParser(api_url=..., api_key=...)`. The document library's default remote target is a library runtime setting, not part of `config.yaml` |
| Remote VLM inference service | Model inference for local parse pipelines (Standard/Advanced), OpenAI-compatible | `model.vlm.server_url` / `api_key` / `model` in `config.yaml` |
| LLM-aided post-processing | Title leveling and cross-page table cell continuation | `llm_aided.api_key` / `base_url` / `model` in `config.yaml`; both features are disabled by default |

## LLM-aided post-processing {#llm-post-processing}

LLM-aided title leveling and cross-page table cell continuation read `$MINERU_HOME/config.yaml` and support OpenAI-compatible model services:

```yaml
llm_aided:
  api_key: ${MINERU_LLM_API_KEY:-}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen3.8-flash
  enable_thinking: false
  max_concurrency: 16
  features:
    title_leveling: false
    cross_page_table_cell_merge: false
```

- `title_leveling` groups paragraph titles into levels 2 through 6 by document-title boundaries and runs only when `MiddleJson.is_full_document` is `true`. Page-selected input is persisted as `false` and skips title leveling.
- `cross_page_table_cell_merge` asks the LLM whether each pair of boundary-row cells continues after the existing rules identify a cross-page table.
- Table cell merge does not require whole-document input. Both features are disabled by default and share one asynchronous client, one connection configuration, and the `max_concurrency` request limit, which defaults to 16.
- `max_concurrency` must be an integer of at least 1 and can be overridden with `MINERU_LLM_AIDED_MAX_CONCURRENCY`.
- Enabling either feature requires non-empty `api_key`, `base_url`, and `model` values.
- `enable_thinking` is optional. When omitted, the extension parameter is not sent to the model service.
- The legacy `llm-aided-config` section in `mineru.json` is no longer read.

## Download troubleshooting

Hugging Face uses `hf_xet` by default. If your network cannot reach Xet CAS, set `HF_HUB_DISABLE_XET=1` before downloading to use regular HTTP, or switch to ModelScope. For offline deployment, download and verify before selecting `local`. See [extension modules](../quick_start/extension_modules.md) for installation combinations.
