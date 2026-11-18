# Frequently Asked Questions

## Why did I only get the first 10 PDF pages? {#first-ten-pages}

`mineru parse` requests the first 10 pages by default. To export the whole document, run `mineru parse document.pdf --pages all -o document.md`. For a one-time conversion, `mineru-kit parse document.pdf -o document.md` processes all pages by default. If the terminal shows only part of the text, follow the continuation command or use `-o` to write the complete requested range. [Library and page selection](../usage/quick_usage.md#pdf-page-selection)

## How do I recover a failed model download or run offline? {#offline-models}

Use `mineru-kit models show` on the deployment host to identify its effective backend and engine. On a connected machine, download and verify the models for that target combination, for example with `mineru-kit models verify --tier standard` (use `basic` for a Basic-only deployment). If Hugging Face is unreachable, select ModelScope. On the offline host, use the prepared model directory, the same backend and engine, and `model.source` set to `local`. [Model downloads and offline setup](../usage/model_source.md#offline-models)

## How do I get my result after a wait timeout? {#wait-timeout}

| Interface | What to do next |
| --- | --- |
| `mineru parse` | On `parse_wait_timeout`, repeat the same command to keep waiting. Use `--wait 600` for a longer wait. [CLI waits and exports](../usage/cli_tools.md) |
| Tasks | Keep the `task_id` from creation, poll `GET /v1/tasks/{task_id}`, and download each completed file from its `result_url`. [Tasks client examples](../usage/tasks.md#client-examples) |
| file-parse | On `202`, use `status_url` or the `Location` header. If you supplied a `request_id` before a disconnect, use it to query the wrapper and follow `result_url` once complete. [Waiting and recovery](../usage/file_parse.md#recovery) |

## Runtime and upgrade questions

??? question "Why is Windows inference slow, or engine installation failing?"

    ### Windows CUDA acceleration {#windows-cuda-acceleration}

    The base package can automatically use ONNX DirectML or CPU on Windows x64, with llama.cpp for the VLM. Use `mineru-kit hardware info --json` to inspect the environment and existing calibration evidence. For Torch small models and LMDeploy, install `"mineru[full]>=4.0,<5"` in the owning environment, then check Torch CUDA support, GPU drivers, and the Python range of engine wheels.

    The current version uses `lmdeploy>=0.17.0,<0.19` and Transformers 5. Do not reuse old LMDeploy 0.11.x wheel recipes or bypass dependency checks. See [extension modules](../quick_start/extension_modules.md) and [tiers](../usage/tiers.md).

??? question "What changed in commands, APIs, and configuration?"

    The 4.x WebUI uses `mineru-kit webui` / `mineru-webui`, parsing APIs use `/v1/*`, and configuration uses `config.yaml`. Old `/file_parse`, `/tasks`, and `model.stack` cannot be reused directly. See [3.x migration](../reference/migration_4.md). The 4.1 release also replaces standalone uploads/jobs with Tasks; see [4.1 migration](../reference/migration_4_1.md) and the standalone [file-parse API](../usage/file_parse.md).

??? question "Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2"

    ### Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2 {#wsl2-ubuntu2204-libgl}

    The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    Reference: [#388](https://github.com/opendatalab/MinerU/issues/388)

For other questions, consult [DeepWiki](https://deepwiki.com/opendatalab/MinerU) or contact the community on [Discord](https://discord.gg/Tdedn9GTXq) or [WeChat](https://mineru.net/community-portal/?aliasId=3c430f94).
