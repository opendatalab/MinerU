# Using MinerU

## Quick Model Source Configuration
MinerU uses `huggingface` as the default model source. If users cannot access `huggingface` due to network restrictions, they can conveniently switch the model source to `modelscope` through environment variables:
```bash
export MINERU_MODEL_SOURCE=modelscope
```
For more information about model source configuration and custom local model paths, please refer to the [Model Source Documentation](./model_source.md) in the documentation.

## Quick Usage via Command Line
MinerU has built-in command line tools that allow users to quickly use MinerU for document parsing through the command line:
```bash
mineru parse <input_path> --pages all -o <output_path>
```
> [!TIP]
>- `<input_path>`: One local `PDF` / `OFD` / `EPUB` / static `HTML` / image / `CSV` / `RTF` / `DOC`/`DOCX` / `PPT`/`PPTX` / `XLS`/`XLSX` / `ODT`/`ODS`/`ODP` file
>- `<output_path>`: Optional output file; without it, Markdown is written to stdout
>- PDF parsing defaults to the first 10 pages; use `--pages all` for the full document
>
> For more information about output files, please refer to [Output File Documentation](../reference/output_files.md).

> [!NOTE]
> The command line tool will automatically attempt cuda/mps acceleration on Linux and macOS systems. 
> Windows users who need cuda acceleration should visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to select the appropriate command for their cuda version to install acceleration-enabled `torch` and `torchvision`.

If you need to adjust parsing options through custom parameters, you can also check the more detailed [Command Line Tools Usage Instructions](./cli_tools.md) in the documentation.

## Advanced Usage via API, WebUI, and Services

- Start the self-hosted V1 API:
  ```bash
  mineru-kit api-server --host 0.0.0.0 --port 8000 --tier standard
  ```
  >[!TIP]
  >Access `http://127.0.0.1:8000/docs` for the OpenAPI documentation. The supported service surface is `/v1/*`, including health, capability discovery, uploads, files, parse jobs, and usage.
  >
  >HTTP asynchronous call code example: [Python version](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)

- Start Gradio WebUI visual frontend:
  ```bash
  mineru-kit gradio --server-name 0.0.0.0 --server-port 7860
  ```
  >[!TIP]
  >
  >- Access `http://127.0.0.1:7860` in your browser to use the Gradio WebUI.
  >- Without `--api-url`, Gradio manages a loopback `mineru-kit api-server`; with `--api-url`, it connects only to that existing V1 service.
  >- Use `--api-server-preload-models` to preload models for the managed local server.
  >- `mineru-gradio` remains available as a command-name alias with the same modern options.

- Use `mineru-router` for multi-service / multi-GPU orchestration:
  ```bash
  mineru-router --host 0.0.0.0 --port 8002 --local-gpus auto --worker-tier standard
  ```
  >[!TIP]
  >
  >- `mineru-router` and `mineru-kit router` expose the complete `/v1/*` API.
  >- Repeat `--upstream-url` to aggregate multiple existing V1 api-server services, or use `--local-gpus` to launch `mineru-kit api-server` workers automatically.
  >- Use `--preload-models` for router-managed workers; remote upstreams keep their own startup configuration.
  >- Unknown model-engine arguments are not forwarded by Router.
  >- It is intended for advanced multi-service, multi-GPU, and unified-entry deployments.

- Start an OpenAI-compatible VLM server:
  ```bash
  mineru-kit vlm-server --engine auto --port 30000
  ```

> [!NOTE]
> Model-engine parameters apply only to commands that explicitly declare them. `mineru-router` accepts documented Router/worker options and does not forward unknown arguments.
> We have compiled some commonly used parameters and usage methods for `vllm/lmdeploy`, which can be found in the documentation [Advanced Command Line Parameters](./advanced_cli_parameters.md).

## Configuring LLM-aided post-processing with config.yaml

LLM-aided title leveling and cross-page table cell continuation read `$MINERU_HOME/config.yaml` and support OpenAI-compatible model services:

```yaml
llm_aided:
  api_key: ${MINERU_LLM_API_KEY:-}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  model: qwen3.5-plus
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

## Extending MinerU Functionality with Configuration Files

MinerU works out of the box and reads current settings from `$MINERU_HOME/config.yaml`; set `MINERU_CONFIG` to use another file. Legacy `mineru.json` CLI settings are no longer supported. Gradio's LaTeX delimiters are selected with `--latex-delimiters-type`.

Model storage and source settings use the `model` section:

```yaml
model:
  base_dir: ~/.mineru/models
  source: auto
  stack: auto
```

See [Model Source Documentation](./model_source.md) for model download and local-source details.

## PDF page selection

Use `--pages "1-5,8,r3-r1"`: page numbers start at 1, ranges include both endpoints,
and `r1` means the last page. Use `all` for every page. Results are sorted and deduplicated;
partially out-of-bounds ranges select their valid intersection. Reversed or empty selections
fail with `page_range_invalid`. Without `--pages`, `mineru parse` starts with the first 10 pages;
`mineru-kit parse`, Python and Gradio select all pages. New requests use the current syntax. Historical positive result ranges using ASCII `~`
remain readable without rebuilding Doclib caches; result responses and new cache entries use `-`.
Fullwidth `～` and negative page-number notation are not supported.
See [page-range syntax and historical result compatibility](../../next/page-ranges.md).
