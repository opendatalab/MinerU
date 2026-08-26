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
mineru -p <input_path> -o <output_path>
```
> [!TIP]
>- `<input_path>`: Local `PDF` / image / `CSV` / `DOC`/`DOCX` / `PPT`/`PPTX` / `XLS`/`XLSX` file or directory
>- `<output_path>`: Output directory
>- Without `--api-url`, the CLI launches a temporary local `mineru-api`
>- With `--api-url`, the CLI connects to an existing local or remote FastAPI service directly
>
> For more information about output files, please refer to [Output File Documentation](../reference/output_files.md).

> [!NOTE]
> The command line tool will automatically attempt cuda/mps acceleration on Linux and macOS systems. 
> Windows users who need cuda acceleration should visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to select the appropriate command for their cuda version to install acceleration-enabled `torch` and `torchvision`.

If you need to adjust parsing options through custom parameters, you can also check the more detailed [Command Line Tools Usage Instructions](./cli_tools.md) in the documentation.

## Advanced Usage via API, WebUI, http-client/server

- FastAPI calls:
  ```bash
  mineru-api --host 0.0.0.0 --port 8000
  ```
  >[!TIP]
  >Access `http://127.0.0.1:8000/docs` in your browser to view the API documentation.
  >
  >- Health endpoint: `GET /health`
  >  Returns `protocol_version`, `processing_window_size`, `max_concurrent_requests`, and task stats
  >- Asynchronous task submission endpoint: `POST /tasks`
  >- Synchronous parsing endpoint: `POST /file_parse`
  >- Task query endpoints: `GET /tasks/{task_id}`, `GET /tasks/{task_id}/result`
  >- API outputs are controlled by the server and written to `./output` by default
  >- Uploads currently support `PDF`, image, `CSV`, `DOCX`, `PPTX`, and `XLSX` files
  >
  >- `POST /tasks` returns immediately with a `task_id`. `POST /file_parse` uses the same task manager internally, waits for the task to finish, and then returns the final result synchronously.
  >- When a task is waiting in the queue, both the submission response and task-status response may include `queued_ahead` to indicate how many tasks are ahead of it.
  >- Tasks are tracked only in-process for a single `mineru-api` instance. Task status is not preserved across service restarts, `--reload`, or multi-process deployments.
  >- Completed or failed tasks are retained for 24 hours by default, then their task state and output directory are cleaned automatically. After cleanup, task status and result endpoints return `404`.
  >- Use `MINERU_API_TASK_RETENTION_SECONDS` and `MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS` to adjust retention and cleanup polling intervals.
  >- Use `--enable-vlm-preload true` to warm up the local VLM model during service startup instead of waiting for the first VLM or hybrid request.
  >
  >Asynchronous task submission example:
  >```bash
  >curl -X POST http://127.0.0.1:8000/tasks \
  >  -F "files=@demo/pdfs/demo1.pdf" \
  >  -F "return_md=true"
  >```
  >
  >Synchronous parsing example:
  >```bash
  >curl -X POST http://127.0.0.1:8000/file_parse \
  >  -F "files=@demo/pdfs/demo1.pdf" \
  >  -F "return_md=true" \
  >  -F "response_format_zip=true" \
  >  -F "return_original_file=true"
  >```
  >
  >Poll task status and fetch results:
  >```bash
  >curl http://127.0.0.1:8000/tasks/<task_id>
  >curl http://127.0.0.1:8000/tasks/<task_id>/result
  >curl http://127.0.0.1:8000/health
  >```
  >
  >HTTP asynchronous call code example: [Python version](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)

- Start Gradio WebUI visual frontend:
  ```bash
  mineru-gradio --server-name 0.0.0.0 --server-port 7860
  ```
  >[!TIP]
  >
  >- Access `http://127.0.0.1:7860` in your browser to use the Gradio WebUI.
  >- Without `--api-url`, Gradio starts a reusable local `mineru-api`; with `--api-url`, it reuses an existing local or remote service.
  >- `--enable-vlm-preload true` makes Gradio start its local `mineru-api` during WebUI startup and wait for VLM preload to finish. It is ignored when `--api-url` points to an existing service.
  >- The WebUI currently accepts `PDF`, image, `DOCX`, `PPTX`, and `XLSX` uploads.

- Use `mineru-router` for multi-service / multi-GPU orchestration:
  ```bash
  mineru-router --host 0.0.0.0 --port 8002 --local-gpus auto --worker-tier standard
  ```
  >[!TIP]
  >
  >- `mineru-router` and `mineru-kit router` expose the complete `/v1/*` API and no longer expose `/tasks` or `/file_parse`.
  >- Repeat `--upstream-url` to aggregate multiple existing V1 api-server services, or use `--local-gpus` to launch `mineru-kit api-server` workers automatically.
  >- Use `--preload-models` for router-managed workers; remote upstreams keep their own startup configuration.
  >- Unknown model-engine arguments are not forwarded by Router.
  >- It is intended for advanced multi-service, multi-GPU, and unified-entry deployments.

- Using `http-client/server` method:
  ```bash
  # Start openai compatible server (requires vllm or lmdeploy environment)
  mineru-openai-server --port 30000
  ``` 
  >[!TIP]
  >In another terminal, connect to openai server via http client
  > ```bash
  > mineru -p <input_path> -o <output_path> -b hybrid-http-client -u http://127.0.0.1:30000
  > ```
  >`hybrid-http-client` requires local pipeline dependencies such as `mineru[pipeline]` and `torch`.
  >Legacy `vlm-http-client` input is accepted for compatibility and maps to `hybrid-http-client` with `--effort high`.

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

MinerU is now ready to use out of the box, but also supports extending functionality through configuration files. Legacy tool options such as LaTeX delimiters and LLM-aided title hierarchy still use `mineru.json` in your user directory. Model storage and model source settings use `config.yaml`; see [Model Source Documentation](./model_source.md).

Here are some available configuration options:  

- `models-dir`: 
    * Used to specify local model storage directory
    * Please specify model directories for the local lightweight model bundle (`models-dir.pipeline`) and the VLM bundle (`models-dir.vlm`) separately.
    * After specifying the directory, you can use local models by configuring the environment variable `export MINERU_MODEL_SOURCE=local`.
