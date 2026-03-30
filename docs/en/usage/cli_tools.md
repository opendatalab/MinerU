# Command Line Tools Usage Instructions

## View Help Information
To view help information for MinerU command line tools, you can use the `--help` parameter. Here are help information examples for various command line tools:
```bash
mineru --help
Usage: mineru [OPTIONS]

Options:
  -v, --version                   Show version and exit
  -p, --path PATH                 Input file path or directory (required)
  -o, --output PATH               Output directory (required)
  --api-url TEXT                  MinerU FastAPI base URL; if omitted, `mineru` starts a temporary local `mineru-api`
  -m, --method [auto|txt|ocr]     Parsing method: auto (default), txt, ocr (pipeline and hybrid* backend only)
  -b, --backend [pipeline|hybrid-auto-engine|hybrid-http-client|vlm-auto-engine|vlm-http-client]
                                  Parsing backend (default: hybrid-auto-engine)
  -l, --lang [ch|ch_server|ch_lite|en|korean|japan|chinese_cht|ta|te|ka|th|el|latin|arabic|east_slavic|cyrillic|devanagari]
                                  Specify document language (improves OCR accuracy, pipeline and hybrid* backend only)
  -u, --url TEXT                  OpenAI-compatible backend URL passed through to the server when using http-client
  -s, --start INTEGER             Starting page number for parsing (0-based)
  -e, --end INTEGER               Ending page number for parsing (0-based)
  -f, --formula BOOLEAN           Enable formula parsing (default: enabled)
  -t, --table BOOLEAN             Enable table parsing (default: enabled)
  --help                          Show help information
```
> [!TIP]
> `mineru` currently supports local `PDF`, image, and `DOCX` file or directory inputs.

```bash
mineru-api --help
Usage: mineru-api [OPTIONS]

Options:
  --host TEXT     Server host (default: 127.0.0.1)
  --port INTEGER  Server port (default: 8000)
  --reload        Enable auto-reload (development mode)
  --enable-vlm-preload BOOLEAN
                  Preload the local VLM model during mineru-api startup.
  --help          Show this message and exit.
```
```bash
mineru-gradio --help
Usage: mineru-gradio [OPTIONS]

Options:
  --enable-example BOOLEAN        Enable example files for input. The example
                                  files to be input need to be placed in the
                                  `example` folder within the directory where
                                  the command is currently executed.
  --enable-http-client BOOLEAN    Enable http-client backend to link openai-
                                  compatible servers.
  --enable-api BOOLEAN            Enable gradio API for serving the
                                  application.
  --max-convert-pages INTEGER     Set the maximum number of pages to convert
                                  from PDF to Markdown.
  --server-name TEXT              Set the server name for the Gradio app.
  --server-port INTEGER           Set the server port for the Gradio app.
  --api-url TEXT                  MinerU FastAPI base URL. If omitted, gradio
                                  starts a reusable local mineru-api service.
  --enable-vlm-preload BOOLEAN    Preload the local VLM model when gradio
                                  starts a local mineru-api service.
  --latex-delimiters-type [a|b|all]
                                  Set the type of LaTeX delimiters to use in
                                  Markdown rendering: 'a' for type '$', 'b' for
                                  type '()[]', 'all' for both types.
  --help                          Show this message and exit.
```
```bash
mineru-router --help
Usage: mineru-router [OPTIONS]

Options:
  --host TEXT             Server host (default: 127.0.0.1)
  --port INTEGER          Server port (default: 8002)
  --reload                Enable auto-reload (development mode)
  --upstream-url TEXT     Existing MinerU FastAPI base URL; repeat to add more
  --local-gpus TEXT       Local GPU workers to launch: auto, none, or CSV such
                          as 0,1,2
  --worker-host TEXT      Host for router-managed workers (default: 127.0.0.1)
  --enable-vlm-preload BOOLEAN
                          Preload the local VLM model in router-managed
                          mineru-api workers.
  --help                  Show this message and exit.
```

## Environment Variables Description

> [!NOTE]
> Starting from this version, `mineru` is an orchestration client built on top of `mineru-api`:
> - Without `--api-url`, the CLI launches a temporary local `mineru-api`
> - With `--api-url`, the CLI connects to that FastAPI service directly
> - `--url` is no longer the MinerU API address; it is the OpenAI-compatible backend URL used by server-side `vlm/hybrid-http-client`

Some parameters of MinerU command line tools have equivalent environment variable configurations. Generally, environment variable configurations have higher priority than command line parameters and take effect across all command line tools.
Here are the environment variables and their descriptions:
  
- `MINERU_TOOLS_CONFIG_JSON`: 
    * Used to specify configuration file path
    * defaults to `mineru.json` in user directory, can specify other configuration file paths through environment variables.
  
- `MINERU_FORMULA_ENABLE`:
    * Used to enable formula parsing
    * defaults to `true`, can be set to `false` through environment variables to disable formula parsing.
  
- `MINERU_FORMULA_CH_SUPPORT`:
    * Used to enable Chinese formula parsing optimization (experimental feature)
    * Default is `false`, can be set to `true` via environment variable to enable Chinese formula parsing optimization.
    * Only effective for `pipeline` backend.
  
- `MINERU_TABLE_ENABLE`:
    * Used to enable table parsing
    * Default is `true`, can be set to `false` via environment variable to disable table parsing.

- `MINERU_TABLE_MERGE_ENABLE`:
    * Used to enable table merging functionality
    * Default is `true`, can be set to `false` via environment variable to disable table merging functionality.

- `MINERU_PDF_RENDER_TIMEOUT`:
    * Used to set the timeout (in seconds) for rendering PDFs to images.
    * Default is `300` seconds; you can set a different value via an environment variable to adjust the rendering timeout.
    * Only effective on Linux and macOS systems.

- `MINERU_PDF_RENDER_THREADS`:
    * Used to set the number of threads used when rendering PDFs to images.
    * Default is `4`; you can set a different value via an environment variable to adjust the number of threads for image rendering.
    * Only effective on Linux and macOS systems.

- `MINERU_PROCESSING_WINDOW_SIZE`:
    * Used to control the processing window size, which affects memory use and throughput on large-document workloads.
    * Default is `64`; set it to another positive integer when needed.

- `MINERU_API_MAX_CONCURRENT_REQUESTS`:
    * Used to control the maximum concurrent requests handled by `mineru-api` or router-managed workers.
    * Default is `3`, and it must be a positive integer.

- `MINERU_API_ENABLE_FASTAPI_DOCS`:
    * Used to control whether FastAPI documentation endpoints such as `/docs`, `/openapi.json`, and `/redoc` are enabled.
    * Default is `true`.

- `MINERU_API_OUTPUT_ROOT`:
    * Used to configure the root output directory for `mineru-api`.
    * Default is `./output` under the current working directory.

- `MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS`:
    * Used to control how long CLI tools wait for a locally started `mineru-api` to become healthy.
    * Default is `300` seconds.
    * Applies to temporary local API startup in `mineru`, preload startup in `mineru-gradio`, and router-managed local workers.

- `MINERU_API_TASK_RETENTION_SECONDS`:
    * Used to set how long completed or failed tasks are retained, in seconds.
    * Default is `86400` seconds (24 hours).

- `MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS`:
    * Used to set the cleanup polling interval for expired tasks, in seconds.
    * Default is `300` seconds (5 minutes).

- `MINERU_INTRA_OP_NUM_THREADS`:
    * Used to set the intra_op thread count for ONNX models, affects the computation speed of individual operators
    * Default is `-1` (auto-select), can be set to other values via environment variable to adjust the thread count.

- `MINERU_INTER_OP_NUM_THREADS`:
    * Used to set the inter_op thread count for ONNX models, affects the parallel execution of multiple operators
    * Default is `-1` (auto-select), can be set to other values via environment variable to adjust the thread count.

- `MINERU_HYBRID_BATCH_RATIO`:
    * Used to set the batch ratio for small model processing in `hybrid-*` backends.
    * Commonly used in `hybrid-http-client`, it allows adjusting the VRAM usage of a single client by controlling the batch ratio of small models.
    * Single Client VRAM Size | MINERU_HYBRID_BATCH_RATIO
      ------------------------|--------------------------
      <= 6   GB               | 8
      <= 4   GB               | 4
      <= 3   GB               | 2
      <= 2   GB               | 1

- `MINERU_HYBRID_FORCE_PIPELINE_ENABLE`:
    * Used to force the text extraction part in `hybrid-*` backends to be processed using small models.
    * Defaults to `false`. Can be set to `true` via environment variable to enable this feature, thereby reducing hallucinations in certain extreme cases.

- `MINERU_VL_MODEL_NAME`:
    * Used to specify the model name for the vlm/hybrid backend, allowing you to designate the model required for MinerU to run when multiple models exist on a remote openai-server.

- `MINERU_VL_API_KEY`:
    * Used to specify the API Key for the vlm/hybrid backend, enabling authentication on the remote openai-server.
