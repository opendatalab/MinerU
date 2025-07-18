# Using MinerU

## Quick Model Source Configuration
MinerU uses `huggingface` as the default model source. If users cannot access `huggingface` due to network restrictions, they can conveniently switch the model source to `modelscope` through environment variables:
```bash
export MINERU_MODEL_SOURCE=modelscope
```
For more information about model source configuration and custom local model paths, please refer to the [Model Source Documentation](./model_source.md) in the documentation.

## Quick Usage via Command Line
MinerU has built-in command line tools that allow users to quickly use MinerU for PDF parsing through the command line:
```bash
# Default parsing using pipeline backend
mineru -p <input_path> -o <output_path>
```
> [!TIP]
>- `<input_path>`: Local PDF/image file or directory
>- `<output_path>`: Output directory
>
> For more information about output files, please refer to [Output File Documentation](../reference/output_files.md).

> [!NOTE]
> The command line tool will automatically attempt cuda/mps acceleration on Linux and macOS systems. 
> Windows users who need cuda acceleration should visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to select the appropriate command for their cuda version to install acceleration-enabled `torch` and `torchvision`.


```bash
# Or specify vlm backend for parsing
mineru -p <input_path> -o <output_path> -b vlm-transformers
```
> [!TIP]
> The vlm backend additionally supports `sglang` acceleration. Compared to the `transformers` backend, `sglang` can achieve 20-30x speedup. You can check the installation method for the complete package supporting `sglang` acceleration in the [Extension Modules Installation Guide](../quick_start/extension_modules.md).

If you need to adjust parsing options through custom parameters, you can also check the more detailed [Command Line Tools Usage Instructions](./cli_tools.md) in the documentation.

## Advanced Usage via API, WebUI, sglang-client/server

- Direct Python API calls: [Python Usage Example](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)
- FastAPI calls:
  ```bash
  mineru-api --host 0.0.0.0 --port 8000
  ```
  >[!TIP]
  >Access `http://127.0.0.1:8000/docs` in your browser to view the API documentation.
- Start Gradio WebUI visual frontend:
  ```bash
  # Using pipeline/vlm-transformers/vlm-sglang-client backends
  mineru-gradio --server-name 0.0.0.0 --server-port 7860
  # Or using vlm-sglang-engine/pipeline backends (requires sglang environment)
  mineru-gradio --server-name 0.0.0.0 --server-port 7860 --enable-sglang-engine true
  ```
  >[!TIP]
  >
  >- Access `http://127.0.0.1:7860` in your browser to use the Gradio WebUI.
  >- Access `http://127.0.0.1:7860/?view=api` to use the Gradio API.
- Using `sglang-client/server` method:
  ```bash
  # Start sglang server (requires sglang environment)
  mineru-sglang-server --port 30000
  ``` 
  >[!TIP]
  >In another terminal, connect to sglang server via sglang client (only requires CPU and network, no sglang environment needed)
  > ```bash
  > mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://127.0.0.1:30000
  > ```

> [!NOTE]
> All officially supported sglang parameters can be passed to MinerU through command line arguments, including the following commands: `mineru`, `mineru-sglang-server`, `mineru-gradio`, `mineru-api`.
> We have compiled some commonly used parameters and usage methods for `sglang`, which can be found in the documentation [Advanced Command Line Parameters](./advanced_cli_parameters.md).

## Extending MinerU Functionality with Configuration Files

MinerU is now ready to use out of the box, but also supports extending functionality through configuration files. You can edit `mineru.json` file in your user directory to add custom configurations.  

>[!IMPORTANT]
>The `mineru.json` file will be automatically generated when you use the built-in model download command `mineru-models-download`, or you can create it by copying the [configuration template file](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json) to your user directory and renaming it to `mineru.json`.  

Here are some available configuration options:  

- `latex-delimiter-config`: Used to configure LaTeX formula delimiters, defaults to `$` symbol, can be modified to other symbols or strings as needed.
- `llm-aided-config`: Used to configure parameters for LLM-assisted title hierarchy, compatible with all LLM models supporting `openai protocol`, defaults to using Alibaba Cloud Bailian's `qwen2.5-32b-instruct` model. You need to configure your own API key and set `enable` to `true` to enable this feature.
- `models-dir`: Used to specify local model storage directory, please specify model directories for `pipeline` and `vlm` backends separately. After specifying the directory, you can use local models by configuring the environment variable `export MINERU_MODEL_SOURCE=local`.

