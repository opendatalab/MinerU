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

If you need to adjust parsing options through custom parameters, you can also check the more detailed [Command Line Tools Usage Instructions](./cli_tools.md) in the documentation.

## Advanced Usage via API, WebUI, http-client/server

- Direct Python API calls: [Python Usage Example](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)
- FastAPI calls:
  ```bash
  mineru-api --host 0.0.0.0 --port 8000
  ```
  >[!TIP]
  >Access `http://127.0.0.1:8000/docs` in your browser to view the API documentation.
- Start Gradio WebUI visual frontend:
  ```bash
  mineru-gradio --server-name 0.0.0.0 --server-port 7860
  ```
  >[!TIP]
  >
  >- Access `http://127.0.0.1:7860` in your browser to use the Gradio WebUI.

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

> [!NOTE]
> All officially supported `vllm/lmdeploy` parameters can be passed to MinerU through command line arguments, including the following commands: `mineru`, `mineru-openai-server`, `mineru-gradio`, `mineru-api`.
> We have compiled some commonly used parameters and usage methods for `vllm/lmdeploy`, which can be found in the documentation [Advanced Command Line Parameters](./advanced_cli_parameters.md).

## Extending MinerU Functionality with Configuration Files

MinerU is now ready to use out of the box, but also supports extending functionality through configuration files. You can edit `mineru.json` file in your user directory to add custom configurations.  

>[!IMPORTANT]
>The `mineru.json` file will be automatically generated when you use the built-in model download command `mineru-models-download`, or you can create it by copying the [configuration template file](https://github.com/opendatalab/MinerU/blob/master/mineru.template.json) to your user directory and renaming it to `mineru.json`.  

Here are some available configuration options:  

- `latex-delimiter-config`: 
    * Used to configure LaTeX formula delimiters
    * Defaults to `$` symbol, can be modified to other symbols or strings as needed.
  
- `llm-aided-config`:
    * Used to configure parameters for LLM-assisted title hierarchy
    * Compatible with all LLM models supporting `openai protocol`, defaults to using Alibaba Cloud Bailian's `qwen3-next-80b-a3b-instruct` model. 
    * You need to configure your own API key and set `enable` to `true` to enable this feature.
    * If your API provider does not support the `enable_thinking` parameter, please manually remove it.
        * For example, in your configuration file, the `llm-aided-config` section may look like:
          ```json
          "llm-aided-config": {
             "api_key": "your_api_key",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen3-next-80b-a3b-instruct",
             "enable_thinking": false,
             "enable": false
          }
          ```
        * To remove the `enable_thinking` parameter, simply delete the line containing `"enable_thinking": false`, resulting in:
          ```json
          "llm-aided-config": {
             "api_key": "your_api_key",
             "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen3-next-80b-a3b-instruct",
             "enable": false
          }
          ```
  
- `models-dir`: 
    * Used to specify local model storage directory
    * Please specify model directories for `pipeline` and `vlm` backends separately.
    * After specifying the directory, you can use local models by configuring the environment variable `export MINERU_MODEL_SOURCE=local`.
