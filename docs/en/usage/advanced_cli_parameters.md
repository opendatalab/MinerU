# Advanced Command Line Parameters

## Pass-through of inference engine parameters

### Parameter Passing Instructions
> [!TIP]
> - All officially supported vllm/lmdeploy parameters can be passed to MinerU through command line arguments, including the following commands: `mineru`, `mineru-openai-server`, `mineru-gradio`, `mineru-api`, `mineru-router`
> - Command-line options support both `--foo value` and `--foo=value` forms
> - If you want to learn more about `vllm` parameter usage, please refer to the [vllm official documentation](https://docs.vllm.ai/en/latest/cli/serve.html)
> - If you want to learn more about `lmdeploy` parameter usage, please refer to the [lmdeploy official documentation](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html)

## GPU Device Selection and Configuration

### CUDA_VISIBLE_DEVICES Basic Usage
> [!TIP]
> - In any situation, you can specify visible GPU devices by adding the `CUDA_VISIBLE_DEVICES` environment variable at the beginning of the command line. For example:
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - This method works for all command-line invocations, including `mineru`, `mineru-openai-server`, `mineru-gradio`, `mineru-api`, and `mineru-router`, and it applies to both the `pipeline` and `vlm` backends.

### Common Device Configuration Examples
> [!TIP]
> Here are some common `CUDA_VISIBLE_DEVICES` setting examples:
>   ```bash
>   CUDA_VISIBLE_DEVICES=1  # Only device 1 will be seen
>   CUDA_VISIBLE_DEVICES=0,1  # Devices 0 and 1 will be visible
>   CUDA_VISIBLE_DEVICES="0,1"  # Same as above, quotation marks are optional
>   CUDA_VISIBLE_DEVICES=0,2,3  # Devices 0, 2, 3 will be visible; device 1 is masked
>   CUDA_VISIBLE_DEVICES=""  # No GPU will be visible
>   ```

## Practical Application Scenarios
> [!TIP]
> Here are some possible usage scenarios:
> 
> - If you have multiple GPUs and need to start two `openai-server` services on GPU 0 and GPU 1, each listening on a different port, you can use the following commands:
>   ```bash
>   # In terminal 1
>   CUDA_VISIBLE_DEVICES=0 mineru-openai-server --engine vllm --port 30000
>   # In terminal 2
>   CUDA_VISIBLE_DEVICES=1 mineru-openai-server --engine vllm --port 30001
>   ```
> 
> - If you have multiple GPUs and need to start two `fastapi` services on GPU 0 and GPU 1, each listening on a different port, you can use the following commands:
>   ```bash
>   # In terminal 1
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # In terminal 2
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```
> 
> - If you have multiple GPUs and want to use `router` to launch and manage `fastapi` services across four GPUs, you can use the following command:
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1,2,3 mineru-router --host 127.0.0.1 --port 8002
>   ```
