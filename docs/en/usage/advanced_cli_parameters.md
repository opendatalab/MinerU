# Advanced Command Line Parameters

## Pass-through of inference engine parameters

### vllm Acceleration Parameter Optimization
> [!TIP]
> If you can already use vllm normally for accelerated VLM model inference but still want to further improve inference speed, you can try the following parameters:
> 
> - If you have multiple graphics cards, you can use vllm's multi-card parallel mode to increase throughput: `--data-parallel-size 2`

### Parameter Passing Instructions
> [!TIP]
> - All officially supported vllm/lmdeploy parameters can be passed to MinerU through command line arguments, including the following commands: `mineru`, `mineru-openai-server`, `mineru-gradio`, `mineru-api`
> - If you want to learn more about `vllm` parameter usage, please refer to the [vllm official documentation](https://docs.vllm.ai/en/latest/cli/serve.html)
> - If you want to learn more about `lmdeploy` parameter usage, please refer to the [lmdeploy official documentation](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html)

## GPU Device Selection and Configuration

### CUDA_VISIBLE_DEVICES Basic Usage
> [!TIP]
> - In any situation, you can specify visible GPU devices by adding the `CUDA_VISIBLE_DEVICES` environment variable at the beginning of the command line. For example:
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - This specification method is effective for all command line calls, including `mineru`, `mineru-openai-server`, `mineru-gradio`, and `mineru-api`, and applies to both `pipeline` and `vlm` backends.

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
> - If you have multiple graphics cards and need to specify cards 0 and 1, using multi-card parallelism to start `openai-server`, you can use the following command:
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1 mineru-openai-server --engine vllm --port 30000 --data-parallel-size 2
>   ```
>       
> - If you have multiple graphics cards and need to start two `fastapi` services on cards 0 and 1, listening on different ports respectively, you can use the following commands:
>   ```bash
>   # In terminal 1
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # In terminal 2
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```
