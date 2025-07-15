# Advanced Command Line Parameters

## SGLang Acceleration Parameter Optimization

### Memory Optimization Parameters
> [!TIP]
> SGLang acceleration mode currently supports running on Turing architecture graphics cards with a minimum of 8GB VRAM, but graphics cards with <24GB VRAM may encounter insufficient memory issues. You can optimize memory usage with the following parameters:
> 
> - If you encounter insufficient VRAM when using a single graphics card, you may need to reduce the KV cache size with `--mem-fraction-static 0.5`. If VRAM issues persist, try reducing it further to `0.4` or lower.
> - If you have two or more graphics cards, you can try using tensor parallelism (TP) mode to simply expand available VRAM: `--tp-size 2`

### Performance Optimization Parameters
> [!TIP]
> If you can already use SGLang normally for accelerated VLM model inference but still want to further improve inference speed, you can try the following parameters:
> 
> - If you have multiple graphics cards, you can use SGLang's multi-card parallel mode to increase throughput: `--dp-size 2`
> - You can also enable `torch.compile` to accelerate inference speed by approximately 15%: `--enable-torch-compile`

### Parameter Passing Instructions
> [!TIP]
> - All officially supported SGLang parameters can be passed to MinerU through command line arguments, including the following commands: `mineru`, `mineru-sglang-server`, `mineru-gradio`, `mineru-api`
> - If you want to learn more about `sglang` parameter usage, please refer to the [SGLang official documentation](https://docs.sglang.ai/backend/server_arguments.html#common-launch-commands)

## GPU Device Selection and Configuration

### CUDA_VISIBLE_DEVICES Basic Usage
> [!TIP]
> - In any situation, you can specify visible GPU devices by adding the `CUDA_VISIBLE_DEVICES` environment variable at the beginning of the command line. For example:
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - This specification method is effective for all command line calls, including `mineru`, `mineru-sglang-server`, `mineru-gradio`, and `mineru-api`, and applies to both `pipeline` and `vlm` backends.

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
> - If you have multiple graphics cards and need to specify cards 0 and 1, using multi-card parallelism to start `sglang-server`, you can use the following command:
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1 mineru-sglang-server --port 30000 --dp-size 2
>   ```
> 
> - If you have multiple GPUs and need to specify GPU 0–3, and start the `sglang-server` using multi-GPU data parallelism and tensor parallelism, you can use the following command:
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1,2,3 mineru-sglang-server --port 30000 --dp-size 2 --tp-size 2
>   ```
>       
> - If you have multiple graphics cards and need to start two `fastapi` services on cards 0 and 1, listening on different ports respectively, you can use the following commands:
>   ```bash
>   # In terminal 1
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # In terminal 2
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```
