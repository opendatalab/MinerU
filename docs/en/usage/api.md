# API Calls or Visual Invocation

1. Directly invoke using Python API: [Python Invocation Example](https://github.com/opendatalab/MinerU/blob/master/demo/demo.py)
2. Invoke using FastAPI:
   ```bash
   mineru-api --host 127.0.0.1 --port 8000
   ```
   Visit http://127.0.0.1:8000/docs in your browser to view the API documentation.

3. Use Gradio WebUI or Gradio API:
   ```bash
   # Using pipeline/vlm-transformers/vlm-sglang-client backend
   mineru-gradio --server-name 127.0.0.1 --server-port 7860
   # Or using vlm-sglang-engine/pipeline backend
   mineru-gradio --server-name 127.0.0.1 --server-port 7860 --enable-sglang-engine true
   ```
   Access http://127.0.0.1:7860 in your browser to use the Gradio WebUI, or visit http://127.0.0.1:7860/?view=api to use the Gradio API.


> [!TIP]  
> - Below are some suggestions and notes for using the sglang acceleration mode:  
> - The sglang acceleration mode currently supports operation on Turing architecture GPUs with a minimum of 8GB VRAM, but you may encounter VRAM shortages on GPUs with less than 24GB VRAM. You can optimize VRAM usage with the following parameters:  
>   - If running on a single GPU and encountering VRAM shortage, reduce the KV cache size by setting `--mem-fraction-static 0.5`. If VRAM issues persist, try lowering it further to `0.4` or below.  
>   - If you have more than one GPU, you can expand available VRAM using tensor parallelism (TP) mode: `--tp-size 2`  
> - If you are already successfully using sglang to accelerate VLM inference but wish to further improve inference speed, consider the following parameters:  
>   - If using multiple GPUs, increase throughput using sglang's multi-GPU parallel mode: `--dp-size 2`  
>   - You can also enable `torch.compile` to accelerate inference speed by about 15%: `--enable-torch-compile`  
> - For more information on using sglang parameters, please refer to the [sglang official documentation](https://docs.sglang.ai/backend/server_arguments.html#common-launch-commands)  
> - All sglang-supported parameters can be passed to MinerU via command-line arguments, including those used with the following commands: `mineru`, `mineru-sglang-server`, `mineru-gradio`, `mineru-api`

> [!TIP]  
> - In any case, you can specify visible GPU devices at the start of a command line by adding the `CUDA_VISIBLE_DEVICES` environment variable. For example:  
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 mineru -p <input_path> -o <output_path>
>   ```
> - This method works for all command-line calls, including `mineru`, `mineru-sglang-server`, `mineru-gradio`, and `mineru-api`, and applies to both `pipeline` and `vlm` backends.  
> - Below are some common `CUDA_VISIBLE_DEVICES` settings:  
>   ```bash
>   CUDA_VISIBLE_DEVICES=1 Only device 1 will be seen
>   CUDA_VISIBLE_DEVICES=0,1 Devices 0 and 1 will be visible
>   CUDA_VISIBLE_DEVICES="0,1" Same as above, quotation marks are optional
>   CUDA_VISIBLE_DEVICES=0,2,3 Devices 0, 2, 3 will be visible; device 1 is masked
>   CUDA_VISIBLE_DEVICES="" No GPU will be visible
>   ```
> - Below are some possible use cases:  
>   - If you have multiple GPUs and need to specify GPU 0 and GPU 1 to launch 'sglang-server' in multi-GPU mode, you can use the following command:  
>   ```bash
>   CUDA_VISIBLE_DEVICES=0,1 mineru-sglang-server --port 30000 --dp-size 2
>   ```
>   - If you have multiple GPUs and need to launch two `fastapi` services on GPU 0 and GPU 1 respectively, listening on different ports, you can use the following commands:  
>   ```bash
>   # In terminal 1
>   CUDA_VISIBLE_DEVICES=0 mineru-api --host 127.0.0.1 --port 8000
>   # In terminal 2
>   CUDA_VISIBLE_DEVICES=1 mineru-api --host 127.0.0.1 --port 8001
>   ```

---
