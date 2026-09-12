# Extension Modules

All examples target MinerU 4.x and run inside an activated virtual environment.

| Installation | Includes | Use case |
| --- | --- | --- |
| `mineru>=4.0,<5` | ONNX, llama.cpp, WebUI, SDK/API; Torch is automatic on Apple Silicon | Native documents, CPU small models, llama.cpp, or an existing service |
| `mineru[torch]>=4.0,<5` | Base package + Torch, Torchvision, Transformers, Accelerate | Torch small models with an independently selected VLM engine |
| `mineru[full]>=4.0,<5` | Torch extra + vLLM on Linux / LMDeploy on Windows | Higher-throughput serving on supported devices |

```bash
uv pip install -U "mineru[torch]>=4.0,<5"
```

Or install the platform-specific inference engine:

```bash
uv pip install -U "mineru[full]>=4.0,<5"
```

`all` composes `full`. `core`, `pipeline`, `vlm`, `vllm`, `lmdeploy`, `gradio`, and `mlx` are no longer MinerU 4.0 extras. Tier names are not extras either.

## Optional engine constraints

The current 4.0 dependency declarations are `torch>=2.7.0,<3`, `transformers>=5.10.1,<6`, Linux `vllm>=0.19.1,<0.29.0`, and Windows `lmdeploy>=0.17.0,<0.18`. Intersect the package's Python range with the wheels actually available for these dependencies. Check the [legacy platform policy](../usage/compatibility.md) before upgrading a vendor environment.

Apple Silicon defaults to llama.cpp for the VLM; `full` does not install MLX. Install and select MLX explicitly when needed:

```bash
uv pip install "mineru>=4.0,<5" "mlx-vlm>=0.7.0,<0.8.0"
mineru config set model.vlm.engine mlx
```

The WebUI uses Gradio 6 and bundled PDF.js preview assets; it does not require `gradio-pdf`. A separate UI environment can use the base package and connect with `mineru-kit webui --api-url http://127.0.0.1:8000`.

## Upgrade an existing environment

Stop that environment's MinerU services before upgrading with the original installer. Preserve the required extras, then check the version, dependencies, and model configuration. Downloads and inference must use the same `model.small_backend` and `model.vlm.engine`; see [Model Source](../usage/model_source.md).
