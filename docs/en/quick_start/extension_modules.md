# MinerU Extension Modules Installation Guide
MinerU supports installing extension modules on demand based on different needs to enhance functionality or support specific model backends.

## Common Scenarios

### Core Functionality Installation
The `core` module is the core dependency of MinerU, containing all functional modules except `vllm`. Installing this module ensures the basic functionality of MinerU works properly.
```bash
uv pip install mineru[core]
```

---

### Using `vllm` to Accelerate VLM Model Inference
The `vllm` module provides acceleration support for VLM model inference, suitable for graphics cards with Turing architecture and later (8GB+ VRAM). Installing this module can significantly improve model inference speed.
In the configuration, `all` includes both `core` and `vllm` modules, so `mineru[all]` and `mineru[core,vllm]` are equivalent.
```bash
uv pip install mineru[all]
```
> [!TIP]
> If exceptions occur during installation of the complete package including vllm, please refer to the [vllm official documentation](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) to try to resolve the issue, or directly use the [Docker](./docker_deployment.md) deployment method.

---

### Installing Lightweight Client to Connect to vllm-server
If you need to install a lightweight client on edge devices to connect to `vllm-server`, you can install the basic mineru package, which is very lightweight and suitable for devices with only CPU and network connectivity.
```bash
uv pip install mineru
```
