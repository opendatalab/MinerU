# MinerU Extension Modules Installation Guide
MinerU supports installing extension modules on demand based on different needs to enhance functionality or support specific model backends.

## Common Scenarios

### Core Functionality Installation
The `core` module is the core dependency of MinerU, containing all functional modules except `vllm`/`lmdeploy`. Installing this module ensures the basic functionality of MinerU works properly.
```bash
uv pip install "mineru[core]"
```

---

### Using `vllm` to Accelerate VLM Model Inference
> [!NOTE]
> `vllm` and `lmdeploy` have nearly identical VLM inference acceleration effects and usage methods. You can choose one of them to install and use based on your actual needs, but it is not recommended to install both modules simultaneously to avoid potential dependency conflicts.

The `vllm` module provides acceleration support for VLM model inference, suitable for graphics cards with Volta architecture and later (8GB+ VRAM). Installing this module can significantly improve model inference speed.

```bash
uv pip install "mineru[core,vllm]"
```
> [!TIP]
> If exceptions occur during installation of the extra package including vllm, please refer to the [vllm official documentation](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) to try to resolve the issue, or directly use the [Docker](./docker_deployment.md) deployment method.

---

### Using `lmdeploy` to Accelerate VLM Model Inference
> [!NOTE]
> `vllm` and `lmdeploy` have nearly identical VLM inference acceleration effects and usage methods. You can choose one of them to install and use based on your actual needs, but it is not recommended to install both modules simultaneously to avoid potential dependency conflicts.

The `lmdeploy` module provides acceleration support for VLM model inference, suitable for graphics cards with Volta architecture and later (8GB+ VRAM). Installing this module can significantly improve model inference speed.

```bash
uv pip install "mineru[core,lmdeploy]"
```
> [!TIP]
> If exceptions occur during installation of the extra package including lmdeploy, please refer to the [lmdeploy official documentation](https://lmdeploy.readthedocs.io/en/latest/get_started/installation.html) to try to resolve the issue.

---

### Installing Lightweight Client to Connect to OpenAI-compatible servers (for vlm-http-client mode)
If you need to install a lightweight client on edge devices to connect to an OpenAI-compatible server for using VLM mode, you can install the basic mineru package, which is very lightweight and suitable for devices with only CPU and network connectivity.
```bash
uv pip install mineru
mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://127.0.0.1:30000
```

---

### Installing Lightweight Client to Connect to OpenAI-compatible servers (for hybrid-http-client mode)
If you need to install a lightweight client on edge devices to connect to an OpenAI-compatible server for using hybrid mode, you can install the mineru pipeline extension package, which is relatively lightweight and can be used on devices with only CPU and network connectivity, while running faster on devices that support GPU acceleration.
```bash
uv pip install mineru[pipeline]
mineru -p <input_path> -o <output_path> -b hybrid-http-client -u http://127.0.0.1:30000
```