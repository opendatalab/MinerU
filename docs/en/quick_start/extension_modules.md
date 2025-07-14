# MinerU Extension Modules Installation Guide
MinerU supports installing extension modules on demand based on different needs to enhance functionality or support specific model backends.

## Common Scenarios

### Core Functionality Installation
The `core` module is the core dependency of MinerU, containing all functional modules except `sglang`. Installing this module ensures the basic functionality of MinerU works properly.
```bash
uv pip install mineru[core]
```

---

### Using `sglang` to Accelerate VLM Model Inference
The `sglang` module provides acceleration support for VLM model inference, suitable for graphics cards with Turing architecture and later (8GB+ VRAM). Installing this module can significantly improve model inference speed.
In the configuration, `all` includes both `core` and `sglang` modules, so `mineru[all]` and `mineru[core,sglang]` are equivalent.
```bash
uv pip install mineru[all]
```
> [!TIP]
> If exceptions occur during installation of the complete package including sglang, please refer to the [sglang official documentation](https://docs.sglang.ai/start/install.html) to try to resolve the issue, or directly use the [Docker](./docker_deployment.md) deployment method.

---

### Installing Lightweight Client to Connect to sglang-server
If you need to install a lightweight client on edge devices to connect to `sglang-server`, you can install the basic mineru package, which is very lightweight and suitable for devices with only CPU and network connectivity.
```bash
uv pip install mineru
```

---

### Using Pipeline Backend on Outdated Linux Systems
If your system is too outdated to meet the dependency requirements of `mineru[core]`, this option can minimally meet MinerU's runtime requirements, suitable for old systems that cannot be upgraded and only need to use the pipeline backend.
```bash
uv pip install mineru[pipeline_old_linux]
```
