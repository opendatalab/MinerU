# MinerU 扩展模块安装指南
MinerU 支持根据不同需求，按需安装扩展模块，以增强功能或支持特定的模型后端。

## 常见场景

### 核心功能安装
`core` 模块是 MinerU 的核心依赖，包含了除`vllm`/`lmdeploy`外的所有功能模块。安装此模块可以确保 MinerU 的基本功能正常运行。
```bash
uv pip install "mineru[core]"
```

---

### 使用`vllm`加速 VLM 模型推理
> [!NOTE]
> `vllm`和`lmdeploy`对vlm的推理加速效果和使用方式几乎相同，您可以根据实际情况选择其中之一进行安装和使用，但不建议同时安装这两个模块，以避免潜在的依赖冲突。

`vllm` 模块提供了对 VLM 模型推理的加速支持，适用于具有 Volta 及以后架构的显卡（8G 显存及以上）。安装此模块可以显著提升模型推理速度。
```bash
uv pip install "mineru[core,vllm]"
```
> [!TIP]
> 如在安装包含`vllm`的扩展包过程中发生异常，请参考 [vllm 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) 尝试解决，或直接使用 [Docker](./docker_deployment.md) 方式部署镜像。

---

### 使用`lmdeploy`加速 VLM 模型推理
> [!NOTE]
> `vllm`和`lmdeploy`对vlm的推理加速效果和使用方式几乎相同，您可以根据实际情况选择其中之一进行安装和使用，但不建议同时安装这两个模块，以避免潜在的依赖冲突。

`lmdeploy` 模块提供了对 VLM 模型推理的加速支持，适用于具有 Volta 及以后架构的显卡（8G 显存及以上）。安装此模块可以显著提升模型推理速度。
```bash
uv pip install "mineru[core,lmdeploy]"
```
> [!TIP]
> 如在安装包含`lmdeploy`的扩展包过程中发生异常，请参考 [lmdeploy 官方文档](https://lmdeploy.readthedocs.io/en/latest/get_started/installation.html) 尝试解决。

---

### 安装轻量版client连接兼容openai服务器使用 (适用vlm-http-client模式)
如果您需要在边缘设备上安装轻量版的 client 端以连接兼容 openai 接口的服务端来使用vlm模式，可以安装mineru的基础包，非常轻量，适合在只有cpu和网络连接的设备上使用。
```bash
uv pip install mineru
mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://127.0.0.1:30000
```

---

### 安装轻量版client连接兼容openai服务器使用 (适用hybrid-http-client模式)
如果您需要在边缘设备上安装轻量版的 client 端以连接兼容 openai 接口的服务端来使用hybrid模式，可以安装mineru的pipeline扩展包，相对较轻量，可以在只有cpu和网络连接的设备上使用，同时在支持gpu加速的设备上可以更快运行。
```bash
uv pip install "mineru[pipeline]"
mineru -p <input_path> -o <output_path> -b hybrid-http-client -u http://127.0.0.1:30000
```
