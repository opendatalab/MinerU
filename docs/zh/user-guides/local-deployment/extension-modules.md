# MinerU 扩展模块安装指南
MinerU 支持根据不同需求，按需安装扩展模块，以增强功能或支持特定的模型后端。

## 常见场景

### 核心功能安装
`core` 模块是 MinerU 的核心依赖，包含了除`vllm`外的所有功能模块。安装此模块可以确保 MinerU 的基本功能正常运行。
```bash
uv pip install mineru[core]
```

---

### 使用`vllm`加速 VLM 模型推理
`vllm` 模块提供了对 VLM 模型推理的加速支持，适用于具有 Turing 及以后架构的显卡（8G 显存及以上）。安装此模块可以显著提升模型推理速度。
在配置中，`all`包含了`core`和`vllm`模块，因此`mineru[all]`和`mineru[core,vllm]`是等价的。
```bash
uv pip install mineru[all]
```
> [!TIP]
> 如在安装包含vllm的完整包过程中发生异常，请参考 [vllm 官方文档](https://docs.vllm.ai/en/latest/getting_started/installation/index.html) 尝试解决，或直接使用 [Docker](./docker_deployment.md) 方式部署镜像。

---

### 安装轻量版client连接vllm-server使用
如果您需要在边缘设备上安装轻量版的 client 端以连接 `vllm-server`，可以安装mineru的基础包，非常轻量，适合在只有cpu和网络连接的设备上使用。
```bash
uv pip install mineru
```
