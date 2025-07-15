# MinerU 扩展模块安装指南
MinerU 支持根据不同需求，按需安装扩展模块，以增强功能或支持特定的模型后端。

## 常见场景

### 核心功能安装
`core` 模块是 MinerU 的核心依赖，包含了除`sglang`外的所有功能模块。安装此模块可以确保 MinerU 的基本功能正常运行。
```bash
uv pip install mineru[core]
```

---

### 使用`sglang`加速 VLM 模型推理
`sglang` 模块提供了对 VLM 模型推理的加速支持，适用于具有 Turing 及以后架构的显卡（8G 显存及以上）。安装此模块可以显著提升模型推理速度。
在配置中，`all`包含了`core`和`sglang`模块，因此`mineru[all]`和`mineru[core,sglang]`是等价的。
```bash
uv pip install mineru[all]
```
> [!TIP]
> 如在安装包含sglang的完整包过程中发生异常，请参考 [sglang 官方文档](https://docs.sglang.ai/start/install.html) 尝试解决，或直接使用 [Docker](./docker_deployment.md) 方式部署镜像。

---

### 安装轻量版client连接sglang-server使用
如果您需要在边缘设备上安装轻量版的 client 端以连接 `sglang-server`，可以安装mineru的基础包，非常轻量，适合在只有cpu和网络连接的设备上使用。
```bash
uv pip install mineru
```

---

### 在过时的linux系统上使用pipeline后端
如果您的系统过于陈旧，无法满足`mineru[core]`的依赖要求，该选项可以最低限度的满足 MinerU 的运行需求，适用于老旧系统无法升级且仅需使用 pipeline 后端的场景。
```bash
uv pip install mineru[pipeline_old_linux]
```