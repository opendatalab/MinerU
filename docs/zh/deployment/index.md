# 部署与运维

## 选择运行方式 {#services}

| 部署目标 | 入口 | 核验方式 |
| --- | --- | --- |
| 在本机交互解析 | `mineru-kit webui`，默认托管本地解析 worker | 打开启动时打印的地址，按 [WebUI 指南](../usage/webui.md)转换一个文件。 |
| 为应用提供文档解析 | `mineru-kit api-server`，支持异步 Tasks | 访问服务的 `/v1/health` 和 `/v1/tiers`，再运行[单文件 HTTP 示例](../usage/http_api.md#curl-example)。 |
| 多 GPU 或多个现有解析服务共用一个地址 | [Router](router.md) | 向 Router 地址提交同一个 Tasks 示例。 |
| HTTP 上传单文件后直接收取附件 | 在已有 Tasks 服务前启动 [file-parse](../usage/file_parse.md) | 检查同步附件或 `202` 响应中的查询地址。 |
| 仅提供远程 VLM 推理 | `mineru-kit vlm-server` 或已有的 OpenAI 兼容模型服务 | 在解析端配置 `model.vlm.server_url`，用 Standard 或 Advanced 解析样本。[远程连接配置](../usage/model_source.md) |

## 准备硬件与模型 {#prepare}

1. 运行 `mineru-kit hardware info --json` 查看设备、运行时及已有校准记录。要获取推荐档位，运行 `mineru-kit hardware detect`；首次 ONNX 校准可能下载模型。[档位与运行环境](../usage/tiers.md)
2. 原生文档可直接使用基础包。部署模型推理时，按目标设备选择[扩展模块](../quick_start/extension_modules.md)或 [NVIDIA Docker 镜像](../quick_start/docker_deployment.md)。多 GPU 设备分配见 [GPU 与引擎参数](../usage/advanced_cli_parameters.md)。
3. 按部署使用的后端和引擎[下载、验证模型](../usage/model_source.md)。`mineru-kit models show` 展示实际选择和就绪状态，`models verify` 可以用退出码检查所需文件。离线运行前先完成下载和验证。
4. 启动服务并核对产物。服务的 `--tier` 是 PDF/图片的开放上限；原生文档使用 Flash。先用原生 HTML 核对请求与下载，再用代表实际负载的 PDF 或图片验证需要的推理档位。
