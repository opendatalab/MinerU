# mineru-kit vlm-server

状态: Implemented
读者: 服务部署者、核心开发者、VLM backend 集成开发者
范围: `mineru-kit vlm-server` 的定位、协议范围、与 `api-server` 的边界和命令参数
非目标: 完整 Parse API；通用聊天能力说明；底层 engine 私有参数表
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit vlm-server` 是未来唯一正式的本地 VLM 服务启动入口。

它部署的是与 `mineru.net` chat API 同类的 VLM 模型，但模型主要面向文档理解，经过专用微调，因此更适合：

- OCR
- 布局理解
- 页面理解
- 文档局部问答 / 提取

它不承担通用聊天能力承诺。

## 2. 与 api-server 的边界

`mineru-kit vlm-server` 可作为 `mineru-kit api-server` 的远程 VLM 后端。
API 服务通过 `--vlm-server-url` 或全局 `model.vlm.server_url` 连接它，内部使用
`http-client` 执行 Standard/Advanced 的 VLM 推理；本地 Hybrid 处理仍在 API 服务进程中执行。
连接、鉴权及模型名配置见 [api-server](mineru-kit-api-server.md#使用远程-mineru-vlm-服务)。

边界:

| 命令 | 主要职责 |
|------|----------|
| `mineru-kit api-server` | 完整文档解析 API，处理 Files / Uploads / Parse Jobs |
| `mineru-kit vlm-server` | VLM 推理服务，处理 OpenAI-compatible chat 请求 |

`vlm-server` 不处理：

- Parse Jobs
- Files / Uploads
- doclib API

## 3. 协议范围

当前稳定提供：

- `GET /health`（健康检查内容由 engine 定义）
- `GET /v1/models`
- `POST /v1/chat/completions`

当前不把 `/v1/responses` 作为稳定承诺。

某些底层 engine 版本可能支持 `/v1/responses`，但 `mineru-kit vlm-server` 不以它作为统一契约。

## 4. 协议兼容语义

`mineru-kit vlm-server` 兼容 OpenAI Chat Completions 协议。

这里的“兼容”表示：

- 请求 / 响应结构兼容
- 可以复用 OpenAI-style client 和 serving stack

不表示：

- 提供通用聊天产品语义
- 提供开放域通用助手能力

## 5. Usage

统一参数只有一个：

```bash
mineru-kit vlm-server --engine auto
mineru-kit vlm-server --engine vllm
mineru-kit vlm-server --engine lmdeploy
mineru-kit vlm-server --engine mlx
```

`--engine` 决定使用哪类底层 serving engine。当前合法值为 `auto`、`vllm`、`lmdeploy`、`mlx`；`auto` 按 vLLM、LMDeploy、MLX-VLM 的顺序选择已安装且可用的 engine。

除统一参数外，其余参数原样透传到底层 engine server。

## 6. 与旧入口的迁移关系

`mineru-kit vlm-server` 替代：

- `mineru-vllm-server`
- `mineru-lmdeploy-server`
- `mineru-openai-server`

未来正式入口统一为：

```bash
mineru-kit vlm-server
```

完整设计背景见 [ADR-0018](../decisions/0018-mineru-kit-vlm-server-command.md)。

## 7. Apple Silicon / MLX

安装 `mineru[full]`（需要 mineru-vl-utils 2.0.1+），使用 macOS 14+、arm64 和 `mlx-vlm>=0.7.0,<0.8.0`：

```bash
mineru-kit vlm-server --engine mlx --host 127.0.0.1 --port 8080
# 旧命令继续复用同一实现
mineru-openai-server --engine mlx --model /path/to/model --port 8080
mineru-kit api-server --vlm-server-url http://127.0.0.1:8080
```

MLX 默认监听 `127.0.0.1:8080`，未指定 `--model` 时使用 MinerU 默认 VLM 模型。
其余选项由 mlx-vlm 原生 CLI 解析，包括鉴权和生成配置；不支持的参数会报错。
服务复用上游的生命周期、连续批处理和流式响应。

新版直接采用上游 HTTP 协议：请求必须携带 `model`，模型标识从
`GET /v1/models` 获取；MinerU HTTP client 已自动执行该步骤。
兼容模型的标识可能是临时目录路径，不应跨服务重启保存该标识。
健康检查使用 `/health`；旧包装层的默认模型注入和自建路由别名已移除。
上游其他端点不属于 MinerU 的稳定接口承诺。

Qwen 模型通过 mineru-vl-utils 的公开路径准备接口生成独立兼容配置，
原始配置和权重不变，临时目录在服务退出后清理。MLX server 不修改上游函数。

若同一环境也运行 Gradio，请让依赖解析器同时解析 `mineru[full,gradio]`。
mlx-vlm 0.7.0 要求 Starlette 1.x，Gradio 6.8.0 与其不兼容；本次验收使用 Gradio 6.26.0。

## 8. 推理服务与 Parse API 的接口区别

vLLM、LMDeploy 和 MLX 的原生推理服务使用相同的核心路径：

| 接口 | vLLM | LMDeploy | MLX |
| --- | --- | --- | --- |
| 健康检查 | `GET /health` | `GET /health` | `GET /health` |
| 模型列表 | `GET /v1/models` | `GET /v1/models` | `GET /v1/models` |
| 多模态聊天 | `POST /v1/chat/completions` | `POST /v1/chat/completions` | `POST /v1/chat/completions` |

`mineru-kit api-server` 和 Router 提供的是完整 MinerU 解析协议，其健康检查为
`/v1/health`。Router 的 worker 需要完整 Parse API，不能直接指向上述原生推理服务。

路径一致不表示所有响应字段都相同：健康检查可能返回空响应或 engine 专用 JSON，
探针应优先判断 HTTP 状态码，不要求统一 `status` 字段。模型列表的 ID 和附加字段也由 engine 定义。
客户端应从模型列表获取 ID，并使用标准 `messages`、`image_url` 和聊天输出字段。

非核心接口不统一。例如 vLLM/LMDeploy 提供 `/v1/completions`，本次验证的
mlx-vlm 0.7.0 未提供该接口；MLX 还提供不带 `/v1` 的 `/models`、`/chat/completions` 别名。
不要将别名、metrics 响应格式、模型卸载、tokenize/encode、Responses 或私有采样参数视为跨 engine 契约。

核对依据：MLX 0.7.0 本机路由实测、
[vLLM serving 文档](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)、
[LMDeploy 0.17.0 管理路由](https://github.com/InternLM/lmdeploy/blob/v0.17.0/lmdeploy/serve/openai/endpoints/management.py)及同版本模型/聊天路由。

## 9. MLX 的并发边界

- `mlx-engine` 直接调用 `mlx_vlm.generate()`，同一客户端的生成仍保留互斥锁。
  0.7.0 实测去锁并发 2 会报 GPU stream 跨线程错误；server 支持并发不意味着此路径线程安全。
  `batch_predict` 使用公开 `BatchGenerator` 在锁内批处理，默认 batch 8；显式 batch 1 保留逐张路径。
- `http-client` 连接 MLX server 不使用本地生成锁，不存在 MLX 专用的并发 1 限制。
  上游 server 使用一个 GPU 工作线程调度并批处理多个请求；不需要增加 uvicorn workers。
- 可将 `model.vlm.max_concurrency` 设为 `8` 作为起点，再按图片尺寸、输出长度和内存调整。
  此项限制并行 HTTP 请求，不控制本地 engine 的生成线程数；不修改其他远程引擎的既有默认值。

例如让完整解析 API 通过并发 8 连接 MLX 推理服务：

```bash
mineru-kit api-server --vlm-server-url http://127.0.0.1:8080 --vlm-max-concurrency 8
```

这不会改变 MLX 本地 engine 的串行生成边界，也不会把 Parse API 的任务并发与 VLM 请求并发混为一谈。

本地 MLX batch 使用 `mlx_vlm.generate.BatchGenerator`，保留 MinerU 原有聊天模板与图文顺序。
不直接调用会重新组织聊天模板的 `batch_generate()`，也不替换上游函数。

批次先按有效采样配置和图片/纯文本模态分组，图片按像素数排序；输出按原始索引恢复。
默认 `batch_size=0` 解析为 8，显式 `batch_size=1` 可回退到逐张生成。
每批同时受样本数与 9,000,000 原始图片总像素预算限制；超预算单图独立处理，不改变图片分辨率。
这个预算不是严格的内存上限，长输出仍可能增加 KV cache；batch 的生成操作继续串行持锁。

直接使用 Python engine 时，可通过既有 `MinerUClient(..., batch_size=8)` 配置批次大小。
此参数不同于 HTTP 客户端的 `max_concurrency`，不新增 Parse API/CLI 参数。

900 万像素预算可容纳 8 张 1036×1036 的 layout 图片（总计 8,586,368 像素），
避免原 400 万预算将 layout 的 batch 4 拆成 3+1。
默认 batch 为 8；需要降低内存时可显式设置为 4、2 或 1。
超过预算仍自动拆批，生成锁继续生效；像素预算不等同于硬内存上限。

MLX server 默认传入 `--max-num-seqs 8`，最多八条序列参与活跃推理/连续批处理，额外请求排队。
显式 `--max-num-seqs` 优先于 `MLX_VLM_MAX_NUM_SEQS` 环境变量，环境变量优先于默认值。
上游 prefill 批次上限已为八，活跃序列限制也将实际 decode batch 限制为八；仍使用一个 GPU 工作线程。
这不改变其他推理引擎或通用 HTTP client 的全局默认值。
