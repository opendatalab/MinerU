# mineru-kit api-server

状态: Draft
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、用途、与 doclib 的协作和 tier 能力发现
非目标: 统一 REST API 的字段级定义；模型服务内部实现
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru-kit api-server` 启动一个无状态解析服务，用作 local parse-server 或自部署 parse-server。一个进程只服务一个 tier，可以是 `standard` 或 `pro`。

`mineru doclib` 可以通过 HTTP 调用它执行解析任务。

## 2. 与 doclib 的协作

```text
mineru CLI
  -> doclib
    -> local parse-server (`mineru-kit api-server`)
```

doclib 负责：

- 文件入库。
- SHA256 缓存。
- 任务排队。
- 解析产物存储。
- 搜索索引。
- parse-server 健康检查。

api-server 负责：

- 暴露可用 tier。
- 接收解析请求。
- 执行模型解析。
- 返回解析产物。
- 在 managed 模式下负责模型下载、预热、重试和退避。

## 3. Tier 能力发现

api-server 必须提供能力发现接口，让 doclib 或客户端知道当前服务支持哪个 tier。

默认选择策略依赖该能力发现：

| api-server 支持 | 默认选择结果 |
|----------------|-------------|
| `standard` | `standard` |
| `pro` | `pro` |

api-server 不应把默认选择解析为 `flash`。

如果用户需要同时提供 `standard` 和 `pro`，应启动两个 api-server 进程，由 doclib 或上层配置分别管理地址和 tier。

## 4. 部署模式

| 模式 | 说明 |
|------|------|
| managed | 由 `mineru server` 启停 local parse-server |
| self-hosted | 用户自行启动，doclib 连接指定 URL |
| remote-compatible | 用作远端 API 的兼容实现 |

managed 模式下，`mineru server` 负责拉起和停止 api-server 进程，但不负责模型生命周期细节。模型下载、预热、失败重试和退避属于 api-server 自身职责。

## 5. Usage

api-server 启动时应优先使用 `--tier`：

```bash
mineru-kit api-server --tier standard --port 15981
mineru-kit api-server --tier pro --port 15982
mineru-kit api-server --tier standard --language en --ocr-mode ocr --disable-table
```

`--tier` 会选择该 tier 的默认 backend。高级部署者可以同时传 `--tier` 和 `--backend`，用于在同一 tier 存在多个 backend 实现时选择具体实现；如果二者不兼容，启动应报错。启动完成后，HTTP API 仍只暴露 `tier`，不暴露 backend。

`--backend` 可选值应使用 parser backend 名称，具体集合以 parser backend 常量为准。

裸 `vlm` / `hybrid` 不是合法的 api-server 启动 backend；它们只可作为 Middle JSON 来源标记或内部分类概念。

具体参数仍以底稿为准，稳定后应覆盖：

- host / port
- supported tier
- backend 高级覆盖
- language
- ocr-mode
- disable-table / disable-formula / disable-image-analysis
- model path
- device
- concurrency
- API key
- health endpoint
- tiers endpoint: `GET /v1/tiers`

本地 api-server 默认监听 loopback。它可以通过 `--api-key` 设置固定 API Key；默认不设置 API Key。设置后，客户端必须发送 `Authorization: Bearer <api-key>`。

## 未决问题

api-server 参数稳定性等级、默认端口和多进程发现方式，集中维护在 [开放问题清单](../open-questions.md)。
