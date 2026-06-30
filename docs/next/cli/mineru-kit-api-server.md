# mineru-kit api-server

状态: Draft
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、self-hosted 边界、与 doclib 的协作和参数契约
非目标: 统一 REST API 的字段级定义；模型服务内部实现
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru-kit api-server` 是未来唯一正式的 parse-server 启动入口。它启动一个 self-hosted parse server。一个进程只服务一个 tier，可以是 `standard` 或 `pro`。

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
- 负责模型下载、预热、重试和退避。

## 3. Tier 能力发现

api-server 必须提供能力发现接口，让 doclib 或客户端知道当前服务支持哪个 tier。

默认选择策略依赖该能力发现：

| api-server 支持 | 默认选择结果 |
|----------------|-------------|
| `standard` | `standard` |
| `pro` | `pro` |

api-server 不应把默认选择解析为 `flash`。

如果用户需要同时提供 `standard` 和 `pro`，应启动两个 api-server 进程，由 doclib 或上层配置分别管理地址和 tier。

## 4. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户自行启动，doclib 或其它客户端连接指定 URL
- managed：由 `mineru server` / doclib 在运行时拉起和停止 parse-server 进程

managed 是生命周期管理方式，不是用户直接执行的命令模式。

## 5. Usage

api-server 启动时应优先使用 `--tier`：

```bash
mineru-kit api-server --tier standard --port 15981
mineru-kit api-server --tier pro --port 15982
mineru-kit api-server --tier pro --effort high --language en --ocr-mode ocr --disable-table
```

`--tier` 会选择该 tier 的默认 backend。高级部署者可以同时传 `--tier` 和 `--backend`，用于选择具体实现；如果二者不兼容，启动应报错。

`--tier` 默认值是 `standard`。

启动完成后，HTTP API 不暴露 backend。`GET /v1/tiers` 也不新增 backend 字段；调用方如需推断实现，只能从 `current_model` 做弱推断。

`--backend` 可选值应使用 parser backend 名称，具体集合以 parser backend 常量为准。

裸 `vlm` / `hybrid` 不是合法的 api-server 启动 backend；它们只可作为 Middle JSON 来源标记或内部分类概念。

正式参数分层：

### 稳定公开参数

- host / port
- tier
- API key

### 稳定解析参数

- language
- ocr-mode
- effort
- disable-table / disable-formula / disable-image-analysis
- concurrency
- upload-dir
- url-timeout
- max-wait

### 专家参数

- backend

`--reload` 不进入正式命令设计。

本地 api-server 默认监听 loopback。它可以通过 `--api-key` 设置固定 API Key；默认不设置 API Key。设置后，客户端必须发送 `Authorization: Bearer <api-key>`。

## 6. API 覆盖范围

`mineru-kit api-server` 目标是实现 v1 API（非 doclib API）中的绝大多数 path。

当前明确排除：

- chat 的两个 path

同时明确不实现：

- doclib 的 `/docs`
- `/parses`
- `/search`
- `/invalidate`

完整设计背景见 [ADR-0017](../decisions/0017-mineru-kit-api-server-command.md)。
