# mineru-kit api-server

状态: Draft
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、self-hosted 边界、与 doclib 的协作和参数契约
非目标: 统一 REST API 的字段级定义；模型服务内部实现
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit api-server` 是正式的 self-hosted parse-server 启动入口。当前一个进程可以通过重复 `--tier` 暴露一个或多个 tier；不传 `--tier` 时暴露 `flash`、`medium`、`high`、`xhigh`。

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

裸 api-server 的请求默认 tier 由启动配置决定：

| api-server 启动 tier | 请求未指定 tier 时 |
|---------------------|--------------------|
| 包含 `high` | `high` |
| 不包含 `high`，但包含 `xhigh` | `xhigh` |
| 不包含 `high` / `xhigh`，但包含 `medium` | `medium` |
| 只包含 `flash` | 返回 `quality_tier_unavailable`，除非请求显式传 `tier=flash` |
| 未传 `--tier` | 暴露 `flash`、`medium`、`high`、`xhigh`，默认 `high` |

因此，如果只以 `--tier flash` 启动裸 api-server，请求未指定 tier 时不应静默使用 `flash`。需要 `flash` 时调用方必须显式传 `tier=flash`；非 PDF/image 文件的批量归一规则见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。

如果用户需要在同一端口提供多个 tier，可以重复 `--tier`；如果不同 tier 需要不同硬件、并发或生命周期策略，则启动多个 api-server 进程并由 doclib 或上层配置分别管理 URL。

## 4. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户自行启动，doclib 或其它客户端连接指定 URL
- managed：由 `mineru server` / doclib 在运行时拉起和停止 parse-server 进程

managed 是生命周期管理方式，不是用户直接执行的命令模式。

## 5. Usage

api-server 启动时可使用 `--tier` 限定暴露档位；该选项可重复：

```bash
mineru-kit api-server --tier medium --port 16580
mineru-kit api-server --tier high --port 15982
mineru-kit api-server --tier medium --tier high --port 8000
mineru-kit api-server --tier high --language en --ocr-mode ocr --disable-image-analysis
```

未传 `--tier` 时暴露 `flash`、`medium`、`high`、`xhigh`；PDF/image 请求未指定 tier 时默认 `high`。

启动完成后，HTTP API 不暴露 backend。`GET /v1/tiers` 也不新增 backend 字段；调用方如需推断实现，只能从 `current_model` 做弱推断。

裸 `vlm` / `hybrid` 不是合法的 api-server 启动 backend；它们只可作为 Middle JSON 来源标记或内部分类概念。

正式参数分层：

### 稳定公开参数

- host / port
- tier，可重复
- API key

### 稳定解析参数

- language
- ocr-mode
- disable-image-analysis
- concurrency
- upload-dir
- url-timeout
- allow-local-source
- max-inline-bytes
- allow-http-source

### 专家参数

- 当前 `mineru-kit api-server` 命令层不暴露 `--backend`

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
