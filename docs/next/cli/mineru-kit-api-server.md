# mineru-kit api-server

状态: Implemented
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、self-hosted 边界、与 doclib 的协作和参数契约
非目标: 统一 REST API 的字段级定义；模型服务内部实现
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit api-server` 是正式的 self-hosted parse-server 启动入口。`--tier` 是单值启动能力上限，只接受 `flash`、`basic` 或 `standard`；不传时默认以 `standard` 启动，并暴露全部四个请求 tier。

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

| api-server 启动配置 | `/v1/tiers` | 请求未指定 tier 时 |
|---------------------|-------------|--------------------|
| `--tier flash` | `flash` | 返回 `quality_tier_unavailable`，除非请求显式传 `tier=flash` |
| `--tier basic` | `flash`、`basic` | `basic` |
| `--tier basic --no-flash` | `basic` | `basic` |
| `--tier basic --no-advanced` | `flash`、`basic` | `basic` |
| `--tier standard` 或未传 `--tier` | `flash`、`basic`、`standard`、`advanced` | `standard` |
| `--tier standard --no-flash` | `basic`、`standard`、`advanced` | `standard` |
| `--tier standard --no-advanced` | `flash`、`basic`、`standard` | `standard` |
| `--tier standard --no-flash --no-advanced` | `basic`、`standard` | `standard` |

因此，如果只以 `--tier flash` 启动裸 api-server，请求未指定 tier 时不应静默使用 `flash`。需要 `flash` 时调用方必须显式传 `tier=flash`；非 PDF/image 文件的批量归一规则见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。

`--no-flash` 会同时关闭 Flash 能力发现和执行。显式 Flash 请求以及 OFD/EPUB/Office/HTML/CSV 等必须归一到 Flash 的输入都会被拒绝。`--tier flash --no-flash` 因为没有可用能力而启动失败。

`--no-advanced` 会同时关闭 Advanced 能力发现和执行：`GET /v1/tiers` 不再发布 `advanced`，显式 Advanced 请求返回该 tier 不可用。Advanced 不是启动 tier，因此它与 `--tier flash` 或 `--tier basic` 组合时不会改变原有能力；Standard 仍保留自身所需的共享模型。

如果不同启动能力需要不同硬件、并发或生命周期策略，应启动多个 api-server 进程并由 doclib 或上层配置分别管理 URL。Doclib managed server 固定使用 `--no-flash`，因为 Flash 在 doclib 进程内执行。

## 4. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户自行启动，doclib 或其它客户端连接指定 URL
- managed：由 `mineru server` / doclib 在运行时拉起和停止 parse-server 进程

managed 是生命周期管理方式，不是用户直接执行的命令模式。

## 5. Usage

api-server 启动时可使用单个 `--tier` 指定能力上限：

```bash
mineru-kit api-server --tier basic --port 16580
mineru-kit api-server --tier standard --port 15982
mineru-kit api-server --tier standard --no-flash --port 8000
mineru-kit api-server --tier standard --no-advanced --port 8000
mineru-kit api-server --tier standard --no-flash --no-advanced --port 8000
mineru-kit api-server --tier standard --preload-models
mineru-kit api-server --tier standard --language en --disable-image-analysis
```

未传 `--tier` 时暴露 `flash`、`basic`、`standard`、`advanced`；PDF/image 请求未指定 tier 时默认 `standard`。

OCR 模式通过每次 `POST /v1/parse/jobs` 请求的 `ocr_mode` 设置，可选 `auto`、`txt`、`ocr`，省略时为 `auto`。启动参数 `--ocr-mode` 已移除，传入会报错；Python `create_app()` 同样不再接受该参数。

模型默认在首次解析时懒加载。`--preload-models` 会在 Basic 或 Standard 服务启动时提前初始化所需模型或 VLM 客户端，并在失败时让能力接口返回明确错误；Flash 没有本地模型，该参数对 Flash 无操作。Doclib managed parse-server 会自动启用模型预加载。

### 使用远程 MinerU VLM 服务

```bash
mineru-kit api-server --tier standard --vlm-server-url http://127.0.0.1:30000/v1
mineru-api --vlm-server-url https://vlm.example.com/proxy/v1 --vlm-model mineru-model --preload-models
```

两个命令和 `python -m mineru.parser.api_server` 支持相同的连接参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vlm-server-url` | 空 | 远程服务地址，空值选择本地 VLM |
| `--vlm-api-key` | 空 | 远程 VLM 服务的 Bearer Key |
| `--vlm-model` | 空 | 上游模型名；为空时通过 `/v1/models` 唯一发现 |
| `--vlm-http-timeout` | 600 | VLM HTTP 读写超时，单位秒，必须为正整数 |
| `--vlm-max-concurrency` | 100 | VLM 推理请求并发，必须为正整数 |

表中为未配置时的默认值。命令行只覆盖显式传入的字段，其余读取全局 `model.vlm`；
环境变量覆盖 YAML。详见 [远程 VLM 配置](../config.md#远程-vlm)。例如：

```bash
export MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000/v1
export MINERU_MODEL_VLM_API_KEY=your-vlm-key
mineru-kit api-server --vlm-http-timeout 120 --vlm-max-concurrency 16
# 本次启动覆盖全局地址，恢复本地 VLM：
mineru-kit api-server --vlm-server-url ""
```

`--api-key` 认证解析 API 的调用方，`--vlm-api-key` 认证 API server 对上游 VLM 的请求，
两者独立。远程连接只用于 Standard/Advanced 的 VLM 推理，仍需本地 Hybrid 模型；
启动依赖预检不要求本地 VLM 引擎。Basic/Flash 不会连接远程 VLM。

启用 `--preload-models` 时，Standard 服务连接上游进行模型发现或校验，并预加载本地
Hybrid 模型；否则上游连接推迟到首次 Standard/Advanced 解析。连接、认证或模型校验失败
通过已有预加载错误或解析任务错误返回，不自动回退本地 VLM。推理重试继续使用底层默认设置。

Python 可使用 `create_app(vlm_config=VlmConfig(...))` 完整覆盖全局配置。配置由服务实例
传入其任务，不写回全局设置。此功能不增加 HTTP 请求字段；`/v1/models` 和 `/v1/tiers`
继续发布 MinerU 逻辑模型及质量档位，上游 serving 模型别名仅用于推理请求。

启动完成后，HTTP API 不暴露 backend。`GET /v1/tiers` 也不新增 backend 字段；调用方如需推断实现，只能从 `current_model` 做弱推断。

裸 `vlm` / `hybrid` 不是合法的 api-server 启动 backend；它们只可作为 Middle JSON 来源标记或内部分类概念。

正式参数分层：

### 稳定公开参数

- host / port
- tier，单值 `flash` / `basic` / `standard`
- no-flash
- no-advanced
- preload-models
- API key
- vlm-server-url / vlm-api-key / vlm-model / vlm-http-timeout / vlm-max-concurrency

### 稳定解析参数

- language
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
