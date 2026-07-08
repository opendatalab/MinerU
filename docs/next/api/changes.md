# API 变更记录

范围: NEXT v1 API 标准的破坏性或需要迁移的行为变化。

## 2026-07-08: 明确 Local Parse Server base URL

Local Parse Server 的 base URL 是 `http://localhost:8000`，endpoint 直接挂在 `/v1` 下。`/api` 只属于 `https://mineru.net/api` 这类官方远程 API base URL。

## 2026-07-08: 增加 `features.output_formats`

`GET /v1/health` 的 `features` 增加 `output_formats`，用于声明当前部署实际支持的 parse job 输出格式。客户端应以该字段作为能力发现事实，而不是假设所有部署都支持同一组输出格式。

## 2026-07-08: 增加 `features.sources`

`GET /v1/health` 的 `features` 增加 `sources`，用于声明当前部署允许的 parse job source 类型。客户端应以该字段决定是否可以使用 `local` source；没有声明 `local` 时，应通过 Uploads API 取得 `file_id` 后再创建解析任务。

## 2026-07-08: 收紧 Local Parse Server source 策略

Local Parse Server 对 `local`、`inline` 和 `url` source 增加启动时可配置的安全策略: `local` source 默认关闭，只有显式开启 `--allow-local-source` 后才允许读取 server 进程权限范围内的本地路径；`inline` source 受 `--max-inline-bytes` 限制；`url` source 默认只允许 HTTPS，只有显式开启 `--allow-http-source` 后才允许 HTTP。

## 2026-07-08: 删除 parse job SSE events

NEXT v1 API 不再定义 `GET /v1/parse/jobs/{job_id}/events`，parse job 响应中也不再返回 `links.events`。客户端统一通过 `GET /v1/parse/jobs/{job_id}` 查询任务状态和结果。

线上 API Server 不支持长连接；本地 SSE 实现也只是服务端每秒检查 job 状态后通过长连接转发，相比客户端轮询没有实质效率收益。删除这一路径可以避免线上/本地协议分叉，并让 parse job 保持单一轮询模型。

## 2026-07-08: 删除 parse job `wait` 参数

`POST /v1/parse/jobs` 不再支持 `wait`。创建 parse job 始终返回 `202` 和 job 引用，客户端通过 `GET /v1/parse/jobs/{job_id}` 查询状态和结果。

API 不提供在创建请求中同步等待完成的模式，也不内联返回 Markdown 内容。所有产物内容仍通过 `output_files` 对应的 File API 下载。

这样可以避免线上服务长连接阻塞和本地/线上实现差异，让 parse job 保持单一异步模型。

## 2026-07-08: 删除独立 `images` 输出格式

`images` 不再是 `POST /v1/parse/jobs` 的合法 `output_formats` 值，parse job 响应中也不再返回 `output_files.images`。

图片 sidecar 只通过 `zip` 产物返回。`markdown`、`middle_json`、`content_list`、`structured_content` 输出仍可能包含 `image_path` 引用，但这些引用对应的图片字节不再作为独立 parse output 暴露。

这样可以避免同一批图片同时通过独立 output 和 zip 两条路径返回，减少产物语义重复，并让需要自包含解析结果的客户端统一依赖 `zip`。
