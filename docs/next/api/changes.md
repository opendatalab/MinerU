# API 变更记录

范围: NEXT v1 API 标准的破坏性或需要迁移的行为变化。

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
