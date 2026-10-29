# 任务通知

当前 Tasks 契约采用轮询，不接受 `callback`，也不提供 SSE 或 Webhook。客户端使用配置的 API 基址与 `GET /v1/tasks/{task_id}` 查询；批量任务的单个文件完成即可下载，无需等待整个批次终结。

SDK 提供进程内 `status_callback` 与 `duration_callback`，它们不是服务端 Webhook。未来增加推送能力需要另行定义认证、重试和幂等契约。
