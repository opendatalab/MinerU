# 用量与部署限制

本页描述仓库中的 Python API Server 与 Router。云服务的计费、限流、套餐和保留期须查阅对应部署说明，
不再把早期 SaaS 设计数字作为当前承诺。

## GET `/v1/usage`

启用 API Key 后需要 Bearer 认证。返回 `UsageResponse`，实现见 `mineru/parser/api_server.py`。

| 字段 | 当前 Python Server 含义 |
|---|---|
| `object` | 固定 `usage` |
| `access_level` | 无认证配置为 anonymous，通过认证为 registered；不代表套餐等级 |
| `billing_period.start/end` | 当前进程启动时间／null，不是云端账期 |
| `current.pages_processed/files_processed` | 进程内累计完成统计 |
| `current.jobs_created` | 内部逐文件执行计数，保留历史字段名，不是父 Task 数量 |
| `limits.max_pages_per_file` | null，无固定上传 PDF 页数上限 |
| `limits.max_file_size_bytes` | null，无固定直接上传大小上限；URL 下载和部署网关限制另算 |
| `limits.max_files_per_job` | 100，Tasks 清单上限 |
| `limits.max_concurrent_jobs` | 当前配置的逐文件执行并发上限 |
| `limits.max_file_retention_days` | retention 秒数按天向上取整；禁用回收时为 null |

默认结果保留 24 小时，清理已过期记录不会减少累计统计。进程重启不保证保留任务或用量。
Router 的 jobs/files 来自当前进程按调用方隔离的统计，pages_processed 当前为 0；并发上限是健康 worker 的并发和，保留天数为 null。它不累计 worker 历史用量，也不是跨重启计费账本。

## 认证与限流

服务未配置 API Key 时，受支持的输出格式均可使用；配置后受保护路由要求有效 Key。
没有 anonymous/registered 不同的输出格式套餐，也不提供文档 Chat/Responses 路由。

当前 Python 实现不承诺固定的每分钟配额或 `X-RateLimit-*` 头。
反向代理和 SaaS 可以添加自己的限流策略，客户端应按实际 HTTP 状态和响应处理。
不能凭 GET /usage 的历史 `jobs` 字段名调用已经删除的 Jobs API。

上传有效期、URL 下载大小限制与结果保留期是不同约束，详见
[Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/) 和 [服务参数](../cli/mineru-kit-api-server.md)。
