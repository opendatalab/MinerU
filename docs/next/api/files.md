# Files

Files 保存源文件视图与已生成的结果。异步解析统一使用 [Tasks](tasks.md)，客户端按 `files[].result_url` 下载，不自行拼接存储路径。

| 方法 | 地址 | 行为 |
|---|---|---|
| GET | `/v1/files` | 按 `purpose/limit/after/order` 查询当前调用方的文件列表。 |
| GET | `/v1/files/{file_id}` | 返回文件元数据。 |
| GET | `/v1/files/{file_id}/content` | 流式下载已保存的解析产物；源文件不提供公共下载。 |
| DELETE | `/v1/files/{file_id}` | 删除文件视图；正在执行和下载的引用继续受保护。 |

元数据包括 `id/object/filename/bytes/created_at/expires_at/purpose/sha256sum`。
`purpose` 为 `parse`（源文件）或 `parse_output`（结果）；删除返回 `id/object/deleted`。
列表使用 `object: list`、`data`、`first_id`、`last_id`、`has_more`；默认 20 条，上限 100。

源文件通过 Tasks 的 JSON 清单和返回的上传链接创建。相同 hash 的有效缓存免传；不同文件名可以共享一份源字节，文件名和解析参数仍属于各自任务。
API Server 与 Router 默认保留 86400 秒，每 300 秒扫描。`--retention-seconds` 优先于 `MINERU_API_RETENTION_SECONDS`，0 禁用正常保留期清理。上传链接固定的一小时窗口不随该设置改变。

清理只删除过期视图和无引用字节。排队、执行、上传、下载持有的引用阻止物理删除；结果保留期从父任务完成资源清理后开始。
Router 映射自身 Files 地址，按调用方隔离缓存与元数据，后台维持 worker 结果租约。SaaS 可以提供对象存储签名地址，不要求实现本地 Files 下载路由。
