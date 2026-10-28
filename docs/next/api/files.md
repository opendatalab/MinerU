# Files

Files 保存源文件视图与已生成的结果，不提供文件列表。异步解析统一使用 [Tasks](tasks.md)，客户端按 `files[].result_url` 下载，不自行拼接存储路径。

| 方法 | 地址 | 行为 |
|---|---|---|
| GET | `/v1/files/{file_id}` | 返回文件元数据。 |
| GET | `/v1/files/{file_id}/content` | 流式下载已保存的解析产物；源文件不提供公共下载。 |

## FileObject：文件元数据

元数据接口成功返回 `200 OK`，`Content-Type: application/json`，JSON 正文使用 UTF-8。
下列字段在当前本地 API Server 和 Router 的文件对象中均会出现；可空值写为 `null`，不省略字段。

| 字段 | JSON 类型 | 可为 null | 含义 |
|---|---|---|---|
| `id` | string | 否 | 不透明的 Files ID；Router 返回自己的公共 ID。 |
| `object` | string | 否 | 固定为 `file`。 |
| `file_name` | string | 否 | 文件逻辑名称；结果采用 `<输入文件主名>.<产物扩展名>`。 |
| `bytes` | integer | 否 | 实际保存的文件大小，单位为字节。 |
| `created_at` | integer | 否 | 文件视图创建时间，Unix 秒级时间戳。 |
| `expires_at` | integer | 是 | 文件视图的过期时间，Unix 秒；活动引用或保留策略可能使其暂为 `null`。 |
| `purpose` | string | 否 | `parse` 表示源文件，`parse_output` 表示结果。 |
| `sha256sum` | string | 是 | 保存字节的小写 SHA-256；未提供元数据时为 `null`。 |

Files 的时间字段为 Unix 秒级整数，与 Tasks 的 ISO-8601 时间字符串不同。

## GET /v1/files/{file_id}：查询元数据

成功返回 **`200 OK` + `application/json`**，正文直接是 `FileObject`，没有额外包装：

```json
{
  "id": "file-result0123456789abcdef",
  "object": "file",
  "bytes": 4096,
  "created_at": 1792728001,
  "expires_at": null,
  "file_name": "report.zip",
  "purpose": "parse_output",
  "sha256sum": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

示例中的 ID、大小、时间和 hash 为示意值。本地源文件元数据使用相同结构，`purpose` 为 `parse`，`file_name` 为输入文件名。Tasks 不公开源缓存 ID；创建响应中的 files[].file_id 即本次解析结果的公共 Files ID，Router 将它映射到 worker 的结果 ID。Tasks 的 file_name/bytes 描述输入，Files 的 file_name/bytes 描述实际产物。

产物生成前只预留 ID，不注册 Files 对象：GET 元数据和 GET content 均返回 404 file_not_found。创建任务的 ID 预留不代表产物已经存在，准备与执行状态通过 GET Tasks 查询。

## 产物保留与清理

当前不提供产物删除接口，`DELETE /v1/files/{file_id}` 在认证通过后返回 `405 Method Not Allowed`。
任务取消使用 `POST /v1/tasks/{task_id}/cancel`，保留已完成的产物和下载链接；取消任务不会删除这些资源。
产物按既有保留期由内部机制自动回收，活动执行、下载引用和 Router 租约继续受保护。

## GET /v1/files/{file_id}/content：下载结果

完整下载成功时返回 **`200 OK` 和文件二进制正文**，不是 JSON，也不是 Base64 字符串：

```http
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Disposition: attachment; filename="report.zip"

<ZIP 文件二进制内容>
```

| 产物 | Content-Type | 正文格式 |
|---|---|---|
| default、LaTeX 或多格式 ZIP | `application/zip` | ZIP 文件字节。 |
| 单独 DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | DOCX 文件字节。 |
| 单独 HTML | `text/html; charset=utf-8`（本地） | UTF-8 HTML 文件字节。 |

下载建议文件名由 `Content-Disposition` 提供，非 ASCII 文件名可能使用标准的 `filename*` 编码。
JSON 字段使用 `file_name`；HTTP 附件头的标准参数仍是 `filename` 或 `filename*`。ZIP 成员和输出格式见 [Tasks 输出格式与下载](tasks.md#输出格式与下载)。
本地 API Server 直接流式返回产物；Router 代理 worker 的字节及下载头。Tasks 返回的 SaaS `result_url` 可能直接指向对象存储，不要求经过本路由。

## 错误与保留期

请求失败使用非 2xx 状态码和 JSON 错误对象，`error.code/message` 必带，`error.param` 可选，不包含 `type`，详见 [响应与错误](responses.md)。
常见情况为 `401 invalid_api_key`、`404 file_not_found`；尝试通过 `/content` 下载源文件返回 `403 feature_requires_api_key`，源文件下载不属于公共能力。

源文件通过 Tasks 的 JSON 清单和返回的上传链接创建。相同 hash 的有效缓存免传；不同文件名可以共享一份源字节，文件名和解析参数仍属于各自任务。
API Server 与 Router 默认保留 86400 秒，每 300 秒扫描。`--retention-seconds` 优先于 `MINERU_API_RETENTION_SECONDS`，0 禁用正常保留期清理。上传链接固定的一小时窗口不随该设置改变。

清理只删除过期视图和无引用字节。排队、执行、上传、下载持有的引用阻止物理删除；结果保留期从父任务完成资源清理后开始。
Router 映射结果 Files 地址，源缓存由按调用方隔离的私有引用管理，不依赖下游源 Files ID。父任务清理后开始私有源缓存保留期，后台独立回收并维持 worker 结果租约。SaaS 可以提供对象存储签名地址，不要求实现本地 Files 下载路由。
