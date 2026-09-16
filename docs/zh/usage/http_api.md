# V1 HTTP API 完整示例

自部署 V1 API 与 Python SDK、WebUI 使用同一套接口。本页用纯 HTTP 调用演示一次完整闭环：创建上传、上传字节、提交解析任务、轮询到终态、下载产物。Python 客户端见 [Python SDK](sdk_api.md)。

先启动本地服务，或将示例指向任意 V1 部署（包括官方云服务）：

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

服务以 `--api-key` 启动时，`/v1/*` 请求携带 `Authorization: Bearer $MINERU_API_KEY`；匿名本地访问可省略该头。第 2 步的字节上传**不走** `/v1`：必须原样使用第 1 步返回的 URL、方法和请求头。

## 完整示例（curl + jq）

```bash
BASE=http://127.0.0.1:8000
AUTH="Authorization: Bearer $MINERU_API_KEY"    # 服务允许匿名访问时省略
FILE=document.pdf
SIZE=$(stat -c%s "$FILE")                       # macOS 使用 stat -f%z "$FILE"

# 1. 创建上传会话
RESP=$(curl -s -X POST "$BASE/v1/uploads" -H "$AUTH" -H "Content-Type: application/json" -d "{
  \"filename\": \"$FILE\", \"bytes\": $SIZE, \"mime_type\": \"application/pdf\", \"purpose\": \"parse\"
}")

# 2. 用返回的 method/url/headers 上传字节（仅 pending 状态需要）
if [ "$(echo "$RESP" | jq -r .status)" = "pending" ]; then
  curl -s -X PUT "$(echo "$RESP" | jq -r .upload_url)" \
       -H "Content-Type: application/pdf" --data-binary "@$FILE"
  # 3. 完成上传，取得 file id
  RESP=$(curl -s -X POST "$BASE/v1/uploads/$(echo "$RESP" | jq -r .id)/complete" -H "$AUTH")
fi
FILE_ID=$(echo "$RESP" | jq -r .file.id)

# 4. 提交解析任务
JOB=$(curl -s -X POST "$BASE/v1/parse/jobs" -H "$AUTH" -H "Content-Type: application/json" -d "{
  \"files\": [{\"source\": {\"type\": \"file_id\", \"file_id\": \"$FILE_ID\"}, \"page_range\": \"1-10\"}],
  \"tier\": \"standard\",
  \"output_formats\": [\"markdown\", \"zip\"]
}")
JOB_ID=$(echo "$JOB" | jq -r .job_id)

# 5. 轮询直到终态（有界循环，请自行加上尝试上限）
while : ; do
  JOB=$(curl -s "$BASE/v1/parse/jobs/$JOB_ID" -H "$AUTH")
  STATUS=$(echo "$JOB" | jq -r .status)
  case "$STATUS" in completed|partial|failed|canceled) break ;; esac
  sleep 3
done
echo "$JOB" | jq '{status, progress, files: [.files[] | {name, status, error}]}'

# 6. 下载每个成功文件的产物
for FID in $(echo "$JOB" | jq -r ".files[].output_files.markdown.file_id? // empty"); do
  curl -s "$BASE/v1/files/$FID/content" -H "$AUTH" -o document.md
done
```

示例刻意展示的要点：

- **上传生命周期。** 创建上传时传 `sha256sum` 可启用完整性校验和秒传；命中已存在文件时响应直接为 `completed` 并内嵌 `file` 对象，跳过第 2–3 步。
- **字节上传。** 始终原样使用返回的 `upload_method`、`upload_url` 和 `upload_headers`。自部署服务返回同源回环 URL；官方 API 返回自带授权的预签名对象存储 URL——不要额外添加 Bearer 头。
- **终态。** `completed`、`partial`、`failed`、`canceled` 都是终态。`partial` 表示部分文件成功：从 `status` 为 `completed` 的文件条目读取 `output_files`，其余查看 `error`；部分成功不等于全部完成。
- **客户端超时不等于任务失败。** 轮询预算耗尽时任务可能仍是 `queued` 或 `running`；之后继续轮询 `GET /v1/parse/jobs/{job_id}` 即可，任务不会因为客户端停止询问而被取消。
- **产物。** 所有产物都是 `purpose:"parse_output"` 的文件；内容一律通过 `GET /v1/files/{file_id}/content`、用 `files[].output_files` 下的 id 获取。以 `GET /v1/health`（`features.sources`、`features.output_formats`）为当前部署实际能力的依据；渲染器支持的格式不代表 API 暴露。

## 生产部署注意事项

- 谨慎绑定 `--host`。默认 `127.0.0.1` 只监听回环；使用 `0.0.0.0` 暴露到网络时，应配合 `--api-key` 或带鉴权的反向代理。
- 上传与产物存储在服务主机上，`--upload-dir` 控制上传目录。高吞吐部署需规划磁盘用量与保留策略；产物文件的 `expires_at` 以部署配置为准。
- `--concurrency` 限制并发解析任务数；超出后表现为 `queued` 时间变长，而不是失败。
- `GET /v1/usage` 报告用量。用 `/v1/health` 和 `/docs`（启用时）的 OpenAPI 文档区分服务端故障与客户端错误；重试 `failed` 任务前先查看服务端日志。

V1 服务不提供旧 `/file_parse`、`/tasks` 路由；旧客户端迁移见[迁移指南](../reference/migration_4.md)。
