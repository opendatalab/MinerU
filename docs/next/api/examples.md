# 端到端示例

状态: Draft
读者: API 使用者、SDK 开发者
范围: 官方 API 与本地 server 的常见调用流程
来源: 由根目录旧 Unified API 底稿迁移整理而来

## 官方 API: 上传并解析 PDF

```bash
SHA=$(sha256sum report.pdf | cut -d' ' -f1)
SIZE=$(stat -f%z report.pdf)

RESP=$(curl -s -X POST https://mineru.net/api/v1/uploads \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"report.pdf\",\"bytes\":$SIZE,\"mime_type\":\"application/pdf\",\"purpose\":\"parse\",\"sha256sum\":\"$SHA\"}")

STATUS=$(echo "$RESP" | jq -r '.status')

if [ "$STATUS" = "pending" ]; then
  UPLOAD_ID=$(echo "$RESP" | jq -r '.id')
  URL=$(echo "$RESP" | jq -r '.upload_url')
  METHOD=$(echo "$RESP" | jq -r '.upload_method')

  curl -X "$METHOD" "$URL" \
    -H "Content-Type: application/pdf" \
    --data-binary @report.pdf

  RESP=$(curl -s -X POST "https://mineru.net/api/v1/uploads/$UPLOAD_ID/complete" \
    -H "Authorization: Bearer $MINERU_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"sha256sum\":\"$SHA\"}")
fi

FILE_ID=$(echo "$RESP" | jq -r '.file.id')

RESP=$(curl -s -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"files\": [{
      \"source\": {\"type\":\"file_id\",\"file_id\":\"$FILE_ID\"}
    }],
    \"tier\": \"high\",
    \"output_formats\": [\"markdown\",\"middle_json\",\"zip\"]
  }")
```

如果 create upload 返回 `status:"completed"`，说明发生秒传，客户端可以直接使用响应里的 `file.id` 创建 parse job。

创建任务后，保存响应里的 `job_id`:

```bash
JOB_ID=$(echo "$RESP" | jq -r '.job_id')
```

## 官方 API: 轮询任务并下载 Markdown

```bash
while true; do
  JOB=$(curl -s "https://mineru.net/api/v1/parse/jobs/$JOB_ID" \
    -H "Authorization: Bearer $MINERU_API_KEY")

  STATUS=$(echo "$JOB" | jq -r '.status')
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "partial" ] || \
     [ "$STATUS" = "failed" ] || [ "$STATUS" = "canceled" ]; then
    break
  fi

  sleep 2
done

MD_FILE_ID=$(echo "$JOB" | jq -r '.files[0].output_files.markdown.file_id')

curl -L "https://mineru.net/api/v1/files/$MD_FILE_ID/content" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -o report.md
```

实际客户端应使用指数退避轮询，最大间隔建议 30 秒。

## 官方 API: URL 来源

```bash
curl -X POST https://mineru.net/api/v1/parse/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "files": [{
      "source": {"type":"url","url":"https://example.com/doc.pdf"}
    }],
    "tier": null,
    "output_formats": ["markdown"]
  }'
```

不传 API Key 时进入 anonymous access level。anonymous 可以使用基础解析能力，但不能使用 callback 或高级导出格式。

## 下载产物

```bash
curl -L "https://mineru.net/api/v1/files/$MD_FILE_ID/content" \
  -H "Authorization: Bearer $MINERU_API_KEY" \
  -o report.md
```

官方 API 返回 302，`-L` 会自动跟随重定向。

## 本地 Server 示例

先发现本地能力:

```bash
curl http://localhost:8000/v1/health
curl http://localhost:8000/v1/tiers
```

本地 server 可以直接使用 `local` source:

```bash
mineru-kit api-server --allow-local-source

curl -X POST http://localhost:8000/v1/parse/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "files": [{
      "source": {"type":"local","path":"/data/docs/report.pdf"}
    }],
    "tier": null,
    "output_formats": ["markdown"]
  }'
```

省略 `tier` 或传 `null` 时，本地 server 只会选择当前可发现的 `medium` 或 `high`。如果两者都不可用，请求应失败，而不是使用 `flash` 返回解析结果。

## 本地 Server: 复用上传流程

客户端也可以对本地 server 使用与官方 API 相同的上传流程。差异只在于 `upload_url` 指向本地 server:

```bash
SIZE=$(stat -f%z report.pdf)

RESP=$(curl -s -X POST http://localhost:8000/v1/uploads \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"report.pdf\",\"bytes\":$SIZE,\"mime_type\":\"application/pdf\",\"purpose\":\"parse\"}")

UPLOAD_ID=$(echo "$RESP" | jq -r '.id')
URL=$(echo "$RESP" | jq -r '.upload_url')
METHOD=$(echo "$RESP" | jq -r '.upload_method')

curl -X "$METHOD" "$URL" \
  -H "Content-Type: application/pdf" \
  --data-binary @report.pdf

RESP=$(curl -s -X POST "http://localhost:8000/v1/uploads/$UPLOAD_ID/complete")
FILE_ID=$(echo "$RESP" | jq -r '.file.id')
```

这样写的客户端可以在官方 API 和 Local Parse Server 之间复用绝大多数代码。
