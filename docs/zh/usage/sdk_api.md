# Python SDK 与 V1 API

## 本地 Python SDK

`mineru.parser` 提供无状态的 `parse`、`parse_async`、`MinerUParser` 和统一的 `ParseResult`，不会建立文档库。PDF 页范围默认全部，`page_range="1-3"` 使用从 1 开始的页码。

```python
from pathlib import Path
from mineru.parser import parse, ParseResult

result = parse("document.pdf", tier="flash", ocr_mode="txt", page_range="1-3")
Path("document.md").write_text(result.markdown(), encoding="utf-8")
Path("document.json").write_text(result.to_json(), encoding="utf-8")
restored = ParseResult.from_json(result.to_json())
assert restored.to_dict() == result.to_dict()
```

原生文档调用 `parse("report.docx", tier="flash")`；不传 PDF 专用页范围。PDF/图片的高质量解析可使用 `tier="standard"` 或 `tier="advanced"`，需要准备对应运行环境和模型。

异步入口接受同样的解析选项：

```python
import asyncio
from mineru.parser import parse_async

result = asyncio.run(parse_async("report.docx", tier="flash"))
print(result.markdown())
```

在已有事件循环中直接 `await parse_async(...)`。异步接口不保证所有引擎采用相同的执行机制。

## 连接自部署 V1 API

先在服务环境启动：

```bash
mineru-kit api-server --host 127.0.0.1 --port 8000 --tier standard
```

Python 客户端负责能力发现、文件提交、任务轮询和结果下载：

```python
from mineru.parser import MinerUApiParser

parser = MinerUApiParser(
    api_url="http://127.0.0.1:8000",
    api_key="",
    tier="standard",
    include_images=True,
)
result = parser.parse("document.pdf", page_range="1-3")
print(result.markdown())
```

`api_url` 是 `/v1` 之前的服务根地址。API Key 从配置或环境读取时，使用 `MINERU_API_KEY`；不要把密钥写入源码。连接其他机器意味着文件可能上传到该服务。WebUI 可连接同一地址：

```bash
mineru-kit webui --api-url http://127.0.0.1:8000
```

## HTTP 工作流

先通过健康检查和档位发现获取实际能力：

```bash
curl http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/v1/tiers
```

1. `POST /v1/uploads` 创建上传，根据响应的 URL、HTTP 方法和请求头上传文件；需要时调用 `/v1/uploads/{id}/complete`，取得 `file.id`。已完成的去重上传可直接取得文件引用。
2. `POST /v1/parse/jobs` 提交任务，例如下方 JSON。
3. `GET /v1/parse/jobs/{job_id}` 轮询。`completed`、`partial`、`failed`、`canceled` 都是终态；逐文件检查错误，不能将部分成功当作全部完成。
4. 按任务响应中的产物引用调用 `GET /v1/files/{file_id}/content` 下载结果。

```json
{
  "files": [{"source": {"type": "file_id", "file_id": "file-id-from-upload"}, "page_range": "1-3"}],
  "tier": "standard",
  "ocr_mode": "auto",
  "output_formats": ["markdown", "middle_json", "structured_content", "zip"]
}
```

非 PDF 文件省略 `page_range`。当前自部署服务提供上例四种产物；以服务能力和 `/docs` 的 OpenAPI 为准，不能由渲染层的格式列表推断 API 支持范围。

4.0 V1 服务不提供旧 `/file_parse` 和 `/tasks` 路由。既有客户端迁移见[迁移指南](../reference/migration_4.md)，Python 渲染与结果保存见[输出格式](../reference/output_files.md)。
