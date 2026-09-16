# Python SDK

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

### 批量处理与实例复用

批量处理时复用同一个 `MinerUApiParser` 实例，不要每个文件新建一个；每次调用各自开合 HTTP 会话，无需显式释放资源。HTTP 层以任务状态和逐文件 `error` 表达失败；Python SDK 会将 `failed`/`canceled` 终态（以及网络/HTTP 错误）转换为异常，因此批处理应逐文件捕获、记录汇总，并按明确的策略退出：

```python
import sys
from pathlib import Path
from mineru.parser import MinerUApiParser

pdfs = sorted(Path("./documents").glob("*.pdf"))
if not pdfs:
    sys.exit("./documents 下没有输入文件")

output_dir = Path("out")
output_dir.mkdir(parents=True, exist_ok=True)

parser = MinerUApiParser(api_url="http://127.0.0.1:8000", tier="standard", include_images=True)
failures: list[tuple[str, str]] = []
for pdf in pdfs:
    try:
        result = parser.parse(str(pdf))
        (output_dir / f"{pdf.stem}.md").write_text(result.markdown(), encoding="utf-8")
    except Exception as exc:  # 终态任务失败与传输错误都会抛出异常
        failures.append((pdf.name, str(exc)))
        print(f"failed: {pdf.name}: {exc}")

if failures:
    sys.exit(f"{len(failures)}/{len(pdfs)} 个文件失败")
```

关注吞吐时检查服务日志和 `GET /v1/usage`。档位选择见[档位与运行环境](tiers.md)，底层请求周期见 [V1 HTTP API 完整示例](http_api.md)。

## 不使用 SDK 的 HTTP 调用

上传 → 任务 → 轮询 → 下载的闭环也可以直接用 HTTP 调用完成。[V1 HTTP API 完整示例](http_api.md) 提供了完整的 curl 示例，覆盖完成上传、终态处理（含 `partial`）、客户端超时后继续轮询和产物下载。

4.0 V1 服务不提供旧 `/file_parse` 和 `/tasks` 路由。既有客户端迁移见[迁移指南](../reference/migration_4.md)，Python 渲染与结果保存见[输出格式](../reference/output_files.md)。
