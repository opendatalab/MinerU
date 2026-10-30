# ParseResult

核对入口：`mineru/parser/base.py`、`mineru/integrations/docvortex.py`。

`ParseResult` 是本地解析和 `MinerUApiParser` 共用的结果对象。唯一 dataclass 字段是
`middle_json: MiddleJson`；`pages` 是返回 `middle_json.pages` 的属性。
结果不持有 PDFium 对象，不包含 `_pdf_doc`、`_image_cache` 或 ModelJson，也没有 `images()` 方法。

## 读取与序列化

| 方法 | 当前行为 |
|---|---|
| `from_dict(d)` | 委托 `MiddleJson.from_dict()` 严格读取共享协议，再校验 MinerU 产品扩展 |
| `from_json(s)` | JSON 顶层必须是对象，再调用 `from_dict()` |
| `to_dict(skip_defaults=True)` | 校验扩展并序列化；PDF block 的 `image_base64` 不进入独立结构 JSON |
| `to_json()` | UTF-8 语义的 JSON 字符串，使用 `to_dict()` |
| `export_pages()` | 返回页面树深拷贝，避免调用方修改原结果 |

协议身份是 `schema="docvortex.middle"`、`schema_version="2.0"`。
文件类型与生产者在 `metadata`；实际 `tier/parse_mode` 在 `extensions["mineru"]`。
不存在顶层 `effort/file_suffix/parse_mode/mineru_version`。
通用读取不转换旧协议；历史兼容仅在 [Doclib 缓存边界](../middle-json/doclib-compatibility.md)。

## 渲染与保存

| 方法 | 参数与结果 |
|---|---|
| `markdown(...)` | `add_markers=False`、`mode=None`、`asset_base_url=""`、`image_renderer=None`；返回字符串 |
| `structured_content(asset_base_url="")` | 返回派生内容字典，不是 Middle JSON 协议 |
| `save(writer)` | 在文档副本上物化和校验素材，随后写出正文、结构和图片 |

`markdown()` 未传 `mode` 时，由 `add_markers` 选择 `RenderMode.FULL` 或 `DEFAULT`。
传入 `mode` 时以它为准。更多格式使用 `mineru.render`，见 [输出参考](https://opendatalab.github.io/MinerU/zh/reference/output_files/)。

```python
from mineru.parser import ParseResult, parse
from mineru.parser.writer import FileBasedDataWriter

result = parse("report.html", tier="flash")
restored = ParseResult.from_json(result.to_json())
assert restored.to_dict() == result.to_dict()
result.save(FileBasedDataWriter("output"))
```

`save()` 写入 `markdown.md`、`middle_json.json`、`structured_content.json` 和引用的图片。
writer 提供 `write_string(path, text)` 与 `write(path, bytes)`。缺少素材时在写出前失败；
不会重新裁图、读取当前目录或访问网络来补齐图片。
独立 JSON 不能替代携带素材的结果包。ModelJson 不进入 SDK、CLI 或 HTTP 产物。
Tasks 的多格式组合由 `mineru/parser/task_outputs.py` 独立打包，不能把 `save()` 的文件布局套用到所有输出格式。
