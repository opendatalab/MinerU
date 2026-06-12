# ParseResult

状态: Draft
读者: SDK 开发者、内容输出开发者、集成方
范围: `ParseResult` 的职责、输出格式、保存行为和类型稳定性
底稿: `../../../NEXT-SDK.md`

## 定位

`ParseResult` 是 Tool SDK 的统一结果对象。无论解析来自本地 backend、Office parser、HTML parser，还是 `MinerUApiParser` 连接的 v1 API，最终都应向调用方返回 `ParseResult`。

它承担两件事:

- 持有结构化 pages。
- 基于 pages 派生 markdown、content list、images 等输出。

## 当前字段

当前 `ParseResult` 是 dataclass，核心字段包括:

| 字段 | 说明 |
|------|------|
| `pages` | `list[PageInfo]`，解析后的页面结构。 |
| `_pdf_doc` | PDF/image 结果用于裁剪图片的文档对象。 |
| `_model_output` | backend 原始模型输出，可选。 |
| `_images_cache` | 图片缓存。 |

`ParseResult` 顶层不承载 backend 或版本字段。backend 属于 page / middle-json 层的内部过渡信息；版本信息应由 API 响应、文件 envelope 或调用方上下文提供，不作为 `ParseResult` 的稳定字段。

带下划线的字段是内部实现细节，不应成为外部稳定 API。外部用户应优先使用方法。

## 公开方法

目标公开方法:

```python
class ParseResult:
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_dict(d: dict) -> "ParseResult": ...
    @staticmethod
    def from_json(s: str) -> "ParseResult": ...

    def markdown(self, *, add_markers: bool = False) -> str: ...
    def content_list(self) -> list[dict]: ...
    def content_list_v2(self) -> list[dict]: ...
    def images(self) -> dict[str, bytes]: ...
    def save(self, writer) -> None: ...
```

当前状态:

| 方法 | 当前状态 | 目标 |
|------|----------|------|
| `to_dict()` | 已有 | 保持，输出稳定 pages envelope。 |
| `to_json()` | 已有 | 保持。 |
| `from_dict()` | TODO | 必须实现，用于 API JSON / 缓存恢复。 |
| `from_json()` | TODO | 必须实现。 |
| `markdown()` | 已有 | 保持，参数名稳定。 |
| `content_list()` | 已有 | 保持。 |
| `content_list_v2()` | 已有 | 保持。 |
| `images()` | 已有 | 保持，返回 path -> bytes。 |
| `save()` | 已有 | 需要稳定输出文件命名。 |

## 序列化格式

目标 `to_dict()` 输出:

```json
{
  "pages": [
    {
      "page_idx": 0,
      "para_blocks": []
    }
  ]
}
```

兼容要求:

- 读取旧格式时，如果顶层只有 `pages`，仍应成功。
- 读取 v1 API 的 `json` output 时，应接受顶层 `pages` 或直接 page list。
- 未识别字段应保留或忽略，但不能导致常规读取失败。

## Markdown 与 marker

`markdown(add_markers=False)` 输出面向最终阅读。`add_markers=True` 用于 CLI、debug 或需要页码定位的场景。

命名建议:

- 保留 `add_markers`。
- CLI 的 `--no-marker` / `--marker` 只映射到该参数。
- 不在 `ParseResult` 中混入 CLI 格式化逻辑。

## Images

`images()` 返回 `dict[str, bytes]`:

```python
images = result.images()
for path, data in images.items():
    ...
```

规则:

- PDF/image 结果可以通过 PDF 页面和 span bbox 裁剪。
- Office/HTML 结果可以从 span 的 base64 或 image path 中抽取。
- 图片路径是产物内部路径，不是本地绝对路径。

## Save

`save(writer)` 将结果写入 writer。writer 需要提供:

```python
writer.write_string(path: str, content: str) -> None
writer.write(path: str, data: bytes) -> None
```

当前输出包括:

| 文件 | 内容 |
|------|------|
| `{prefix}.md` | Markdown。 |
| `{prefix}_middle.json` | `to_json()`。 |
| `{prefix}_content_list.json` | content list。 |
| `{prefix}_content_list_v2.json` | content list v2。 |
| `{prefix}_model.json` | 原始模型输出，可选。 |
| image paths | 图片 bytes。 |

目标上需要统一命名，避免 CLI、parser 和 API server 各自定义不同文件名。

## 与 Middle JSON

`ParseResult.pages` 是 middle structure 的 typed form。Middle JSON 的跨 backend 对齐见 [Middle JSON](../middle-json.md)。

SDK 设计约束:

- `ParseResult` 可以暴露 `pages` 给高级用户。
- 普通用户应通过 `markdown()`、`content_list()`、`images()` 消费结果。
- middle structure 的 schema 变化应通过 `from_dict()` 做兼容。

## 未决问题

`to_dict()` 元数据字段、便利方法和 `save()` writer protocol，集中维护在 [开放问题清单](../open-questions.md)。
