# ParseResult

状态: Draft
读者: SDK 开发者、内容输出开发者、集成方
范围: `ParseResult` 的职责、输出格式、保存行为和类型稳定性
来源: 由旧 SDK 底稿迁移整理而来；旧底稿已归档删除

## 定位

`ParseResult` 是 Tool SDK 的统一结果对象。无论解析来自本地 backend、Office parser、HTML parser，还是 `MinerUApiParser` 连接的 v1 API，最终都应向调用方返回 `ParseResult`。

它承担两件事:

- 持有 schema 2.0 的 `MiddleJson` 结构化表示。
- 基于它派生 markdown、structured content、images 等输出。

## 当前字段

当前 `ParseResult` 是 dataclass，核心字段包括:

| 字段 | 说明 |
|------|------|
| `middle_json` | `MiddleJson`，schema 2.0 结构化中间表示，是结果的主字段。 |
| `pages` | property，返回 `list[PageInfo]`，委托给 `middle_json.pages`，不是 dataclass 字段。 |
| `_pdf_doc` | PDF/image 结果用于裁剪图片的文档对象，可选。 |
| `_model_output` | backend 原始模型输出，可选。 |
| `_image_cache` | 顶层图片缓存（`ImagePayloadCache`），初始化时规范化，不再从 span 携带图片字节。 |

`ParseResult` 通过持有的 `middle_json` 携带 schema 2.0 顶层字段（`is_full_document`、`file_suffix`、`effort`、`parse_mode`、`mineru_version`）；`ParseResult` 自身不额外维护 backend 或版本字段。

带下划线的字段是内部实现细节，不应成为外部稳定 API。外部用户应优先使用方法。

## 公开方法

当前公开方法:

```python
class ParseResult:
    def to_dict(self, *, skip_defaults: bool = True) -> dict: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_dict(d: dict) -> "ParseResult": ...
    @staticmethod
    def from_json(s: str) -> "ParseResult": ...

    def markdown(
        self,
        *,
        add_markers: bool = False,
        mode: RenderMode | None = None,
        asset_base_url: str = "",
        image_renderer: ImageRenderer | None = None,
    ) -> str: ...
    def structured_content(self, *, asset_base_url: str = "") -> dict: ...
    def images(self) -> dict[str, bytes]: ...
    def save(self, writer) -> None: ...
```

当前状态:

| 方法 | 当前状态 | 目标 |
|------|----------|------|
| `to_dict()` | 已有 | 保持，输出 schema 2.0 envelope（顶层 `schema_version: "2.0"` + `MiddleJson` 顶层字段）。 |
| `to_json()` | 已有 | 保持。 |
| `from_dict()` | 已有 | 保持，用于 API JSON / 缓存恢复；仅接受 dict，schema 2.0 与受支持旧 payload 之外的版本抛 `ValueError`。 |
| `from_json()` | 已有 | 保持。 |
| `markdown()` | 已有 | 保持，参数名稳定；`add_markers` 之外另有 `mode`、`asset_base_url`、`image_renderer`。 |
| `structured_content()` | 已有 | 保持，返回 `dict`；`asset_base_url` 控制资源 base URL。 |
| `images()` | 已有 | 保持，返回 path -> bytes。 |
| `save()` | 已有 | 文件命名已在 `save()` 内固定（见 Save）。 |

## 序列化格式

`to_dict()` 顶层先写入 `schema_version: "2.0"`，再合并 `MiddleJson` 的顶层字段:

```json
{
  "schema_version": "2.0",
  "pages": [],
  "is_full_document": true,
  "file_suffix": "pdf",
  "effort": "high",
  "parse_mode": "ocr",
  "mineru_version": "4.0.0a7"
}
```

PDF 输入序列化时排除 block 内的 `image_base64` 字段，图片字节统一走 `images()`。

兼容要求:

- 读取 schema 2.0 payload：直接按严格 `MiddleJson` 校验构造。
- 读取 MinerU 3.4.5 的 `pdf_info` 原始 payload 或 schema 1.0 的 `pages` 包装：经 legacy adapter 回推为 raw model-list 后重走统一后处理。
- 其它未知 `schema_version` 抛 `ValueError`，必须从源文件重新解析，不做自动迁移。
- `from_dict()` 只接受 dict；非 dict 输入抛 `ValueError`。
- schema 2.0 使用 `extra="forbid"` 严格校验，未识别字段会导致校验失败，不会被静默保留。

## Markdown 与 marker

`markdown(add_markers=False)` 输出面向最终阅读。`add_markers=True` 映射到 `mode="full"` 渲染，用于 CLI、debug 或需要页码定位的场景。

`markdown()` 另接受:

| 参数 | 说明 |
|------|------|
| `mode` | `RenderMode`（`default` / `full`）；未传时按 `add_markers` 推导。 |
| `asset_base_url` | 输出资源（图片等）的 base URL 前缀。 |
| `image_renderer` | 自定义图片渲染器（`render` 层的 `ImageRenderer` 协议），未传时使用默认渲染。 |

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
| `markdown.md` | `markdown()` 输出。 |
| `middle_json.json` | `to_json()`。 |
| `structured_content.json` | `structured_content()` 的 JSON。 |
| `model_output.json` | 原始模型输出，仅 `_model_output` 存在时写出。 |
| image paths | `images()` 返回的图片 bytes，路径即 `images()` 的 key。 |

以上文件名由 `save()` 内部固定，调用方无需（也不能）通过参数调整。

## 与 Middle JSON

`ParseResult.pages` 是 middle structure 的 typed form。Middle JSON 的跨 backend 对齐见 [Middle JSON](../middle-json.md)。

SDK 设计约束:

- `ParseResult` 可以暴露 `pages` 给高级用户。
- 普通用户应通过 `markdown()`、`structured_content()`、`images()` 消费结果。
- middle structure 的 schema 变化应通过 `from_dict()` 做兼容。

## 未决问题

`to_dict()` 元数据字段、便利方法和 `save()` writer protocol，集中维护在 [开放问题清单](../open-questions.md)。
