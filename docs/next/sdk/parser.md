# Tool SDK: `mineru.parser`

状态: Draft
读者: SDK 开发者、`mineru-kit` 开发者、核心开发者
范围: 无状态解析工具层的公开入口、统一 parser 类和目标契约
来源: 由旧 SDK 底稿迁移整理而来；旧底稿已归档删除

## 定位

`mineru.parser` 是无状态解析 SDK。它接受本地文件路径，由统一的 `MinerUParser` 完成解析，返回 `ParseResult`。

适用场景:

- 用户在 Python 代码中直接解析单个文件。
- `mineru-kit parse` 执行一次性解析。
- parse-server worker 调用 parser 执行实际解析。
- doclib worker 使用 `flash` 做快速索引或调用本地轻量解析。

非目标:

- 不管理 doclib 缓存。
- 不做 watch、搜索、配置持久化。
- 不隐式启动 doclib server。
- 不在未显式配置远端时上传文件。

## 当前公开导出

`mineru.parser` 当前导出:

| 名称 | 类型 | 说明 |
|------|------|------|
| `parse` | function | 根据文件后缀和参数构造 `MinerUParser` 并执行解析。 |
| `parse_async` | function | `parse` 的异步版本。 |
| `MinerUParser` | class | 统一解析器，支持 PDF、OFD、EPUB、HTML、图片、CSV、RTF、OOXML 与 OpenDocument。 |
| `ParseResult` | dataclass | 解析结果对象。 |
| `MinerUApiParser` | class | API-backed parser，详见 [API-backed Parser](api-parser.md)。 |

## `MinerUParser`

`MinerUParser` 是 `mineru.parser` 中唯一的本地解析器类。过去按文件类型/后端分散的多个 parser 类以及内部路由函数已统一合并到 `MinerUParser`。

导入方式:

```python
from mineru.parser import MinerUParser
```

构造参数（关键字参数）:

| 参数 | 类型 | 说明 |
|------|------|------|
| `tier` | `Literal["flash","basic","standard","advanced"]` | 解析投入档位，对应原 SDK 的 `effort` 概念。 |
| `parse_mode` | `str` | 解析模式，对应原 SDK 的 `parse_mode`/`backend` 概念。 |
| `image_analysis` | `bool` | 是否启用图片分析。对应原 SDK 的 `disable_image_analysis` 取反语义。 |

`MinerUParser` 根据 `tier` / `parse_mode` / `image_analysis` 以及输入文件后缀，在内部选择具体的 PDF、EPUB 或结构化文档解析路径，不再要求调用方分别实例化不同 parser 类。

支持输入:

- PDF
- EPUB（严格按 OPF spine 输出逻辑页并保留可解析的作者内部链接）
- 图片（PNG/JPEG 等）
- CSV、RTF、DOC/DOCX、PPT/PPTX、XLS/XLSX
- ODT、ODS、ODP
- HTML、HTM（静态源码、固定 `auto` 正文选择）
- OFD（原生固定版式文字、图片、全线表格和 bbox）

EPUB 始终使用 `flash/txt` 并只支持整本解析。每个 `page_idx` 与 OPF spine 项一一对应；位于 spine 中的 navigation XHTML 按普通内容页解析，spine 外的 nav/NCX 不生成合成目录页。正文、列表和表格中的安全内部链接统一映射到实际标题 anchor；严格匹配单一标题的目录表格行会让相邻标题单元格继承同一跳转。单条 Footnote/Endnote 输出为带可选 canonical anchor 的 `page_footnote`，正文 `noteref` 可跨 spine 正向链接到该 anchor；脚注集合容器不会整体改型。spine XHTML 先按严格 XML 解析；局部标签闭合错误会记录 warning，再经过禁用网络的受限 XML 安全审计与 HTML DOM 恢复。恢复后仍要求 `html` 根节点并重新执行节点数、深度预算；DTD 实体、加密、资源超限以及 OPF/container/SVG 等控制或固定版式 XML 不进入宽松回退。恢复仍失败时保留空 spine 占位。fixed-layout/SVG 仅尽力保留 DOM 文本与包内静态图片，不执行 OCR、脚本、外部资源下载或 DRM 解密。所有非 PDF Flash 格式显式传入 `page_range` 都会返回 `page_range_invalid`。

HTML 始终使用 `flash/txt` 并输出一个无 bbox 的逻辑页。解析器不执行 JavaScript 或远程资源下载；固定 `auto`
模式只在正文候选置信度和保留率均达标时裁剪导航噪声，否则保留完整 body。本地相对图片限制在源文件目录内，
远程图片仅作为安全 HTTP(S) `image_url` 保留。完整约束见 [ADR-0031](../decisions/0031-html-native-semantic-parsing.md)。

OFD 始终使用 `flash/txt` 并按 `OFD.xml` 的 DocBody 顺序和 `Document.xml` 的 Pages 声明顺序输出物理页。
文字 bbox 由 TextCode、Delta、CGTransform、CTM 和字体指标恢复，不把 TextObject Boundary 直接当作 tight bbox。
首版支持常见位图与高置信全线表格；不执行扫描 OCR、JBIG2 解码、签章验签或纯路径页面渲染。完整约束见
[ADR-0032](../decisions/0032-ofd-native-fixed-layout-parsing.md)。

## `parse()` / `parse_async()` 入口

`parse()` 与 `parse_async()` 是面向用户的便捷函数。它们在内部直接构造 `MinerUParser`，不再经过额外路由函数。

公开签名保持稳定:

```python
from pathlib import Path
from typing import Literal
from mineru.parser import ParseResult

def parse(
    path: str | Path,
    *,
    tier: Literal["flash","basic","standard","advanced"] | None = None,
    parse_mode: str | None = None,
    image_analysis: bool = False,
    **kwargs,
) -> ParseResult: ...

async def parse_async(...) -> ParseResult: ...
```

设计规则:

- `tier` / `parse_mode` / `image_analysis` 与 `MinerUParser` 的构造参数一致。
- 返回值始终是 `ParseResult`。
- `parse_async()` 默认可以通过线程池调用同步实现。

`mineru-kit api-server` 内部应复用 `parse_async()`，避免在 server 层重复维护 parser dispatch 和兼容规则。

## `DocumentParser`

`MinerUParser` 实现统一 parser 接口:

```python
class DocumentParser:
    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult: ...
    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult: ...
    def parse_batch(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]: ...
    async def parse_batch_async(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]: ...
    def close(self) -> None: ...
```

约束:

- `parse()` 必须检查输入路径是否存在。
- `parse_async()` 默认可以通过线程池调用同步实现。
- `parse_batch()` 默认按顺序解析；能批处理的 parser 可以覆盖。
- parser 可以作为 context manager 使用，退出时调用 `close()`。

## 重依赖边界

公开 import 不应触发 torch、transformers、模型权重或解析 server 启动。重依赖应在 parser 构造或执行时惰性加载。

目标:

- `from mineru.parser import parse, ParseResult` 应足够轻。
- `MinerUParser` 只在执行对应解析路径时加载重依赖，不应在模块 import 时加载。

## 示例

```python
from mineru.parser import parse

result = parse("report.pdf", tier="basic", page_range="1~5")
print(result.markdown())
```

高级用法:

```python
from mineru.parser import MinerUParser

with MinerUParser(tier="basic") as parser:
    result = parser.parse("report.pdf", page_range="1~10")
    images = result.images()
```

## 未决问题

`parse_batch()` 进度回调等未决项集中维护在 [开放问题清单](../open-questions.md)。
