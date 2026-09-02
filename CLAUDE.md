# MinerU 项目编码规范

## 开发环境

使用 **uv** 管理 Python 虚拟环境和依赖。

```bash
# 在项目根目录 MinerU-Repo 中创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e ".[core]"
```

运行 Python 代码须在项目根目录 `MinerU-Repo` 中执行：

```bash
.venv/bin/python -m mineru.path.to.submodule
```

## GitHub Issue / PR 处理规范

处理 GitHub issue 时，commit message 和 PR body 中不要使用会导致 issue 在 merge 后自动关闭的关键词。

禁止使用：
- `fixes #123` / `fix #123`
- `closes #123` / `close #123`
- `resolves #123` / `resolve #123`

需要关联 issue 时，使用不会自动关闭 issue 的表述：
- `Refs #123`
- `Related to #123`
- `Issue: #123`

## Import 规范

mineru 模块内部子模块之间的引用统一使用 **relative import**，不使用 `import mineru.xxx` 形式的绝对导入。

```python
# 正确 — relative import
from .base import DocumentParser
from ..render import RenderMode

# 错误 — 项目内不允许 absolute import 引用自身模块
from mineru.api.base import DocumentParser
from mineru.render import RenderMode
```

只引用外部第三方库时使用 absolute import（如 `from loguru import logger`）。

## 格式化

- 格式化工具：**ruff**
- 保存时自动格式化：开启
- 保存时自动 organize imports：开启
- 行宽：**128**

## Lint 规则

启用的规则集：
- **C** — 复杂度/编码规范 (mccabe, pycodestyle 约定等)
- **E** — pycodestyle 错误
- **F** — pyflakes 检查
- **W** — pycodestyle 警告
- **ANN** — flake8-annotations（类型注解）

忽略的规则：
- **C901** — 函数过复杂（允许必要的复杂函数）
- **ANN002** — `*args` 不需要类型注解
- **ANN003** — `**kwargs` 不需要类型注解
- **ANN204** — 特殊方法（如 `__init__`）不需要返回类型注解
- **ANN401** — 允许使用 `Any` 类型标注

## 编程原则

### 确定性优先

agent 通过静态分析理解代码。避免运行时注册、猴子补丁、`globals()` 等动态模式，使用显式映射。

```python
# 差 — agent 无法静态追踪
registry.register(MyParser)

# 好 — 显式映射
TIER_TO_EFFORT: dict[Tier, Effort] = {
    "flash": "flash",
    "basic": "low",
    "standard": "medium",
    "advanced": "high",
}
```

### 入口一致性

每个模块的 `__init__.py` 必须明确导出 `__all__`，让 agent 能快速获取模块边界。

```python
# mineru/api/__init__.py
from .base import DocumentParser
from .parse_result import ParseResult

__all__ = ["DocumentParser", "ParseResult"]
```

### 类型优先

公开函数必须有完整类型注解。避免 `**kwargs: Any` 透传内部配置。

```python
# 差 — agent 不知道合法参数
def parse(path, **kwargs): ...

# 好
def parse(
    path: str | Path,
    *,
    tier: Tier = "standard",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_range: str = "",
) -> ParseResult: ...
```

### 副作用隔离

模块级别不得有隐式副作用（读取环境变量、创建目录、注册 handler 等）。`import` 只应定义符号。

```python
# 差 — import 时执行
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# 好 — 使用时执行
def parse(...):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
```

### 惰性加载

对外 API 层的 `import` 不得触发重依赖（torch、transformers 等）。重依赖应在函数体内按需导入。

```python
class MinerUParser(DocumentParser):
    def __init__(self, *, tier: Tier = "standard", **kwargs):
        super().__init__(**kwargs)
        from ..backend.analyze import doc_analyze  # 惰性加载

        self._analyze_fn = doc_analyze
```

## Middle JSON Schema 2.0 架构

Middle JSON 已收敛为 schema 2.0 的统一结构，并使用结构化 InlineSpan 表达自然语言行内语义，不再有 pipeline/vlm/office 三套独立实现。

### 1. 统一分析入口

`backend/analyze.py:doc_analyze()` 是 PDF、OFD、EPUB、HTML、CSV 与 Office 文档的唯一公共入口，通过 `file_suffix` 路由到 `backend/analysis/pdf/pipeline.py:analyze_pdf`、`backend/analysis/ofd.py:analyze_ofd`、`backend/analysis/epub.py:analyze_epub`、`backend/analysis/html.py:analyze_html`、`backend/analysis/csv.py:analyze_csv` 或 `backend/analysis/office.py:analyze_office`，生成严格 `ModelJson` 后统一交给 `backend/postprocess/document.py:model_json_to_middle_json()` 构造 `MiddleJson`。`MinerUParser` 是 `DocumentParser` 的唯一实现，替代旧的 `PdfHybridParser`/`PdfFlashParser`/`DocxParser` 等。

### 2. MiddleJson 顶层字段（schema 2.0）

`mineru/types.py:MiddleJson` 严格定义：

| 字段 | 类型 | 说明 |
|------|------|------|
| `pages` | `list[PageInfo]` | 严格按 `page_idx` 升序的页面数组 |
| `is_full_document` | `bool` | 是否整本文档解析（空 `page_index_map` 时为 `True`） |
| `file_suffix` | `Literal["pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "csv", "epub", "html", "ofd", "odt", "ods", "odp"]` | 输入文件类型 |
| `effort` | `Literal["flash", "medium", "high", "xhigh"]` | 分析强度 |
| `parse_mode` | `Literal["txt", "ocr"]` | 解析模式 |
| `mineru_version` | `str` | MinerU 版本号 |

schema 2.0 输出不再有 `_backend`/`_version_name`/`pdf_info`/`_ocr_enable`/`_vlm_ocr_enable` 等旧字段；这些字段只能作为受支持旧 payload 的迁移输入出现。

### 2.1 ModelJson 严格容器

`doc_analyze` 返回 `tuple[MiddleJson, ModelJson]`。`ModelJson` 持有 raw model-list（`pages: list[list[dict]]`）+ `page_index_map: list[int]`（空列表表示整本文档），并提供 `is_full_document`/`resolved_page_indices` 派生属性。只有 PDF Analyze 接受非空 `page_index_map`；其它 Flash 格式只支持整本解析。EPUB 逻辑页严格对应 OPF spine 顺序，不再前插合成目录页；位于 spine 中的 navigation XHTML 作为普通内容页保留，spine 外的 nav/NCX 不生成页面。`backend/postprocess/document.py:model_json_to_middle_json()` 是 ModelJson → MiddleJson 的唯一编排入口，内部调用 `model_json_to_pages()` + `apply_llm_aided_postprocess()`。

### 3. PageInfo 结构

`PageInfo` 只有两个字段：`page_idx: int`（从 0 起）+ `blocks: list[PageBlock]`。不再有 `page_size`/`preproc_blocks`/`para_blocks`/`discarded_blocks`/`images`/`tables`/`interline_equations`/`_layout_tree`/`layout_bboxes` 等字段——所有内容统一在 `blocks` 树里表达。

### 4. Block 类型体系

`mineru/types.py` 使用严格 Pydantic 模型替代旧 `Block`/`Line`/几何 `Span` dataclass：

- 自然语言块（正文、标题、页眉页脚、脚注、视觉注释与算法正文）持有 `content: list[InlineSpan]`
- `InlineSpan` 是 `TextSpan`/`EquationInlineSpan`/`CodeInlineSpan`/`HyperlinkSpan` 的严格联合，只表达行内语义，不携带 bbox、score、字体或图片字段
- `EquationBlock`、图片/表格/图表 body 与普通 `CodeBodyBlock` 等专用叶子继续使用各自的字符串或图片载荷字段
- 视觉父块（`ImageBlock`/`TableBlock`/`ChartBlock`/`CodeBlock`）持有 `content: list[child blocks]`，子块包含唯一 body + 可选 caption/footnote
- `BlockType.INTERLINE_EQUATION` 已重命名为 `BlockType.EQUATION`
- 不再生成旧版独立 `Line`/几何 `Span` 类型及行级、span 级 bbox

### 5. 统一 render 入口

`render/api.py:render(middle_json, output_format, options)` 是唯一渲染入口，支持：

- `RenderFormat.MARKDOWN` → `render_markdown`
- `RenderFormat.HTML` → `render_html`
- `RenderFormat.LATEX` → `render_latex`
- `RenderFormat.DOCX` → `render_docx`
- `RenderFormat.EPUB` → `render_epub`
- `RenderFormat.STRUCTURED_CONTENT` → `render_structured_content`
- `RenderFormat.CONTENT_LIST` → `render_content_list`
- `RenderFormat.CONTENT_LIST_V2` → `render_content_list_v2`
- `RenderFormat.PDF` → `render_pdf`

`render/_internal/` 下按目标格式分目录组织共享逻辑（`common/`/`markdown/`/`html/`/`latex/`/`docx/`/`epub/`/`pdf/`/`content_list/`/`structured_content/`），顶层同名模块只是惰性公共门面。InlineSpan 的规范化、连接、裁剪和可见文本操作归 `backend/postprocess/inline.py`，renderer 按 Span discriminator 分派，并只能单向依赖该模块和 `backend/postprocess/table_merge`。不再有 `pipeline_union_make`/`vlm_union_make`/`office_union_make` 三套逻辑；Content List V1/V2 是由严格 MiddleJson 派生的正式兼容输出，不恢复旧 backend 并行实现。LaTeX、EPUB、PDF 与两套 Content List 只扩展低层 `mineru.render` 公共面，不自动扩展 ParseResult、CLI、API Server 或 doclib 输出合同。

### 5.1 目录职责

- `model/runtime/` 负责设备、显存、ONNX 与 Hybrid 本地模型生命周期；模型仓库和下载分别位于 `model/registry.py`、`model/download.py`。
- `model/flash/pdf/` 负责 PDFDocument、PDFium、原生文本、样式和表格恢复；`model/flash/ofd/` 负责 OFD 包、资源、固定版式场景与阅读顺序；`model/flash/epub/` 负责 OCF、OPF、spine 与 XHTML/SVG；`model/flash/csv.py` 负责分隔符文本解析；`model/flash/office/` 负责十类 Office/RTF/ODF 格式。
- `utils/` 只保留 geometry、image、image payload、language/text、URL/超链接目标校验与共享安全 scheme 集合、platform 和 stdio 等无业务依赖的叶子能力；是否允许相对链接、fragment 等格式策略由调用方显式选择，活动代码不得把格式解析或其它业务实现放入 utils。
- `utils/types` 共同组成基础能力层，`types.py` 可以复用不依赖上层模块的 leaf utility；稳定依赖方向为 `utils/types → model → backend → render → parser/kit/doclib`，禁止上层模块被基础层反向引用。

### 6. ParseResult 与 MiddleJson 的关系

`ParseResult` 持有 `MiddleJson` 实例，`pages` property 委托给 `middle_json.pages`。`to_dict()` 只输出 schema 2.0；`from_dict()` 直接读取 2.0，并将 MinerU 3.4.5 `pdf_info` 与对应 schema 1.0 `pages` 包装经 `backend/postprocess/legacy_schema_adapter.py:legacy_page_to_model_list` 单向回推为 raw model-list 后重走统一后处理。其它未知版本旧 payload 不自动迁移，必须从源文件重新解析。
