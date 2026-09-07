# MinerU 项目编码规范

## 开发环境

使用 **uv** 管理 Python 虚拟环境和依赖。

```bash
# 在项目根目录中创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e ".[dev,test]"
```

运行 Python 代码须在项目根目录 `MinerU` 中执行：

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
    "basic": "medium",
    "standard": "high",
    "advanced": "xhigh",
}
```

### 入口一致性

每个模块的 `__init__.py` 必须明确导出 `__all__`，让 agent 能快速获取模块边界。

```python
# mineru/parser/__init__.py
from .base import DocumentParser, ParseResult

__all__ = ["DocumentParser", "ParseResult", "parse", "parse_async"]
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
# 对外 API 层不触发重依赖；轻依赖（如 doc_analyze 所在的 backend.analyze）
# 可模块顶层导入，重依赖（torch、transformers 等）在函数体内按需导入。
class MinerUParser(DocumentParser):
    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        from ..model.flash.pdf.document import PDFDocument  # 重依赖惰性加载

        ...
```

## DocVortex 与 MinerU 的文档架构

### 1. 唯一实现归属

`docvortex` 独立拥有原生文档解析、PDF 基础访问与分类、公共文档类型、确定性后处理、素材、七种通用渲染格式和导出实现，不得反向依赖 MinerU。

MinerU 保留 OCR/VLM/Hybrid 推理、模型生命周期、LLM 增强、tier 策略、CLI/API/Gradio/Doclib。通过 DocVortex 公开接口复用能力，不导入其私有实现。

### 2. 路由

`backend/analyze.py:doc_analyze()` 仍是 MinerU 的统一分析门面。只有 `parse_mode="auto"` 调用共享 `PDFDocument.classify()`。

- Flash + txt：使用 DocVortex 的原生 PDF 模型。
- Flash + auto：txt 分类走 DocVortex；ocr 分类走 MinerU 现有 Flash OCR。
- Flash + ocr：直接走现有 Flash OCR。
- 其他 tier：保持已有推理流程，共享 DocVortex 的基础 PDF、类型、后处理和渲染能力。

原生解析不自行追加分类或 OCR 回退。非 PDF 原生格式仍只支持整本解析，PDF 页范围继续采用 `1-5`、`r1`、`all`。

### 3. 协议与类型

`mineru.types` 重新导出 `docvortex.schema` 的文档类型，并保留 MinerU 产品档位类型。

DocVortex 的 ModelJson/MiddleJson 原生 JSON 使用独立 schema 标识和版本 1.0，持有 `producer` 与 `extensions`。MinerU 的 `effort`、`parse_mode`、`mineru_version` 位于 `extensions["mineru"]`，由 `mineru.integrations.docvortex.build_metadata()` 校验。

MinerU 的 ParseResult、CLI、HTTP API 和 Doclib 通过 `mineru.integrations.docvortex` 读写原有 schema 2.0 封装；3.4.5 旧页面转换由 `mineru.backend.postprocess.legacy_schema_adapter` 维护。产品字段校验、当前封装和旧结果兼容均归 MinerU，DocVortex 不提供兼容模块或旧路径别名。不要为旧底层构造参数增加动态兼容别名。

ModelJson 仍保存 raw pages 与 page_index_map；MiddleJson 仍保存有序 PageInfo 数组。PageInfo 只有 page_idx 与 blocks。Block/InlineSpan 的现有语义、几何和父子约束保持不变；自然语言 InlineSpan 不携带字体或几何信息。

### 4. 后处理与渲染

DocVortex 的确定性后处理独立构造有效 MiddleJson。`mineru.backend.postprocess.document` 在其后显式执行 MinerU 的可选 LLM 增强。

`mineru.render` 保留九种输出：Content List V1/V2 实现及专用选项归属 MinerU，其余七种 renderer 和通用选项来自 DocVortex。两边各自定义 RenderFormat，RenderMode 与共享文档类型仍复用。Content List 通过 `docvortex.render.fragments` 调用共享片段能力，不导入私有实现。公式定界符等宿主配置显式传入，不让 DocVortex 读取 MinerU 配置。

九种输出为 Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content、Content List V1/V2。LaTeX/EPUB/PDF/Content List 的低层能力不自动扩展所有产品入口。PDF 输出继续采用语义重排版。

文件写出属于 `docvortex.export`；语义类型不再提供文件导出方法。结果包保存中间协议和物化素材，渲染不依赖已关闭的 PDFium 对象或源文件，也不修改原始语义树。

### 5. PDF 与依赖方向

PDF 字符、片段、行和矩形类型由 DocVortex 维护；项目不再依赖 pdftext，也没有其 0.6/0.7 运行时分支。pypdfium2 最低版本为 5.10.1，DocVortex 约束为 `<6`，MinerU 安装时共同遵守该范围。

访问同一 PDFium 运行时必须使用 DocVortex 的共享锁及资源管理。跨进程传递字节和物化数据，不传递裸句柄。

- DocVortex：schema/foundation → document/content → analyzers/postprocess/render/export → api/result。
- MinerU：共享 DocVortex 能力 → model/backend → parser/kit/doclib/cli。
- `model/runtime` 继续负责 MinerU 设备、显存、ONNX 与本地模型生命周期；`model/registry.py`、`model/download.py` 保持产品模型管理职责。
- 已迁移的 Flash、通用后处理、renderer 私有目录及共享 leaf utilities 不在 MinerU 中保留第二份实现。
