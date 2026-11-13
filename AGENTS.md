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

## 需求、设计与修复审阅

本节适用于 issue 分析、PR review、代码审查及日常项目逻辑讨论。审阅目标是判断用户问题应如何解决，不能仅证明某个补丁符合当前实现。

### 先审需求，再审实现

- 分别回答四件事：用户真正要完成什么、这项能力是否应当存在、现有行为是否违背合理需求、提交的方案是否解决了问题。不要把这四个判断合并成“测试通过，所以应该合入”。
- 先还原具体用户、输入、操作、预期产物与失败后果，再判断属于真实缺陷、体验改善、能力扩张，还是尚未裁决的设计取舍。真实缺陷不自动等于必须立即修复。
- 现有代码、测试、文档和 Accepted ADR 是理解历史意图与兼容成本的证据，不是设计必然合理的证明。应同时挑战 PR 的前提和原有逻辑；用户当前明确的产品决策优先于历史约定。
- 对新增或已有能力追问：是否有独立使用场景，能否通过现有接口组合完成，是否重复已有职责，维护成本是否超过收益。允许得出“不增加功能”“收窄、合并或移除入口”的结论，不能默认不断补齐边角行为。
- 对用户的质疑也做独立判断；有证据支持的合理设计应说明理由，不能为了认同质疑而反向接受未经验证的假设。

### 区分症状、根因与责任边界

- 从真实入口追踪调用链、状态、输出和错误传播，区分触发原因、使问题扩大的机制及最终可见症状。缺失原始异常、日志或现场时，应明确未知，不能把最后一条报错当作根因。
- 涉及线程、异步任务、引擎和子进程时，分别检查它们与宿主进程的生命周期：谁感知失效、停止接单、处理在途任务、失效缓存并负责恢复。不能用进程仍存活推断服务可用，也不能让共享库擅自终止整个宿主进程。
- 区分正常的局部回退与全局能力丢失。核心依赖失效后，不能仅靠吞掉异常、重试或更弱后端维持成功状态；是否降级及如何报告必须来自明确的产品语义。
- 文件名、后缀和模式匹配只是线索；需核对实际路径类型、内容签名及容器结构，并验证误判与漏判。复用已有启发式只能证明实现一致，不能证明策略正确。
- 将输入识别、解析有效性和批次失败策略分开讨论。显式文件与目录发现可以采用不同错误策略，但应说明跳过、单项失败、继续处理和全局终止的边界；不能将所有异常静默跳过后报告完整成功。
- 区分渐进阅读、保存当前片段和导出指定范围的完整产物。不要因为选项都叫 format/output，就默认它们应共享字符预算、续读、素材物化与覆盖语义。
- 单文件与批量语义应明确取决于原始输入形态、展开后的数量还是显式模式。已有文件是否可覆盖应结合该语义和用户指定的输出目标判断，不能仅凭文件被覆盖就宣称数据损坏，也不能因当前恰好只有一个输入就隐式改变输出契约。

### 在设计取舍之后选择修复范围

- “最小修复”以需求和职责边界已经合理为前提。应比较维持现状、不修或移除冗余能力、局部修复、统一或重构原有设计；优化整体规则数量、状态复杂度和维护成本，不只比较改动行数。
- 原有设计合理且问题局部时，优先范围明确的修复；原有语义冲突或职责混杂时，先给出设计判断，再选择修复，不能把历史行为一律当作不可挑战的兼容要求。
- 紧急止血与完整治理可以分阶段，但须说明补丁解决什么、仍遗漏什么，以及依赖发布、部署、生命周期接入等完成条件。避免把异常分类补丁等同于根因修复或完整故障恢复。
- 分析可以挑战原设计，实施仍遵守用户授权范围。用户要求只分析或只修指定事项时，不自行改动其他功能；涉及新的产品语义时先陈述依据、取舍与建议，不把讨论中的假设直接写成既定规则。

### 用反例验证结论

- 先按项目实际支持的系统、后端及用户目标确定验收矩阵。Windows 的 LMDeploy 验收以 TurboMind 为支持基线，PyTorch 后端不作为通过条件。故障处理任务应验证失效传播、拒绝接单、任务清理和恢复，不自行扩展成修复所有崩溃根因或打通不受支持组合；额外诊断失败不自动成为发布阻塞项。
- 复现现象后，还要验证“预期行为”本身是否合理；新增测试若只是固定 PR 作者的假设，不能用其失败证明产品必然有缺陷。
- 除正向样例外，优先保留能推翻方案的反例：合法但命名特殊的输入、名称正常的非法输入、单项与批量切换、部分成功、依赖中途失效、混合错误顺序，以及既有正常回退。
- 测试替身应保持真实协议、异常继承、调用顺序及生命周期。核对运行的代码快照和依赖版本；区分静态推断、合成重放、真实解析与真实设备验证。
- 故障注入要核对实际执行阶段，不能用任务的 running 状态代替底层引擎正在推理的证据。区分引擎失效但宿主存活、进程崩溃和测试主动破坏资源；不安全的注入制造出的崩溃不能当作原事故根因，也不能靠放宽断言把另一类恢复算作原场景通过。
- 验收应同时检查报告、完整日志和真实产物。报告成功后仍出现未处理异常、错误被重复记录或只验证了错误的执行阶段，都需要单独调查；已处理的失败不得通过屏蔽日志伪装成正确收尾。
- 测试数量、构建成功和 CI 通过只说明相应验证范围，不能替代需求必要性、方案最优性或完整交付的判断。
- 审阅结论分别交代需求是否成立、现有设计是否应保留、缺陷与优先级、方案评价、替代方案及证据限制；可以明确建议暂缓裁决、仅作止血或重新设计，不强行把每个 PR 归为应接受的小修复。

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

共享文档协议是带 `schema` 身份的版本 2.0（Python 属性名为 `schema_id`，JSON 序列化键为 `schema`）。MinerU 直接消费共享类型：`ParseResult` 持有 `MiddleJson` 读写，不维护独立的协议封装层。MinerU 通过 `extensions["mineru"]` 记录实际执行的 `tier` 与 `parse_mode`，由 `mineru.integrations.docvortex.MinerUMetadata` 严格校验（`extra="forbid"`，不接受其他字段，没有 `effort`、`mineru_version` 字段）；生产者版本位于 `metadata.producer.version`。

历史协议兼容限定在 Doclib 持久化缓存的读取边界（`mineru.doclib.core.middle_json.read_cached_middle_json`），不扩展为通用 SDK/API 的兼容承诺；3.4.5 旧页面转换由 `mineru.backend.postprocess.legacy_schema_adapter` 在该边界内维护。协议细节以 `docs/next/middle-json/` 为唯一参考，兼容行为对应测试为 `tests/unittest/test_doclib_legacy_schema_adapter.py`。产品字段校验和旧结果兼容均归 MinerU，DocVortex 不提供兼容模块或旧路径别名。不要为旧底层构造参数增加动态兼容别名。

ModelJson 仍保存 raw pages 与 page_index_map；MiddleJson 仍保存有序 PageInfo 数组。PageInfo 只有 page_idx 与 blocks。Block/InlineSpan 的现有语义、几何和父子约束保持不变；自然语言 InlineSpan 不携带字体或几何信息。

### 4. 后处理与渲染

DocVortex 的确定性后处理独立构造有效 MiddleJson。`mineru.backend.postprocess.document` 在其后显式执行 MinerU 的可选 LLM 增强。

`mineru.render` 保留九种输出：Content List V1/V2 实现及专用选项归属 MinerU，其余七种 renderer 和通用选项来自 DocVortex。两边各自定义 RenderFormat，RenderMode 与共享文档类型仍复用。Content List 通过 `docvortex.render.fragments` 调用共享片段能力，不导入私有实现。公式定界符等宿主配置显式传入，不让 DocVortex 读取 MinerU 配置。

九种输出为 Markdown、HTML、LaTeX、DOCX、EPUB、PDF、Structured Content、Content List V1/V2。LaTeX/EPUB/PDF/Content List 的低层能力不自动扩展所有产品入口。PDF 输出布局由 `PdfLayout` 控制：`AUTO`（默认，有几何信息时保留原始布局，缺失时回退语义重排）、`ORIGINAL`（仅原始布局）、`REFLOW`（仅语义重排）。

文件写出属于 `docvortex.export`；语义类型不再提供文件导出方法。结果包保存中间协议和物化素材，渲染不依赖已关闭的 PDFium 对象或源文件，也不修改原始语义树。

### 5. PDF 与依赖方向

PDF 字符、片段、行和矩形类型由 DocVortex 维护；项目不再依赖 pdftext，也没有其 0.6/0.7 运行时分支。pypdfium2 最低版本为 5.10.1，DocVortex 约束为 `<6`，MinerU 安装时共同遵守该范围。

访问同一 PDFium 运行时必须使用 DocVortex 的共享锁及资源管理。跨进程传递字节和物化数据，不传递裸句柄。

- DocVortex：schema/foundation → document/content → analyzers/postprocess/render/export → api/result。
- MinerU：共享 DocVortex 能力 → model/backend → parser/kit/doclib/cli。
- `model/runtime` 继续负责 MinerU 设备、显存、ONNX 与本地模型生命周期；`model/registry.py`、`model/download.py` 保持产品模型管理职责。
- 已迁移的 Flash、通用后处理、renderer 私有目录及共享 leaf utilities 不在 MinerU 中保留第二份实现。
