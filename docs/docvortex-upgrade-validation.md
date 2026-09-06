# DocVortex 接入与标点规则验证

独立引擎已更名为 DocVortex，MinerU 依赖为 `docvortex>=0.1.0,<1.0.0`。
导入及集成模块同步迁移，公开文档类型继续由引擎唯一维护。MinerU 的 schema 2.0、
产品元数据、已有旧结果适配、OCR/VLM/Hybrid 路由与 UI 名称保持不变。

PDF 自然语言及表格可见文本新增 `：．／＼－＿％＋＝＠＃＆＊` 的半角转换。
中文标点、公式、代码、URL 目标、HTML 属性、原始字符和几何保持原有边界。
通用 `full_to_half_exclude_marks()` 仍只转换英数。

本地完整回归结果为 **3954 passed、4 skipped、4 deselected**。四项排除保持此前基线：

- `test_doclib_app_startup.py::test_basic_extra_includes_preflight_runtime_dependencies`
- `test_doclib_app_startup.py::test_standard_is_the_highest_model_runtime_extra`
- `test_legacy_schema_adapter.py::test_doclib_compaction_rejects_unknown_legacy_schema`
- `test_mfr_latex_utils.py::test_pp_formulanet_fix_latex_uses_shared_mathring_repair`

DocVortex [三平台 CI](https://github.com/myhloli/docvortex/actions/runs/34045249479)
通过 Python 3.10–3.14、PDFium 5.10.1/5.13.0 及 Pydantic 最低版本验证。
Linux/macOS 为 418 项通过；Windows 为 417 项通过、1 项 POSIX 检查跳过。
既有少线表诊断仍单独失败，不计入本次修复。独立 Python 3.14 wheel 环境中
107 项协议、格式、标点与完整流程检查通过，且没有 DocGale、MinerU 或 pdftext。

中文论文3/4分别新增 57/239 个符号转换，三平台文档结构、表格属性、原始几何和
全部 78 张源页/Layout PNG 与修改前一致。更名阶段的两份重排版 PDF 共 12 页，
Poppler 渲染像素与更名前一致。HTML、EPUB、MathJax、Mermaid、代码高亮及素材加载
已进行浏览器检查，PDF 页面截图已审阅。

仅 DocVortex 原生 JSON/结果包可由新包直接读取；旧 DocGale 原生协议明确拒绝，
旧 DocGale/MinerU HTML 按普通网页解析。MinerU 自身已有缓存和支持的历史 JSON
仍由宿主适配器读取，但不会自动重新清洗文字。需要当前 PDF 文本结果时重启服务并
使用现有 `--force` 重新解析。

仓库为 `myhloli/docvortex`，保留原仓库历史与 Release。0.1.0 wheel/sdist 更新到
发行草稿，本次不发布到 PyPI；Trusted Publisher 设置见引擎升级说明。
