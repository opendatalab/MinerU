# DocVortex 0.4 配套迁移

MinerU 使用 `docvortex>=0.4.0,<0.5.0`。本轮调整 Python 模块边界，保持文档协议、解析路由、模型回退与渲染语义。

## 公开入口

| 原入口或职责 | 新入口 |
| --- | --- |
| foundation 几何及 PDF 通用坐标原语 | `docvortex.geometry` |
| foundation 文本与链接规则 | `docvortex.content.text`、`docvortex.content.links` |
| 图像统计、素材编码与路径校验 | `docvortex.assets` |
| PDFDocument、PDFPage、字符和几何契约 | `docvortex.document.pdf` |
| HTML 来源上下文 | `docvortex.document.contracts.HtmlSourceContext` |
| 原生文字证据、区域表格及 OCR 文本投影 | `docvortex.analyzers.pdf` |
| 模型文本、坐标自动判断与虚拟 OCR token | MinerU PDF 分析层 |
| xhigh layout 容器补全 | MinerU PDF 分析层 |
| 推理平台支持判断 | MinerU model/runtime |

`docvortex.public_api.PUBLIC_API` 是静态模块及符号清单。新增跨库依赖必须同时更新清单、文档和契约测试；不能通过基础实现、私有模块或动态别名绕过边界。

## 行为与资源所有权

`prepare_text_evidence` 返回具名的 `PDFTextEvidence`，`apply_text_evidence` 统一完成链接、样式、脚本与 InlineSpan 物化。MinerU 仍决定调用时机、公式排除区域、超大字符页和 OCR 回退。

`prepare_table_page` 仅在存在候选表格时调用；同页复用其物化数据。`recover_table_region` 返回最终 HTML、来源、置信度和诊断，`None` 表示没有可接受结果。结构恢复异常以 `PDFTableRecoveryError` 保留原始原因，交给宿主回退；HTML 物化异常继续传播，避免改变旧异常边界。页面几何可以复用，证据不持有 PDFium 裸句柄。

通用 `convert_bbox` 显式声明 `unit`、`pixel` 或 `point`；`page_size` 使用 point，`render_scale` 是每 point 的像素数。模型 bbox 的自动解释与三位小数规则保留在 MinerU。共享 PDFium 锁、进程池和资源关闭职责仍归 DocVortex。

## 升级与回退

本轮直接删除被替代的旧门面及重导出，不提供过渡版本或旧 pickle 路径兼容。JSON、Bundle、HTTP API 和输出格式保持既有契约。

配套升级两个 wheel；旧版 MinerU 必须显式约束 `docvortex<0.4.0`。旧版依赖上限 `<1` 无法阻止解析器选择 0.4，因此不要只升级 DocVortex。回退时同时恢复旧版 MinerU 与其兼容的 DocVortex。

## 验证

`tools/validate_docvortex_boundary.py` 可对指定两仓库源码或当前安装包捕获真实 PDF 的中间协议与全部渲染。记录实际导入路径、样本 SHA256 和页数；比较时只允许忽略 DocVortex producer 版本、DOCX/EPUB 容器时间字段，以及经独立复算确认由 producer 版本派生的 EPUB 标识。

MinerU 的 `DocVortex boundary integration` 手动工作流接受未发布的 0.4 wheel URL，以及可选的 mineru-vl-utils 配套 wheel URL，在三种操作系统及 Python 3.10、3.13、3.14 上验证边界、协议和固定推理结果。DocVortex 自身 CI 验证独立安装与基础算法。配置工作流不代表已经通过远端 CI。
