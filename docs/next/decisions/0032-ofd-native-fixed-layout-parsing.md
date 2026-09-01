# ADR-0032: OFD 原生固定版式 Flash 解析

状态: Accepted，已实现
日期: 2026-08-28
相关文档: ../architecture.md, ../sdk/parser.md, 0024-file-type-tier-normalization.md

## 背景

OFD 是 ZIP/XML 固定版式文档。页面具有毫米坐标、模板、图层、TextCode、PathObject、ImageObject 与资源作用域，
XML 对象顺序主要表示绘制顺序而非阅读顺序。将 OFD 归入 Office 或预先转换为 PDF 会丢失原生字符位置、路径和资源信息，
也会引入外部转换运行时。

## 决策

- OFD 作为独立 `file_suffix="ofd"` 进入原生 `OfdModel -> analyze_ofd -> ModelJson -> MiddleJson` 链路，
  不加入 `OfficeSuffix`，不执行 OFD 到 PDF 转换。
- OFD 固定使用 `flash/txt`、整本文档解析；多个 DocBody 与页面严格保持声明顺序，显式 page_range 返回
  `page_range_invalid`。
- 解析器直接受限读取 ZIP/XML，支持标准及已知旧命名空间、1.0/1.1/1.2 与兼容的其它 1.x，禁止网络、DTD、实体、
  包外路径和加密成员。
- 页面内部保留毫米坐标、Affine 与 quad。文字 bbox 由 TextCode、Delta、CGTransform、CTM 和可用的内嵌字体指标恢复；
  Boundary 只作为容器与裁剪边界。
- 模板、页面和图层先组合为页面场景；高置信全线表使用 OFD 自身路径与 glyph 几何恢复。其余正文、图片、表格经
  OFD-aware 预分类后使用共享 XYCut++ 恢复阅读顺序。
- 所有顶层 OFD block 必须携带相对 PhysicalBox 的归一化 bbox；继续复用现有 Block、renderer 和 Middle JSON 2.0，
  不增加 OFD 私有输出字段。
- PNG/JPEG/BMP/TIFF/WebP 可作为图片输出。首版不增加外部 JBIG2、Java、LibreOffice、系统字体或网络资源依赖。

## 兼容与安全

- 必需 OFD.xml、Document.xml、页面 XML、显式声明的 PublicRes/DocumentRes/PageRes，以及已定义并被页面引用的
  模板页 XML 缺失或损坏时整份失败；ImageObject 明确引用的 MultiMedia 资源 ID 或其 MediaFile 成员缺失、无法读取时
  同样失败。
- 对仍可恢复页面自身 Content 的局部兼容性问题采用 best-effort：页面引用未在 CommonData 中定义的 TemplateID 时记录
  `OFD_TEMPLATE_MISSING` warning，跳过该模板并继续解析页面；不支持的 JBIG2、图片仿射、图片解码和可选字体指标继续
  沿用各自的降级语义。
- 未被首期能力读取的 CustomTag、附件、注释与签章 part 不会因自身损坏阻断正文。
- PublicRes、DocumentRes、PageRes 正常应使用文档级唯一 ID；异常重复时按 Page > Document > Public 选择并告警。
- ZIP、XML、资源、展开文字、glyph、路径命令和对象递归均有累计上限，不截断或返回部分文档。

## 首期边界

- 不执行扫描 OCR、Hybrid/VLM 对齐、页面级光栅渲染、JBIG2 解码或纯路径页面视觉还原。
- 不执行加密解密、签章验签、附件导出、注释或 Bookmark/Outline 公共投影。
- 现代 Parser、API Server、mineru-kit 和 Doclib 自动继承 OFD；`cli_old` 与旧 Gradio 不在范围内。
