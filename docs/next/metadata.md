# 原始文档元数据

文件格式元数据的唯一提取实现归 DocVortex。Doclib 入库通过 `docvortex.extract_metadata()`
获取标题、作者、主题、关键词、语言和计数；MinerU 各解析入口使用相同接口，并通过
Model/Middle JSON 的可选 `metadata.document` 传递完整属性。

源属性在 PDF 选页重写之前保存；其中 `page_count_kind` 区分物理页、声明布局页和结构单位，
不应把选页数量、幻灯片数或工作表数当作整本 PDF 页数。Doclib 调度页数保留原有产品语义。
图片转 PDF 不在本次范围内提取 EXIF，也不将转换器生成的信息冒充原图属性。

`metadata.producer` 仍是 JSON 生产者，原文件生成软件位于 `metadata.document`。
JSON 协议保持 2.0；旧数据缺少新字段仍可读取，但旧版严格读取器需要先升级。
完整扩展信息保存在解析 JSON 中，不增加数据库列或详情 API 字段，不强制回填旧缓存。

Doclib 仅负责数据库映射、截断和调度。作者以 `; ` 拼接，关键词以 `, ` 拼接，语言取首项。
解析完成后非空源属性更新现有列及 FTS 标题作者，不按推理 tier 判断源属性质量，也不清空旧值。
元数据不同的缓存批次保持原有不合并规则。

本次移除 17 项已经归属 DocVortex 的基础直接依赖；这些包仍随 DocVortex 安装。
测试专用的 pypdf、reportlab、lxml 在 test extra 中显式声明。tokenizers 仍是基础依赖，
因为无 Transformers 的 ONNX 公式识别会直接使用它。

发布顺序：先验证并发布包含公共接口的 DocVortex 0.2.5，再将 MinerU 最低版本要求提升到该版本。
本地集成验收使用两边构建的 wheel，不能用旧版 DocVortex 验证新调用链。
