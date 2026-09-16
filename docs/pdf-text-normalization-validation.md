# PDF 输出全角英数清洗下沉验证

DocGale 内容层现在提供 `normalize_pdf_model_text(model_list) -> None`，其公开
PDF 模型出口自动调用。MinerU 在回填、Span 构造及宿主元数据清理后复用同一
实现；原有标题分类、无效块过滤及类型转换继续留在 MinerU。

范围为 PDF 自然语言和表格单元格的全角英文字母、数字；保留标点、公式、代码、
链接目标及结构。共享清洗单测迁入 DocGale，MinerU 保留调用时机及所有 PDF
路由的整合测试，包括 Flash 的 auto/txt/ocr，以及 medium/high/xhigh 的 txt/ocr。

本次完整回归：3,954 passed、4 skipped、4 项既有基线失败 deselected。
测试总数变化来自迁移 25 项共享清洗用例，同时新增 7 项宿主整合用例。
原始字符、字体及几何没有参与全半角转换，现有 Hybrid 回填断言继续通过。
非 PDF 路径不触发新清洗；ModelJson/MiddleJson 协议和 CLI/API 参数保持不变。

DocGale 的 Python 3.10–3.14 三平台 CI、Pydantic 下限和 PDFium 5.13.0 验证：
https://github.com/myhloli/docgale/actions/runs/34034656162

旧 Doclib 缓存继续有效。升级后重启正在运行的服务，再通过 `--force` 重新解析
以获得新输出；读取旧结果不会自动清洗或改写。完整规则见 DocGale 的
`docs/PDF_TEXT_NORMALIZATION.md`，字形及文字对照见其 `docs/validation.md`。

四项既有排除仍为：
- test_basic_extra_includes_preflight_runtime_dependencies
- test_standard_is_the_highest_model_runtime_extra
- test_doclib_compaction_rejects_unknown_legacy_schema
- test_pp_formulanet_fix_latex_uses_shared_mathring_repair
