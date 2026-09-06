# DocGale Droid CJK 接入验证

DocGale 的共享 PDF 运行时现在使用 `droid-cjk-v1`。MinerU 原有 PDF 入口已通过
共享 `PDFDocument`/`pdfium_guard()` 自动接入；未增加高层参数或改变 tier 路由。
渲染 worker 初始化器依次安装父进程退出监控及固定字体。

本次本地验证：3,972 passed、4 skipped、4 项既有基线失败 deselected。
隔离 Doclib home 中，demo1 第一页 Flash 解析成功；第二次命中缓存；`--force`
创建 parse ID 2，`cache_hit=false`。三份 Markdown SHA256 相同。临时服务器已停止。

完整三平台验证和资源归属见 DocGale 的 docs/validation.md 与 docs/PDF_FONTS.md。
DocGale CI（89a3c23）：https://github.com/myhloli/docgale/actions/runs/34028564417

原有四项基线排除为：
- test_basic_extra_includes_preflight_runtime_dependencies
- test_standard_is_the_highest_model_runtime_extra
- test_doclib_compaction_rejects_unknown_legacy_schema
- test_pp_formulanet_fix_latex_uses_shared_mathring_repair
