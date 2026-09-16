# Content List 归属迁移验证

Content List V1/V2 的实现及专用公共逻辑已迁入
`mineru.render._internal.content_list`。公共函数签名、宿主配置读取方式和列表输出
保持不变。`ContentType`、`ContentTypeV2` 由 `mineru.types` 定义。

MinerU 独立定义九种目标的 `RenderFormat` 与专用选项。通用选项、`RenderMode`、
素材接口和文档类型继续共享 DocVortex；DocVortex 自身只提供七种目标。
迁入代码通过 `docvortex.render.fragments` 使用公共能力，没有导入引擎私有模块。

迁移前冻结全类型夹具、中文论文3和中文论文4的 V1/V2 输出，共六份规范化 JSON。
迁移后通过 MinerU 公共函数输出，与冻结文件逐字节一致，源文档树保持不变。
原 Content List 单测继续保留，并迁入两项公式分隔符测试、增加独立契约归属检查。

本地完整回归：**3957 passed、4 skipped、4 deselected**。四项历史排除不变：

- `test_doclib_app_startup.py::test_basic_extra_includes_preflight_runtime_dependencies`
- `test_doclib_app_startup.py::test_standard_is_the_highest_model_runtime_extra`
- `test_legacy_schema_adapter.py::test_doclib_compaction_rejects_unknown_legacy_schema`
- `test_mfr_latex_utils.py::test_pp_formulanet_fix_latex_uses_shared_mathring_repair`

DocVortex 本地完整回归 416 项通过，独立 Python 3.14 wheel 验证 45 项通过；
[三平台 CI](https://github.com/myhloli/DocVortex/actions/runs/34047782292) 已通过。
既有少线表金标差异仍为单独诊断项，不计入本次修复。

仓库显示名为 `myhloli/DocVortex`，包、CLI 和本地目录仍为小写 `docvortex`。
只清理 DocVortex 指定版权头，MinerU 继续遵守原有文件头约定；其他许可证与归属保留。
更新 0.1.0 wheel/sdist 与现有发行草稿，本次不发布到 PyPI。
