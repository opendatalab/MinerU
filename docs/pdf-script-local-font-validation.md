# PDF 同字体局部正文复核验证

DocVortex 的共享上下标分类现在会以同字体连续英数字串中的可信正文样本复核
弱候选。MinerU 的 Hybrid 原生文字回填继续调用同一实现，没有增加产品参数或分支。

新增最小真实字符 fixture 验证中文论文4的 `１２ｇｐｍ` 全部正文、`［８］` 整体上标，
并通过生产 `chars_to_content()` 路径验证输出。原始字符索引、几何和字体策略不变。

本地文本/样式组 **206 项通过**；完整回归 **3958 passed、4 skipped、4 deselected**。
四项排除仍为既有基线：

- `test_doclib_app_startup.py::test_basic_extra_includes_preflight_runtime_dependencies`
- `test_doclib_app_startup.py::test_standard_is_the_highest_model_runtime_extra`
- `test_legacy_schema_adapter.py::test_doclib_compaction_rejects_unknown_legacy_schema`
- `test_mfr_latex_utils.py::test_pp_formulanet_fix_latex_uses_shared_mathring_repair`

引擎 [三平台 CI](https://github.com/myhloli/DocVortex/actions/runs/34052061906) 通过，
源代码为 `be494c615dfa597457e950f43cc1dcc6b1cfc970`。31 份语料仅目标片段四个字符的
上下标样式改变；真实引用、数学角标、表格角标和英文作者名保持原有正确结果。
原有少线表诊断单列，不改金标或放宽阈值。

升级后已保存的 Doclib 结果不自动重写，重启服务并用 `--force` 重新解析以获取修复。
0.1.0 安装包及发行草稿更新，不发布到 PyPI。
