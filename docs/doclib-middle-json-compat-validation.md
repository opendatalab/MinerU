# Doclib 历史 Middle JSON 兼容验证

日期：2026-09-07。实现基于本地 `48fdd2b66` 工作区，仅在 Doclib 持久化读取边界恢复兼容。
使用约定见 [Doclib 历史兼容](next/middle-json/doclib-compatibility.md)。

## 实现与边界

- 新增 `read_cached_middle_json()`，在内存中读取当前格式、3.4.5/旧 1.0 包装及旧 MinerU Schema 2.0。
- 页面适配器与 `dc0944984` 中的原文件逐字节一致；新构造器及元数据处理放在 Doclib 读取层。
- 旧 Schema 2.0 只调整外层，测试明确禁止调用共享语义后处理或 LLM。
- 历史生产版本缺失时记为 unknown，运行参数不可靠时不补造产品记录。
- 缓存命中、覆盖、默认读取档位、FTS 重建和压缩使用同一读取入口。
- 普通读取不改写历史 JSON 或数据库记录；兼容且元数据一致的批次可在正常压缩时写成新格式。
- 通用 ParseResult、DocVortex codec、HTTP/ZIP 和 Gradio 没有增加旧格式读取能力。
- 本次没有调整包版本、协议版本、解析算法或兼容有效期，也未新增兼容开关。

## 验证结果

使用本地 Magic-PDF `.venv1`：

| 检查 | 结果 |
| --- | --- |
| 适配器、Doclib、通用读取、HTTP/ZIP、Gradio 专项 | 560 passed |
| MinerU 完整测试 | 1870 passed、3 skipped、3 个既有失败 |
| 本次修改的 Python 文件 Ruff | 通过 |
| 本次修改的补丁空白检查 | 通过 |

专项覆盖原 3.4.5 文字、公式、链接、图表、列表、目录和坐标转换；旧 2.0 页面保持；
版本缺失、元数据冲突、未知扩展保留、输入不变和转换幂等性；以及显式错误 schema 不兜底。
真实 SQLite/FTS5 用例覆盖两种历史协议族的缓存命中、内容读取、范围覆盖、索引重建、
默认档位、混合协议压缩、重复页取新及冲突/失败时保留源数据。

完整测试仍存在三项此前已确认的失败，均不涉及本次兼容代码：

- `test_basic_extra_includes_preflight_runtime_dependencies`：既有 basic extra 缺失。
- `test_standard_is_the_highest_model_runtime_extra`：既有 test extra 不含 mineru[standard]。
- `test_pp_formulanet_fix_latex_uses_shared_mathring_repair`：既有 UniMERNetDecode 缺少 fix_latex。

执行期间工作区还存在另外进行的 PDF 渲染、依赖、版本和测试调整，本次未修改这些文件。
初轮完整测试额外出现 PDF 补图用例失败；后续并行更新后，最终完整测试已不再出现该失败。
以上完整测试数字对应最终运行时的整个共享工作区，不将其他修改计入本次兼容实现。

## 复跑与证据

```bash
python -m pytest -q tests/unittest/test_doclib_middle_json.py tests/unittest/test_doclib_legacy_schema_adapter.py tests/unittest/test_doclib_legacy_page_ranges.py tests/unittest/test_doclib_cache_semantics.py tests/unittest/test_legacy_schema_adapter.py tests/unittest/test_parse_result_contract.py tests/unittest/test_parser_api_contract.py tests/unittest/test_kit_gradio.py
python -m pytest -q
```

本地日志保存在 `/tmp/doclib-middle-compat/verified-focused.log` 和
`/tmp/doclib-middle-compat/full-final.log`。先前 JSON 2.0 直接迁移的验证报告不改写；
此记录描述后续收敛为 Doclib 专属兼容边界的行为。
