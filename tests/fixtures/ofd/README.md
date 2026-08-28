# OFD fixtures

自动化单元测试通过 `tests/unittest/_ofd_test_utils.py` 在内存中构造最小 OFD 包，避免把来源或隐私不明确的 issue
附件提交到仓库。

本地真实语料测试会在以下目录存在时自动运行：

```text
tmp/ofd_samples/ofdrw_issues_20260828/ofd
```

其中以下文件已在 2026-08-28 对照 OFDRW `7459e35082170061efa6b399a6518dbc219f08ac` 的官方测试资源，
SHA-256 完全一致：

| 本地 SHA-256 前缀 | OFDRW 源路径 |
|---|---|
| `a7b6f9eceb` | `ofdrw-converter/src/test/resources/helloworld.ofd` |
| `a2b9080ae3` | `ofdrw-converter/src/test/resources/999.ofd` |
| `dfed483fa0` | `ofdrw-layout/src/test/resources/AddWatermarkAnnot.ofd` |
| `e7cac8d149` | `ofdrw-layout/src/test/resources/no_page_container.ofd` |
| `bff244b113` | `ofdrw-layout/src/test/resources/keyword2.ofd` |

OFDRW 使用 Apache License 2.0。其余 issue 附件仅作为本地兼容语料，不应在未确认授权和隐私边界前提交。
