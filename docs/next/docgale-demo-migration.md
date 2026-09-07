# DocGale 样例与回归测试归属

完整 PDF、Office 样例与对应原生解析、几何、语义、表格、渲染回归现由
[DocGale](https://github.com/myhloli/docgale/tree/main/demo) 维护。MinerU 只保留
`demo/pdfs/demo1.pdf` 和 `demo/pdfs/demo2.pdf`，用于宿主路由及 Hybrid 集成测试。

两个仓库的普通测试都不读取相邻仓库，也不在运行时下载样例。Office 路由测试
现场生成最小输入；复杂 Office 内容与导出真值在 DocGale 中验证。

`tests/fixtures/hybrid_native_script_inputs.json` 保存 Hybrid 回填使用的 33 组
最小字符输入，来自 7 份原始 PDF。每组保留字体、原始字符索引及 source_indices、
loose bbox（位于 chars.bbox）、tight bbox 和 origin，源文件 SHA256 和提取版本
记录在同一文件中。原有文本、几何归属和上下标断言不变。

只有需要再生或审查这些输入时，才需准备完整 DocGale 语料，并使用记录的 PDFium 版本：

```bash
uv run --no-project python tests/fixtures/capture_hybrid_native_script_inputs.py \
  --source-root /path/to/docgale/demo/pdfs --check
```

省略 `--check` 会显式更新 fixture。普通测试不需要运行此命令。

两边统一使用 `pydantic>=2.12.5,<3`，共享依赖约束保持一致。DocGale 支持
Python 3.10–3.14；MinerU 当前 Python 范围维持 3.10–3.13。此次没有新增
Pydantic 兼容实现。使用新 DocGale wheel 和 Pydantic 2.12.5 的 MinerU 单元测试
结果为 3972 passed、4 skipped、4 个已记录基线失败单独排除；迁走的原生测试
由 DocGale 独立执行。
