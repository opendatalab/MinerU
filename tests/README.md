# MinerU 测试职责

原生解析、共享 Schema、确定性后处理、渲染与导出的完整算法回归已迁入 DocVortex。配套迁移清单和验证结果位于 DocVortex 仓库的 `docs/validation/test-migration.md`、`docs/validation/test-migration.json`。

本仓库保留 OCR/Hybrid/VLM 编排、Flash 模式路由、CLI/API/Doclib、产品元数据与协议适配、渲染参数传递、旧宿主路径移除、LLM 增强及 Content List V1/V2 测试。需要小型 Office 输入的集成测试继续使用本地构造器，不从 DocVortex 的测试目录导入代码。

## 本地运行

```sh
.venv1/bin/python -m pytest tests/unittest -o addopts= -q -m 'not remote and not full_stack'
```

迁移前后均存在的三个失败为两项 Doclib extras 元数据断言及一项 MFR LaTeX 修复断言；本次没有新增排除项或改变对应断言。

## Hybrid 输入与原始语料

`tests/unittest/pdfs` 已移除，MinerU 的真实 PDF 集成输入继续使用 `demo/pdfs/demo1.pdf`、`demo2.pdf`。完整 PDF/Office/EMF+ 语料、表格评测器和合成 PDF 生成器由 DocVortex 维护。

- `fixtures/hybrid_native_script_inputs.json`：原有 33 组字符与几何快照，内容及历史来源信息保持不变。
- `fixtures/hybrid_pdf_script_inputs.json`：新增四个 CJK 探针和一个旋转页场景，包含原始字符、字体、索引、tight/origin 几何及所需页面属性。
- `fixtures/pdf_mixed_font_script_line.json`：继续覆盖 Hybrid 的真实混合字体回填行为。
- `fixtures/rtf/semantic.rtf`：保留产品接入测试仍使用的小型输入。

普通测试只读取本仓库快照。显式再生新增快照时，由调用者提供 DocVortex 源码目录：

```sh
.venv1/bin/python tests/fixtures/capture_hybrid_native_script_inputs.py \
  --fixture tests/fixtures/hybrid_pdf_script_inputs.json \
  --source-root /path/to/DocVortex --check
```

不带 `--check` 时会写回指定快照。旧夹具仍使用原默认 `--fixture` 和 DocVortex 的 `demo/pdfs` 作为 source root。再生前先核对源文件哈希；不要为消除回归差异覆盖已有真值。
