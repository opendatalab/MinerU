# Flash PDF 等价性基准

在仓库根目录执行，默认覆盖版本化的 19 份布局语料和 12 份原生表格文档：

```bash
.venv1/bin/python tests/benchmarks/flash_pdf.py --output output/flash_pdf_refactor/baseline
.venv1/bin/python tests/benchmarks/flash_pdf.py --output output/flash_pdf_refactor/candidate --baseline output/flash_pdf_refactor/baseline
```

每份文档在独立子进程中预热一次、计时五次，报告中位耗时与进程峰值 RSS。耗时包含 PDFDocument 打开、原生分析和关闭，不包含后处理、输出序列化、OCR 或渲染。RSS 在输出后处理和额外剖析之前采集；它包含解释器和已导入模块，不是文档的增量内存。

每份文档保存完整 `model_list` 和通过统一后处理入口生成的 `middle_json`。LLM 增强显式使用默认关闭配置，版本字符串固定，比较不移除任何输出字段。`report.json` 包含源码版本、依赖版本、文件指纹与历史清单差异；历史清单不会被改写。`comparison.json` 分别报告完整输出是否相等、耗时比例和 RSS 比例，输出不等价时命令失败。

使用 `--runs 0` 跳过预热和重复计时，只进行功能比较。`--path` 可重复指定固定子集；基线和候选必须使用同一份语料集合。`--profile` 额外生成逐函数调用次数、自身耗时与累计耗时，可定位提取、几何校准、页面准备、页面组装和样式物化阶段；剖析耗时不参与性能比较。性能测量期间避免同时执行测试、构建或其它 CPU 密集任务。

已有报告目录不能覆盖。先在原始版本建立基线，然后在候选版本使用新的输出目录；不可通过更新基线掩盖输出变化。超过 5% 的持续耗时或 RSS 回退应独立复测并定位。
