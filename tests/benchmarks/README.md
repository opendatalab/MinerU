# Standard PDF CPU gap 基准

在仓库根目录运行，使用已配置的 VLM 后端。DocVortex 需要包含本次优化的 0.4.15 源码或发行包。

```bash
PYTHONPATH=. .venv1/bin/python tests/benchmarks/pdf_cpu_gaps.py \
  --pdf /Users/myhloli/pdf/caibao1.pdf --page-range 1-20 \
  --out output/standard-cpu/live --runs 6
```

首次运行单独记录，后五次报告预热后中位数。`summary.json` 包含模型初始化时间、CPU 时间、分析墙钟、
包含确定性 MiddleJson 后处理的 `parse_wall`、两个 gap 与进程峰值 RSS。模型初始化不计入逐次解析墙钟。
`gap1` 从 Layout 调用完成到 VLM 调用入口；VLM 客户端内部预处理与网络传输仍属于 VLM 调用。
`gap2` 从 OCR-det 返回到正文 OCR-rec 调用入口；不存在对应阶段时记为 null，多窗口时汇总各窗口 gap。
计时不包含 HTML/Content List 导出；可选 LLM 增强未启用。

## 固定模型输出回放

在改动前的源码上录制一次，再用完全相同的 PDF 和页范围在改动后回放：

```bash
PYTHONPATH=. .venv1/bin/python tests/benchmarks/pdf_cpu_gaps.py \
  --pdf /path/caibao1.pdf --page-range 1-20 --record \
  --tape /tmp/caibao20.pkl.gz --out output/standard-cpu/before

PYTHONPATH=. .venv1/bin/python tests/benchmarks/pdf_cpu_gaps.py \
  --pdf /path/caibao1.pdf --page-range 1-20 \
  --tape /tmp/caibao20.pkl.gz --out output/standard-cpu/after --runs 6 \
  --compare output/standard-cpu/before/output.json
```

录制文件包含该 PDF 的渲染图和模型输出，是仅用于本机诊断的可信 pickle，不是通用输入格式。
回放不初始化模型；真实 CPU 分析、PDF 提取和后处理照常执行。
逐值比较 ModelJson、MiddleJson、Content List V1/V2、素材和 HTML，并检查录制调用是否完整消费。
**回放墙钟不能当作真实模型的端到端耗时。** 基线和候选应在独立进程中依次运行，计时期间不要并行跑其他基准。

`--mode ocr` 可验证 OCR 路由；该路由可能没有正文 OCR-rec，因此 gap2 为 null。
