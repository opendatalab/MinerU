# PDF 内存回归验证

窗口生命周期修复始终生效。`MINERU_MALLOC_TRIM` 是独立的可选 CPU 堆回收开关，默认关闭。只有在 Linux/glibc 上完成真实测量，才能判断它是否改善当前负载；macOS 上的通过结果不证明 Linux RSS 收益或 OOM 已解决。

## 对比方法

在同一主机、Python 环境、DocVortex 版本、模型配置和并发条件下比较修复前、修复后关闭 trim、修复后开启 trim。先创建修复前提交的独立 worktree。不要切换或覆盖正在开发的工作区。

激活项目 Python 环境，安装仅供诊断脚本使用的 `psutil`（`uv pip install psutil`）。从修复后的工作区执行下面三组实验；`--repo` 只切换子进程的 MinerU 源码，第三方依赖共用当前 Python 环境。

```bash
python scripts/benchmark_pdf_memory.py --repo /tmp/mineru-memory-baseline --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 0 --output /tmp/mineru-memory-before
python scripts/benchmark_pdf_memory.py --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 0 --output /tmp/mineru-memory-after-off
python scripts/benchmark_pdf_memory.py --pdf demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf --tier basic --mode ocr --window-size 2 --trim 1 --output /tmp/mineru-memory-after-on
```

每组使用独立解析进程，默认预热 3 轮、测量 20 轮。每轮处理一个文件，多个输入依次循环；上一轮的 ParseResult 会在下一轮前释放。输出目录必须不存在，避免覆盖历史数据。

- 重复同一文档：只传一个 `--pdf` 输入。
- 交替不同文档：传两个或更多输入。
- 多窗口文档：使用页数大于 `--window-size` 的文件。窗口大小只影响本次实验，不改变产品默认配置。
- 推理路径分别运行 `basic/standard/advanced` 与 `txt/ocr`，以及 `flash --mode ocr`；需要对应模型和 VLM 配置。
- 没有神经模型时可以先用 `flash --mode txt` 验证脚本、文档级清理和按需补图，但不能据此验证推理窗口的大数组生命周期或推理性能。

## 输出与判断

- `rounds.jsonl`：每轮输入、页数、解析耗时和起止时刻，区分预热与测量。
- `samples.jsonl`：外部每 0.2 秒采样的主进程及全部子进程 RSS/USS，保留 PID。USS 不可读取时记录 `null`；可用 `--interval` 调整频率。
- `parse.log`：生产解析日志，包含窗口页码以及渲染 worker PID，可用于关联采样。
- `summary.json`：测量阶段耗时中位数、各 PID 峰值和同一采样时刻的进程合计峰值。RSS 合计可能重复计算共享页；采样峰值也可能遗漏短暂尖峰，不能视为精确最高水位。

主进程和 worker 分别检查，避免把子进程滞留误认为主进程 trim 无效。观察预热后各轮结束附近的 RSS/USS 是否持续增长、是否稳定到平台，以及峰值与耗时变化。基线与修复后关闭 trim 的耗时中位数若回归超过 5%，在排除并行任务、模型冷启动和采样噪声后调查；启用 trim 的成本单独记录。建议重复完整三组实验，避免凭单次结果下结论。

`malloc_trim` 不回收存活结果、模型缓存、其他分配器的池或其他进程的内存。异常栈仍引用的对象也可能继续存活。本轮不修改 DocVortex worker 的回收策略。

## 结果一致性

使用单独输出目录为三组命令追加 `--warmup 0 --rounds 2 --fingerprints`。比较相同输入的 `middle_sha256` 和 `model_sha256`，摘要包含完整素材字段。`--fingerprints` 额外执行序列化，因此不要用这一组数据评估内存和性能。

只有完成真实 Linux 推理负载测量后，才报告对应场景的 RSS 改善。缺少环境、模型或无法获取 USS 时，应在验收记录中明确说明，不将默认开关改为开启。
