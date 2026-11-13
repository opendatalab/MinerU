# LMDeploy Windows 验收

在已能运行 MinerU Standard 的 Windows CUDA 虚拟环境中执行，使用本次候选 MinerU 与 `mineru-vl-utils 2.0.9` wheel。Windows 的受支持 LMDeploy 后端为 TurboMind；PyTorch 后端不纳入本轮验收，也不作为发布阻塞项。候选包尚未发布；本脚本不会升级 Torch、LMDeploy 或其他推理依赖。

MinerU 候选包沿用当前 `dev` 的 `4.2.0` 版本号，不代表正式版本回退或新版本发布。请按包内 `MANIFEST.json` 和 `SHA256SUMS` 核对本次构建来源，并在用于验收的虚拟环境中安装。

```powershell
.\run-lmdeploy.ps1 -Sample C:\samples\formulas.pdf -InstallCandidates
```

默认后端即 `turbomind`，也可显式传入 `-Backend turbomind`；重复运行时省略 `-InstallCandidates`。如需指定模型根目录，使用 `-ModelsDir C:\MinerU\models`；如有原始故障完整日志，传入 `-IncidentLog C:\logs\original.log`。环境应已准备 Standard 所需模型；服务启动会按正常模型加载流程检查缓存。

分别使用支持的 LMDeploy 0.17／0.18 环境执行。不要为了凑矩阵在同一个正常工作环境中反复自动更换底层依赖。选择包含图像或公式、至少数页的 PDF；若样例太快无法观察到实际活跃推理序列，脚本会报出未完成该项验收，需要换更长样例重跑。API 任务进入 `running` 仍可能只是在做版面分析，不能据此证明 LMDeploy 正在推理。

脚本执行：

1. 启动隔离的回环 API worker，记录真实版本、GPU、实际后端配置和完整日志。
2. 对合成切片执行真实 VLM 推理，并用提供的 PDF 分别完成 Standard／Advanced 转换。
3. 暂缓 Pipeline 循环，验证探测超时返回 `engine_unavailable`、拒绝新任务，随后恢复而不标记永久死亡。
4. 停止空闲后端，验证宿主仍能报告 `engine_dead`；使用正式 `WorkerPool`、`ManagedLocalWorker` 和控制通道替换进程，核对 PID、generation 与恢复后的转换。
5. 确认 LMDeploy 后端有活跃推理序列后，在验收进程内注入明确的健康故障信号，让底层计算自然结束。检查宿主仍能返回正在处理和排队任务的 `engine_dead`、无成功结果链接，并再次重建、完成 Advanced 转换。

第 5 步的 GPU 推理是真实的，健康故障信号是显式注入的；它验证宿主的失效锁存、任务清理和恢复，不宣称复现了自然后端崩溃。不能在活跃的 TurboMind 原生推理期间直接调用 `close()` 来替代该信号：本轮 Windows 0.18 试验中，这种并发资源释放导致了原生访问冲突和整个 worker 退出。进程退出后的管理者恢复应独立记录；旧 worker 已退出、无法读取任务终态时，不能将其计作宿主存活场景通过。

确需复查该原生崩溃时，可在独立验收目录直接调用 Python 脚本并显式传入 `--active-stop close`。该模式保留实际进程退出码、工作目录清理、PID/generation 变化和恢复转换记录；若没有观察到旧任务终态，仍返回 `passed=false` 和非零退出码，即使进程恢复成功也不会算作完整生命周期验收通过。PowerShell 包装入口继续使用默认的 `signal` 模式。

故障注入路由只存在于该脚本启动的临时 worker，不加入正式产品。脚本测试真实 WorkerPool 管理逻辑，不宣称覆盖完整 Router HTTP 代理链；代理链另有本地回归测试。

每次结果写入当前目录下的 `lmdeploy-evidence-*`：`report.json`、各 worker 完整日志和转换 ZIP。`passed=true` 表示本轮自动断言通过，还需检查日志中的未处理异常，并人工核对 ZIP 中的 Markdown/图片等产物。环境存在继承的 `site-packages` 时，同时核对模块实际路径与版本，不能把枚举包元数据时同名覆盖后的值当作实际加载版本。脚本保留在本机的原始事故日志和产物可能包含文档内容，按实际需要提供验收材料。

本轮验收目标是引擎崩溃后的失效锁存、拒绝接单、任务清理和宿主管理恢复，不要求解决导致引擎崩溃的所有后端缺陷。原始事故的第一次异常原因应由原日志和实际复现单独判断；根因尚未确认不阻塞已经独立验证的故障处理修复，也不能因隔离与恢复通过就宣称原始崩溃原因已修复。
