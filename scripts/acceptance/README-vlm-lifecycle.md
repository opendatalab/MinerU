# VLM 生命周期联合验收

先安装同一候选批次的 MinerU 与 mineru-vl-utils wheel，保存版本、源码位置与 SHA256。
真实设备结果与异常源码契约检查分开记录，不以模拟通过代替 GPU 验收。

## Linux / WSL2 vLLM

为 `0.19.1` 与 `0.30.0` 分别使用独立的 uv Python 3.12 环境。环境可复用只读模型缓存，不复用另一版本的包目录。

```bash
uv venv --python 3.12 .venv-vllm019
uv pip install --python .venv-vllm019/bin/python 'vllm==0.19.1'
# 在该环境安装本批候选 wheels 及 MinerU 的解析依赖，再运行：
.venv-vllm019/bin/python scripts/acceptance/vllm_lifecycle.py \
  --sample demo/pdfs/demo2.pdf --models-dir /path/to/models --evidence output/vllm019
```

第二个环境改为 `vllm==0.30.0`，使用独立 evidence 目录。脚本需要同目录的 `lmdeploy_lifecycle.py`。
它复用正式 WorkerPool、Tasks 和结果下载流程，验证真实切片、Standard/Advanced 产物、循环阻塞时超时恢复、
空闲真实 shutdown、运行中信号注入、任务清理及托管进程替换。

运行中注入只在真实引擎已经交出 token、请求尚未退出时触发；其结论是宿主存活时的故障处理，不是原始崩溃根因。
`--active-stop close` 是额外关闭诊断；整进程退出与引擎独立死亡分别报告，不能放宽断言将前者算作后者通过。
验收路由只存在于回环地址上的独立测试 worker，不加入生产服务。

## Windows LMDeploy

沿用 [LMDeploy 验收说明](README-lmdeploy.md)，分别验证 TurboMind 0.17/0.18；不把 Windows PyTorch 列为通过条件。

## macOS MLX

```bash
python scripts/acceptance/mlx_lifecycle.py \
  --sample demo/pdfs/demo2.pdf --evidence output/mlx-lifecycle
```

保存真实解析产物，验证同步生成开始后的取消等待、未知错误向上传播、同实例后续真实推理成功及缓存释放。
未知错误是显式测试注入，不是对 MLX 原生崩溃的宣称。

## 检查报告

同时检查 `report.json`、worker 完整日志与结果 ZIP/Markdown。出现未处理异常、重复异常日志、错误档位或空产物均需调查。
代码测试、上游版本契约、三平台真机、依赖发布及发布包安装验证分别记录状态。
