# 常见问题解答 {#_1}

## 为什么只得到 PDF 的前 10 页？ {#first-ten-pages}

`mineru parse` 默认只请求前 10 页。要导出整本，运行 `mineru parse document.pdf --pages all -o document.md`。只做一次文件转换时，用 `mineru-kit parse document.pdf -o document.md`，默认处理全部页。若终端只显示部分文字，按输出末尾的续读命令继续，或用 `-o` 保存所请求范围的完整 Markdown。[文档库与页范围](../usage/quick_usage.md#pdf)

## 模型下载失败，或需要离线运行怎么办？ {#offline-models}

在部署机用 `mineru-kit models show` 查看实际后端和引擎。在联网机器上按目标组合下载并验证模型，例如执行 `mineru-kit models verify --tier standard`（仅部署 Basic 时改用 `basic`）。Hugging Face 不可达时可选择 ModelScope。离线机器使用准备好的模型目录、相同的后端和引擎，并将 `model.source` 设为 `local`。[模型下载与离线步骤](../usage/model_source.md#offline-models)

## 等待超时后，如何取得原任务的结果？ {#wait-timeout}

| 入口 | 接下来的操作 |
| --- | --- |
| `mineru parse` | 客户端报告 `parse_wait_timeout` 时，重新执行相同命令继续等待，可用 `--wait 600` 延长这次等待。[CLI 等待与导出](../usage/cli_tools.md) |
| Tasks | 保存创建回执中的 `task_id`，继续查询 `GET /v1/tasks/{task_id}`，按已完成文件的 `result_url` 下载。[Tasks 客户端示例](../usage/tasks.md#client-examples) |
| file-parse | 收到 `202` 时读取 `status_url` 或 `Location`。如果断连前已指定 `request_id`，可以用它查询包装服务，已完成时按 `result_url` 下载。[等待与恢复](../usage/file_parse.md#recovery) |

## 运行环境与升级 {#runtime-and-upgrades}

??? question "Windows 推理很慢或引擎安装失败怎么办？"

    ### Windows CUDA 加速 {#windows-cuda-acceleration}

    基础包在 Windows x64 可自动选择 ONNX DirectML 或 CPU，VLM 使用 llama.cpp。可用 `mineru-kit hardware info --json` 检查环境与已有校准证据。需要 Torch 小模型和 LMDeploy 时，在原虚拟环境中安装 `"mineru[full]>=4.0,<5"`，并核对 Torch 的 CUDA 支持、显卡驱动及引擎 wheel 的 Python 范围。

    当前版本使用 `lmdeploy>=0.17.0,<0.19` 和 Transformers 5；不要套用旧版 LMDeploy 0.11.x 的 wheel 或跳过依赖检查。详见[扩展模块](../quick_start/extension_modules.md)和[档位说明](../usage/tiers.md)。

??? question "升级后旧命令、API 或配置不再工作怎么办？"

    4.x WebUI 使用 `mineru-kit webui` / `mineru-webui`，解析 API 使用 `/v1/*`，配置使用 `config.yaml`。旧 `/file_parse`、`/tasks` 和 `model.stack` 不能直接沿用，见 [3.x 迁移指南](../reference/migration_4.md)。4.1 将独立 uploads/jobs 替换为 Tasks，见 [4.1 迁移说明](../reference/migration_4_1.md)和独立 [file-parse 接口](../usage/file_parse.md)。

??? question "在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`"

    ### 在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory` {#wsl2-ubuntu2204-libgl}

    WSL2的Ubuntu22.04中缺少`libgl`库，可通过以下命令安装`libgl`库解决：
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    参考：[#388](https://github.com/opendatalab/MinerU/issues/388)

还有问题时，可查阅 [DeepWiki](https://deepwiki.com/opendatalab/MinerU)，或通过 [Discord](https://discord.gg/Tdedn9GTXq) 和 [WeChat](https://mineru.net/community-portal/?aliasId=3c430f94) 与社区交流。
