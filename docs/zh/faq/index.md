# 常见问题解答

如果未能列出您的问题，您也可以使用[DeepWiki](https://deepwiki.com/opendatalab/MinerU)与AI助手交流，这可以解决大部分常见问题。

如果您仍然无法解决问题，您可通过[Discord](https://discord.gg/Tdedn9GTXq)或[WeChat](https://mineru.net/community-portal/?aliasId=3c430f94)加入社区，与其他用户和开发者交流。

??? question "Windows 推理很慢或引擎安装失败怎么办？"

    ### Windows CUDA 加速 {#windows-cuda-acceleration}

    4.1 基础包在 Windows x64 可自动选择 ONNX DirectML 或 CPU，VLM 使用 llama.cpp。可用 `mineru-kit hardware info --json` 检查环境与已有校准证据。需要 Torch 小模型和 LMDeploy 时，在原虚拟环境中安装 `"mineru[full]>=4.0,<5"`，并核对 Torch 的 CUDA 支持、显卡驱动及引擎 wheel 的 Python 范围。

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
