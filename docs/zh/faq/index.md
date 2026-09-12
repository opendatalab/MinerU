# 常见问题解答

如果未能列出您的问题，您也可以使用[DeepWiki](https://deepwiki.com/opendatalab/MinerU)与AI助手交流，这可以解决大部分常见问题。

如果您仍然无法解决问题，您可通过[Discord](https://discord.gg/Tdedn9GTXq)或[WeChat](https://mineru.net/community-portal/?aliasId=3c430f94)加入社区，与其他用户和开发者交流。

??? question "Windows 推理很慢或引擎安装失败怎么办？"

    ### Windows CUDA 加速 {#windows-cuda-acceleration}

    基础包默认可使用 ONNX / CPU 与 llama.cpp。需要 Torch 小模型和 LMDeploy 时，在原虚拟环境中安装 `"mineru[full]>=4.0,<5"`，并核对 Torch 的 CUDA 支持、显卡驱动及引擎 wheel 的 Python 范围。

    4.0 使用 `lmdeploy>=0.17.0,<0.18` 和 Transformers 5；不要套用旧版 LMDeploy 0.11.x 的 wheel 或跳过依赖检查。详见[扩展模块](../quick_start/extension_modules.md)和[档位说明](../usage/tiers.md)。

??? question "升级后旧命令、API 或配置不再工作怎么办？"

    4.0 WebUI 使用 `mineru-kit webui` / `mineru-webui`，解析 API 使用 `/v1/*`，配置使用 `config.yaml`。旧 `/file_parse`、`/tasks` 和 `model.stack` 不能直接沿用。见[迁移指南](../reference/migration_4.md)。

??? question "在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`"

    ### 在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory` {#wsl2-ubuntu2204-libgl}

    WSL2的Ubuntu22.04中缺少`libgl`库，可通过以下命令安装`libgl`库解决：
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    参考：[#388](https://github.com/opendatalab/MinerU/issues/388)

??? question "在 Linux 系统安装并使用时，解析结果缺失部份文字信息。"

    ### 在 Linux 系统安装并使用时，解析结果缺失部份文字信息。 {#linux-missing-text-cjk-fonts}

    MinerU在>=2.0的版本中使用`pypdfium2`代替`pymupdf`作为PDF页面的渲染引擎，以解决AGPLv3的许可证问题，在某些Linux发行版，由于缺少CJK字体，可能会在将PDF渲染成图片的过程中丢失部份文字。
    为了解决这个问题，您可以通过以下命令安装noto字体包，这在Ubuntu/debian系统中有效：
    ```bash
    sudo apt update
    sudo apt install fonts-noto-core
    sudo apt install fonts-noto-cjk
    fc-cache -fv
    ```
    也可以直接使用我们的[Docker部署](../quick_start/docker_deployment.md)方式构建镜像，镜像中默认包含以上字体包。
    
    参考：[#2915](https://github.com/opendatalab/MinerU/issues/2915)
