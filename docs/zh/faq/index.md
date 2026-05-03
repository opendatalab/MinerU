# 常见问题解答

如果未能列出您的问题，您也可以使用[DeepWiki](https://deepwiki.com/opendatalab/MinerU)与AI助手交流，这可以解决大部分常见问题。

如果您仍然无法解决问题，您可通过[Discord](https://discord.gg/Tdedn9GTXq)或[WeChat](https://mineru.net/community-portal/?aliasId=3c430f94)加入社区，与其他用户和开发者交流。

??? question "Windows 直接安装后推理速度很慢怎么办？"

    ### Windows 直接安装后推理速度很慢怎么办？ {#windows-cuda-acceleration}

    Windows 直接安装后如果推理速度很慢，通常是 CUDA 加速相关依赖未正确安装。请根据显卡架构选择对应方案：

    - Volta / Turing / Ampere / Ada Lovelace 架构显卡，例如 V100、20 系、T4、30 系、40 系：直接安装支持 CUDA 的 `torch` 和 `torchvision` 即可。请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择适合您 CUDA 版本的 Windows 安装命令。
    - Blackwell 架构显卡，例如 RTX 50xx 系列：安装 `lmdeploy 0.11.1 + cu128` 的 Windows wheel。请将 `PYTHON_VERSION` 设置为当前 Python 版本，例如 Python 3.10 / 3.11 / 3.12 / 3.13 分别填写 `310` / `311` / `312` / `313`。

    ```powershell
    $env:LMDEPLOY_VERSION = "0.11.1"
    $env:PYTHON_VERSION = "312"

    $wheel = "https://github.com/InternLM/lmdeploy/releases/download/v$($env:LMDEPLOY_VERSION)/lmdeploy-$($env:LMDEPLOY_VERSION)+cu128-cp$($env:PYTHON_VERSION)-cp$($env:PYTHON_VERSION)-win_amd64.whl"
    pip install $wheel --extra-index-url https://download.pytorch.org/whl/cu128
    ```

    如果 Blackwell 架构显卡环境中已经安装过 cu128 版本的 `torch`，则在定义好 `$wheel` 后只需执行以下命令，避免重新下载低版本 `torch`：

    ```powershell
    pip install $wheel --no-dependencies
    ```


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
