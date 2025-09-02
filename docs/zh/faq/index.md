# 常见问题解答

如果未能列出您的问题，您也可以使用[DeepWiki](https://deepwiki.com/opendatalab/MinerU)与AI助手交流，这可以解决大部分常见问题。

如果您仍然无法解决问题，您可通过[Discord](https://discord.gg/Tdedn9GTXq)或[WeChat](https://mineru.net/community-portal/?aliasId=3c430f94)加入社区，与其他用户和开发者交流。

??? question "在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`"

    WSL2的Ubuntu22.04中缺少`libgl`库，可通过以下命令安装`libgl`库解决：
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    参考：[#388](https://github.com/opendatalab/MinerU/issues/388)


??? question "在 CentOS 7 或 Ubuntu 18 系统安装MinerU时报错`ERROR: Failed building wheel for simsimd`"

    新版本albumentations(1.4.21)引入了依赖simsimd,由于simsimd在linux的预编译包要求glibc的版本大于等于2.28，导致部分2019年之前发布的Linux发行版无法正常安装，可通过如下命令安装:
    ```
    conda create -n mineru python=3.11 -y
    conda activate mineru
    pip install -U "mineru[pipeline_old_linux]"
    ```
    
    参考：[#1004](https://github.com/opendatalab/MinerU/issues/1004)

??? question "在 Linux 系统安装并使用时，解析结果缺失部份文字信息。"

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