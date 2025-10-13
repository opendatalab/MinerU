# 常见问题解答
## 本地部署的常见问题
- 如果您在本地部署过程中遇到问题，我们**强烈推荐**您使用[DeepWiki](https://deepwiki.com/opendatalab/MinerU)与AI助手交流，这可以解决大部分常见问题
- 与此同时，您也可以前往 [GitHub Issues](https://github.com/opendatalab/MinerU/issues) 中进行搜索
- 如果您的问题仍然无法解决，您可通过[WeChat](https://mineru.net/community-portal/?aliasId=3c430f94)或[Discord](https://discord.gg/Tdedn9GTXq)加入社区，与其他用户和开发者交流，我们会不定时的回复部分问题

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
    

## 网页端/桌面客户端的常见问题

- 如果您在网页端/桌面客户端遇到使用问题，或者文件解析的效果出闲问题，您可通过[微信](https://mineru.net/community-portal/?aliasId=73ea3ef3)加入社区，与其他用户和开发者交流，我们会不定时的回复部分问题，其他问题欢迎大家互相帮助解答。

??? question "桌面客户端和网页端有什么区别？"

    客户端支持截图上传，可以提供更稳定和批量的解析能力；网页版无需安装，即开即用。

??? question "解析一篇文档需要多久？"

    通常需要几秒到几分钟不等，具体时间取决于文件的大小、页数以及当前系统解析的文档总量。建议将批量的大文件选择在非高峰时段进行解析，以避免排队。

??? question "桌面客户端解析后的文件存在哪里？"

    本地存储： 通过桌面客户端解析完成的结果文件会保存在您电脑上的本地固定路径中。您可以在客户端的【设置】中查看并修改此存储路径。
    云端存储：用于解析的文件将会在云端存储30天，过期后将会自动删除，MinerU桌面客户端的文件列表仅会展示历史解析记录。

??? question "MinerU的翻译功能应该如何启用和使用？"

    MinerU 提供两种翻译使用模式：
        **划词翻译 & 模块翻译：** 此功能已内置，无需配置即可使用（限时免费）。非常适合在阅读解析结果时，对特定段落或元素进行即时翻译对照。
        **全文翻译 (需配置)：**
        如需翻译整篇文档，您需要先在【设置】→【翻译服务】中，配置第三方翻译服务商（如 DeepL、Google、微软等）的 API Key。配置完成后，在文档解析结果页点击【全文翻译】即可使用。

??? question "为什么全文翻译有时失败或很慢？"

    这取决于您配置的翻译 API 的并发限制和稳定性，建议选择商用级服务商。

??? question "解析化学论文有什么特殊要求？"

    必须在上传文件前开启【化学论文】选项，否则可能无法正确提取分子式和反应式。

## 在线 API 的常见问题

??? question "如何调用在线 API？"

    您需要先申请在线API 权限，待审批通过后，在账户页面创建 Token，即可参照我们的技术文档调用 v4 版 API 接口。

??? question "在线 API Token 过期了怎么办？"
    
    Token 默认有效期为 14 天。过期前，您需要在账户页面创建一个新的 Token，并及时替换到您的代码中。