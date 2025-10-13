
> [!WARNING]
> **安装前必看——软硬件环境支持说明**
> 
> 为了确保项目的稳定性和可靠性，我们在开发过程中仅对特定的软硬件环境进行优化和测试。这样当用户在推荐的系统配置上部署和运行项目时，能够获得最佳的性能表现和最少的兼容性问题。
>
> 通过集中资源和精力于主线环境，我们团队能够更高效地解决潜在的BUG，及时开发新功能。
>
> 在非主线环境中，由于硬件、软件配置的多样性，以及第三方依赖项的兼容性问题，我们无法100%保证项目的完全可用性。因此，对于希望在非推荐环境中使用本项目的用户，我们建议**先仔细阅读文档以及FAQ**，大多数问题已经在FAQ中有对应的解决方案，除此之外我们鼓励社区反馈问题，以便我们能够逐步扩大支持范围。

<table>
    <tr>
        <td>解析后端</td>
        <td>pipeline</td>
        <td>vlm-transformers</td>
        <td>vlm-vllm</td>
    </tr>
    <tr>
        <td>操作系统</td>
        <td>Linux / Windows / macOS</td>
        <td>Linux / Windows</td>
        <td>Linux / Windows (via WSL2)</td>
    </tr>
    <tr>
        <td>CPU推理支持</td>
        <td>✅</td>
        <td colspan="2">❌</td>
    </tr>
    <tr>
        <td>GPU要求</td>
        <td>Turing及以后架构，6G显存以上或Apple Silicon</td>
        <td colspan="2">Turing及以后架构，8G显存以上</td>
    </tr>
    <tr>
        <td>内存要求</td>
        <td colspan="3">最低16G以上，推荐32G以上</td>
    </tr>
    <tr>
        <td>磁盘空间要求</td>
        <td colspan="3">20G以上，推荐使用SSD</td>
    </tr>
    <tr>
        <td>python版本</td>
        <td colspan="3">3.10-3.13</td>
    </tr>
</table>

### 一、安装 MinerU

MinerU开源项目支持以下多种安装方式，您可根据实际需求选择适合的安装形式。在安装开始前，**请务确保系统符合前文表格中所描述的要求。**



#### 1.1.使用pip或uv安装MinerU
```bash
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple

pip install uv -i https://mirrors.aliyun.com/pypi/simple

uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
```

#### 1.2.通过源码安装MinerU
```bash
git clone https://github.com/opendatalab/MinerU.git

cd MinerU

uv pip install -e .[core] -i https://mirrors.aliyun.com/pypi/simple
```

> [!TIP]
> `mineru[core]`包含除`vLLM`加速外的所有核心功能，兼容Windows / Linux / macOS系统，适合绝大多数用户。
> 如果您有使用`vLLM`加速VLM模型推理，或是在边缘设备安装轻量版client端等需求，可以参考文档[扩展模块安装指南](https://opendatalab.github.io/MinerU/zh/quick_start/extension_modules/)。

---
 
#### 1.3.使用docker部署Mineru
MinerU提供了便捷的docker部署方式，这有助于快速搭建环境并解决一些棘手的环境兼容问题。
您可以在文档中获取[Docker部署说明](https://opendatalab.github.io/MinerU/zh/quick_start/docker_deployment/)。

---

### 二、使用 MinerU

#### 2.1.简单的命令行调用
```bash
mineru -p <input_path> -o <output_path>
```

#### 2.2.复杂调用
您可以通过命令行、API、WebUI等多种方式使用MinerU进行PDF解析，具体使用方法请参考[使用指南](https://opendatalab.github.io/MinerU/zh/usage/)。