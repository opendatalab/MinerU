# 快速开始

如果遇到任何安装问题，请先查询 <a href="#faq">FAQ</a> </br>
如果遇到解析效果不及预期，参考 <a href="#known-issues">Known Issues</a></br>
有2种不同方式可以体验MinerU的效果：

- [在线体验](#在线体验)
- [本地部署](#本地部署)


> [!WARNING]
> **安装前必看——软硬件环境支持说明**
> 
> 为了确保项目的稳定性和可靠性，我们在开发过程中仅对特定的软硬件环境进行优化和测试。这样当用户在推荐的系统配置上部署和运行项目时，能够获得最佳的性能表现和最少的兼容性问题。
>
> 通过集中资源和精力于主线环境，我们团队能够更高效地解决潜在的BUG，及时开发新功能。
>
> 在非主线环境中，由于硬件、软件配置的多样性，以及第三方依赖项的兼容性问题，我们无法100%保证项目的完全可用性。因此，对于希望在非推荐环境中使用本项目的用户，我们建议先仔细阅读文档以及FAQ，大多数问题已经在FAQ中有对应的解决方案，除此之外我们鼓励社区反馈问题，以便我们能够逐步扩大支持范围。

<table>
    <tr>
        <td>解析后端</td>
        <td>pipeline</td>
        <td>vlm-transformers</td>
        <td>vlm-sglang</td>
    </tr>
    <tr>
        <td>操作系统</td>
        <td>windows/linux/mac</td>
        <td>windows/linux</td>
        <td>windows(wsl2)/linux</td>
    </tr>
    <tr>
        <td>CPU推理支持</td>
        <td>✅</td>
        <td colspan="2">❌</td>
    </tr>
    <tr>
        <td>GPU要求</td>
        <td>Turing及以后架构，6G显存以上或Apple Silicon</td>
        <td colspan="2">Ampere及以后架构，8G显存以上</td>
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