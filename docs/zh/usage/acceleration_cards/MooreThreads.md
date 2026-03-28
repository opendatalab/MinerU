## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04.4 LTS  
cpu: Intel x86-64
dcu: MTT S4000
driver: 3.0.0-rc-KuaE2.0
docker: 24.0.7
```

## 2. 环境准备

### 2.1 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/musa.Dockerfile
docker build --network=host -t mineru:musa-vllm-latest -f musa.Dockerfile .
```


## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker \
    --network=host \
    --ipc=host \
    --shm-size=80g \
    --privileged \
    -e MTHREADS_VISIBLE_DEVICES=all \
    -e MINERU_VLLM_DEVICE=musa \
    -e MINERU_MODEL_SOURCE=local \
    -it mineru:musa-vllm-latest \
    /bin/bash
```

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。


## 4. 注意事项

不同环境下，MinerU对MooreThreads加速卡的支持情况如下表所示：

>[!NOTE]
> **兼容性说明**：由于摩尔线程（MooreThreads）目前对 vLLM v1 引擎的支持尚待完善，MinerU 现阶段采用 v0 引擎作为适配方案。
> 受此限制，vLLM 的异步引擎（Async Engine）功能存在兼容性问题，可能导致部分使用场景无法正常运行。
> 我们将持续跟进摩尔线程对 vLLM v1 引擎的支持进展，并及时在 MinerU 中进行相应的适配与优化。

<table border="1">
  <thead>
    <tr>
      <th rowspan="2" colspan="2">使用场景</th>
      <th colspan="2">容器环境</th>
    </tr>
    <tr>
      <th>vllm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">命令行工具(mineru)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="3">fastapi服务(mineru-api)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="3">gradio界面(mineru-gradio)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">openai-server服务（mineru-openai-server）</td>
      <td>🟢</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异

>[!TIP]
> - MooreThreads加速卡指定可用加速卡的方式与NVIDIA GPU类似，请参考[GPU 枚举](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/install_guide/#gpu-%E6%9E%9A%E4%B8%BE)
> - 在MooreThreads平台可以通过`mthreads-gmi`命令查看加速卡的使用情况，并根据需要指定空闲的加速卡ID以避免资源冲突。