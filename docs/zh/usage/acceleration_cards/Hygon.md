## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04.3 LTS  
cpu: Hygon C86-4G(x86-64)
dcu: BW200
driver: 6.3.13-V1.12.0a
docker: 20.10.24
```

## 2. 环境准备

### 2.1 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/dcu.Dockerfile
docker build --network=host -t mineru:dcu-vllm-latest -f dcu.Dockerfile .
```


## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker \
    --network=host \
    --ipc=host \
    --shm-size=16G \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e MINERU_MODEL_SOURCE=local \
    -it mineru:dcu-vllm-latest \
    /bin/bash
```

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。


## 4. 注意事项

不同环境下，MinerU对Hygon加速卡的支持情况如下表所示：

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
      <td>🟢</td>
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
      <td>🟢</td>
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
> - DCU加速卡指定可用加速卡的方式与AMD GPU类似，请参考[GPU isolation techniques](https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-isolation.html)
> - 在Hygon平台可以通过`hy-smi`命令查看加速卡的使用情况，并根据需要指定空闲的加速卡ID以避免资源冲突。