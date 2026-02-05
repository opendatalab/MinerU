## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04.5 LTS  
cpu: Intel x86-64
xpu: P800
driver: 515.58
docker: 20.10.5
```

## 2. 环境准备

### 2.1 使用 Dockerfile 构建镜像 （vllm）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/kxpu.Dockerfile
docker build --network=host -t mineru:kxpu-vllm-latest -f kxpu.Dockerfile .
```

## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker \
    --device=/dev/xpu0:/dev/xpu0 \
    --device=/dev/xpu1:/dev/xpu1 \
    --device=/dev/xpu2:/dev/xpu2 \
    --device=/dev/xpu3:/dev/xpu3 \
    --device=/dev/xpu4:/dev/xpu4 \
    --device=/dev/xpu5:/dev/xpu5 \
    --device=/dev/xpu6:/dev/xpu6 \
    --device=/dev/xpu7:/dev/xpu7 \
    --device=/dev/xpuctrl:/dev/xpuctrl \
    --net=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /dev/shm:rw,nosuid,nodev,exec,size=32g \
    --cap-add=SYS_PTRACE \
    -v /home/users/vllm-kunlun:/home/vllm-kunlun \
    -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
    -w /workspace \
    -e MINERU_MODEL_SOURCE=local \
    -e MINERU_FORMULA_CH_SUPPORT=true \
    -e MINERU_VLLM_DEVICE=kxpu \
    -it mineru:kxpu-vllm-latest \
    /bin/bash
```

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。


## 4. 注意事项

不同环境下，MinerU对Cambricon加速卡的支持情况如下表所示：

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
    <tr>
      <td colspan="2">数据并行 (--data-parallel-size)</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异

>[!TIP]
> - Kunlunxin加速卡指定可用加速卡的方式与NVIDIA GPU类似，请参考[使用指定GPU设备](https://opendatalab.github.io/MinerU/zh/usage/advanced_cli_parameters/#cuda_visible_devices)章节说明,
>将环境变量`CUDA_VISIBLE_DEVICES`替换为`XPU_VISIBLE_DEVICES`即可。 
> - 在Kunlunxin平台可以通过`xpu-smi`命令查看加速卡的使用情况，并根据需要指定空闲的加速卡ID以避免资源冲突。