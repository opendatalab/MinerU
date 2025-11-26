# MinerU
## 1. 环境准备
容器启动方式见第3节
### 1.1 获取代码
```
git clone https://github.com/opendatalab/MinerU.git
git checkout fa1149cd4abf9db5e0f13e4e074cdb568be189f4
```
### 1.2 安装依赖
```
source /torch/venv3/pytorch_infer/bin/activate
pip install accelerate==1.11.0 doclayout_yolo==0.0.4 thop==0.1.1.post2209072238 ultralytics-thop==2.0.18 ultralytics==8.3.228
# requirements_check.txt具体内容在下面
pip install -r requirements_check.txt
cd MinerU
pip install -e .[core] --no-deps
```
requirements_check.txt
```
# triton==3.0.0+mlu1.3.1
# torch==2.5.0+cpu
# torchvision==0.20.0+cpu


# === 1. 已安装且版本相同 ===
# (这些包已满足要求, 无需操作)


# === 2. 已安装但版本不同 ===
# (运行 pip install -r 将强制更新到左侧的目标版本)
# accelerate==1.11.0 # 0.33.0
beautifulsoup4==4.14.2 # 4.12.3
cffi==2.0.0 # 1.17.1
huggingface-hub==0.36.0 # 0.25.2
jiter==0.12.0 # 0.8.2
openai==2.8.0 # 1.59.7
pillow==11.3.0 # 10.4.0
sympy==1.14.0 # 1.13.1
tokenizers==0.22.1 # 0.21.0
# torch==2.9.1 # 2.5.0+cpu
# torchvision==0.24.1 # 0.20.0+cpu
transformers==4.57.1 # 4.48.0
# triton==3.5.1 # 3.0.0+mlu1.3.1
typing-extensions==4.15.0 # 4.12.2

# === 3. 未安装 ===
# (运行 pip install -r 将安装这些包)
aiofiles==24.1.0
albucore==0.0.24
albumentations==2.0.8
antlr4-python3-runtime==4.9.3
brotli==1.2.0
coloredlogs==15.0.1
colorlog==6.10.1
cryptography==46.0.3
# doclayout_yolo==0.0.4
fast-langdetect==0.2.5
fasttext-predict==0.9.2.4
ffmpy==1.0.0
flatbuffers==25.9.23
ftfy==6.3.1
gradio-client==1.13.3
gradio-pdf==0.0.22
gradio==5.49.1
groovy==0.1.2
hf-xet==1.2.0
httpx-retries==0.4.5
humanfriendly==10.0
imageio==2.37.2
json-repair==0.53.0
magika==0.6.3
markdown-it-py==4.0.0
mdurl==0.1.2
mineru-vl-utils==0.1.15
mineru==2.6.4
modelscope==1.31.0
# nvidia-cublas-cu12==12.8.4.1
# nvidia-cuda-cupti-cu12==12.8.90
# nvidia-cuda-nvrtc-cu12==12.8.93
# nvidia-cuda-runtime-cu12==12.8.90
# nvidia-cudnn-cu12==9.10.2.21
# nvidia-cufft-cu12==11.3.3.83
# nvidia-cufile-cu12==1.13.1.3
# nvidia-curand-cu12==10.3.9.90
# nvidia-cusolver-cu12==11.7.3.90
# nvidia-cusparse-cu12==12.5.8.93
# nvidia-cusparselt-cu12==0.7.1
# nvidia-nccl-cu12==2.27.5
# nvidia-nvjitlink-cu12==12.8.93
# nvidia-nvshmem-cu12==3.3.20
# nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.0
onnxruntime==1.23.2
orjson==3.11.4
pdfminer.six==20250506
pdftext==0.6.3
polars-runtime-32==1.35.2
polars==1.35.2
pyclipper==1.3.0.post6
pydantic-settings==2.12.0
pydub==0.25.1
pypdf==6.2.0
pypdfium2==4.30.0
python-multipart==0.0.20
reportlab==4.4.4
rich==14.2.0
robust-downloader==0.0.2
ruff==0.14.5
safehttpx==0.1.7
scikit-image==0.25.2
seaborn==0.13.2
semantic-version==2.10.0
shapely==2.1.2
shellingham==1.5.4
simsimd==6.5.3
stringzilla==4.2.3
# thop==0.1.1.post2209072238
tifffile==2025.5.10
typer==0.20.0
typing-inspection==0.4.2
# ultralytics-thop==2.0.18
# ultralytics==8.3.228
```
### 1.3 修改代码
/raid_data/home/yqk/mineru-251114/MinerU/mineru/backend/pipeline/pipeline_analyze.py, line 1
添加代码
```
# 添加MLU支持
import torch_mlu.utils.gpu_migration
# 高版本镜像为
# import torch.mlu.utils.gpu_migration
```

## 2. 使用方法
```
export HF_ENDPOINT=https://hf-mirror.com
mineru-api --host 0.0.0.0 --port 8009
```

## 3. 其他

### 3.1 Dify插件配置问题
给Dify的MinerU插件使用时，需将Dify的.env文件中FILES_URL设置为http://{ip}:{dify的网页访问端口}。
根据网上找到的很多回答可能是要暴露5001，并将FILES_URL设置为http://{ip}:5001，并暴露5001端口，但其实设置为dify的网页访问端口即可。

### 3.2 容器启动方式

```
export MY_CONTAINER="[容器名称]"
num=`docker ps -a|grep "$MY_CONTAINER" | wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -d \
        --privileged \
        --pid=host \
        --net=host \
        --shm-size 64g \
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_ipcm0 \
        --device /dev/cambricon_ctl \
        --name $MY_CONTAINER \
        -v [/path/to/your/data:/path/to/your/data] \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        [镜像名称] \
        sleep infinity
docker exec -ti $MY_CONTAINER /bin/bash
else
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
fi
```

### 3.3 将上面的过程进行打包

准备好前面的requirements_check.txt

Dockerfile

```
# 1. 使用指定的基础镜像
FROM cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310

# 2. 设置环境变量
ENV HF_ENDPOINT=https://hf-mirror.com

# 3. 定义 venv_pip 路径以便复用
# 基础镜像中的虚拟环境路径
ARG VENV_PIP=/torch/venv3/pytorch_infer/bin/pip

# 4. 设置工作目录
WORKDIR /app

# 5. 安装 git (基础镜像可能不包含)
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 6. 复制 requirements_check.txt 到镜像中
# (这个文件需要您在宿主机上和 Dockerfile 放在同一目录下)
COPY requirements_check.txt .

# 7. 步骤 1.1 & 1.2: 获取代码并安装所有依赖
#    在一个 RUN 层中执行所有安装，以优化镜像大小
RUN \
    # 1.1 获取代码
    echo "Cloning MinerU repository..." && \
    git clone https://gh-proxy.org/https://github.com/opendatalab/MinerU.git && \
    cd MinerU && \
    git checkout fa1149cd4abf9db5e0f13e4e074cdb568be189f4 && \
    cd .. && \
    \
    # 1.2 安装依赖
    # 第1个pip install (来自您的步骤)
    echo "Installing initial dependencies..." && \
    ${VENV_PIP} install accelerate==1.11.0 doclayout_yolo==0.0.4 thop==0.1.1.post2209072238 ultralytics-thop==2.0.18 ultralytics==8.3.228 && \
    \
    # 第2个pip install (来自 requirements_check.txt)
    echo "Installing dependencies from requirements_check.txt..." && \
    # 注意：基础镜像已包含 torch 和 triton，requirements_check.txt 中的注释行会被 pip 自动忽略
    ${VENV_PIP} install -r requirements_check.txt && \
    \
    # 第3个pip install (本地安装 MinerU)
    echo "Installing MinerU in editable mode..." && \
    cd MinerU && \
    ${VENV_PIP} install -e .[core] --no-deps

# 8. 步骤 1.3: 修改代码
#    将 MLU 支持代码添加到指定文件的开头
RUN echo "Applying MLU patch to pipeline_analyze.py..." && \
    sed -i '1i# 添加MLU支持\nimport torch_mlu.utils.gpu_migration\n# 高版本镜像为\n# import torch.mlu.utils.gpu_migration\n' \
    /app/MinerU/mineru/backend/pipeline/pipeline_analyze.py
```

该镜像的启动

```
docker run -d --restart=always \
    --privileged \
    --pid=host \
    --net=host \
    --shm-size 64g \
    --device /dev/cambricon_dev0 \
    --device /dev/cambricon_ipcm0 \
    --device /dev/cambricon_ctl \
    --name mineru_service \
    mineru-mlu:latest \
    /torch/venv3/pytorch_infer/bin/python /app/MinerU/mineru/cli/fast_api.py --host 0.0.0.0 --port 8009
```





