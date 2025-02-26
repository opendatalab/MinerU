# Windows10/11

## 1. 安装cuda和cuDNN

需要安装的版本 CUDA 11.8 + cuDNN 8.7.0

- CUDA 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive
- cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x https://developer.nvidia.com/rdp/cudnn-archive

## 2. 安装anaconda

如果已安装conda，可以跳过本步骤

下载链接：
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

## 3. 使用conda 创建环境

需指定python版本为3.10

```bash
conda create -n MinerU python=3.10
conda activate MinerU
```

## 4. 安装应用

```bash
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple
```

> [!IMPORTANT]
> 下载完成后，务必通过以下命令确认magic-pdf的版本是否正确
>
> ```bash
> magic-pdf --version
> ```
>
> 如果版本号小于0.7.0，请到issue中向我们反馈

## 5. 下载模型

详细参考 [如何下载模型文件](how_to_download_models_zh_cn.md)

## 6. 了解配置文件存放的位置

完成[5.下载模型](#5-下载模型)步骤后，脚本会自动生成用户目录下的magic-pdf.json文件，并自动配置默认模型路径。
您可在【用户目录】下找到magic-pdf.json文件。

> [!TIP]
> windows用户目录为 "C:/Users/用户名"

## 7. 第一次运行

从仓库中下载样本文件，并测试

```powershell
 wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf -O small_ocr.pdf
 magic-pdf -p small_ocr.pdf -o ./output
```

## 8. 测试CUDA加速

如果您的显卡显存大于等于 **8GB** ，可以进行以下流程，测试CUDA解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**

```bash
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 "numpy<2.0.0" --index-url https://download.pytorch.org/whl/cu118
```

**2.修改【用户目录】中配置文件magic-pdf.json中"device-mode"的值**

```json
{
  "device-mode":"cuda"
}
```

**3.运行以下命令测试cuda加速效果**

```bash
magic-pdf -p small_ocr.pdf -o ./output
```

> [!TIP]
> CUDA加速是否生效可以根据log中输出的各个阶段的耗时来简单判断，通常情况下，`layout detection time` 和 `mfr time` 应提速10倍以上。

## 9. 为ocr开启cuda加速

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**

```bash
pip install paddlepaddle-gpu==2.6.1
```

**2.运行以下命令测试ocr加速效果**

```bash
magic-pdf -p small_ocr.pdf -o ./output
```
> [!TIP]
> CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，`ocr time`应提速10倍以上。
