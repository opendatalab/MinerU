# Windows10/11

## 1. 安装cuda环境

需要安装符合torch要求的cuda版本，具体可参考[torch官网](https://pytorch.org/get-started/locally/)

- CUDA 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive
- CUDA 12.4 https://developer.nvidia.com/cuda-12-4-0-download-archive
- CUDA 12.6 https://developer.nvidia.com/cuda-12-6-0-download-archive
- CUDA 12.8 https://developer.nvidia.com/cuda-12-8-0-download-archive

## 2. 安装anaconda

如果已安装conda，可以跳过本步骤

下载链接：
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

## 3. 使用conda 创建环境

```bash
conda create -n mineru 'python=3.12' -y
conda activate mineru
```

## 4. 安装应用

```bash
pip install -U magic-pdf[full] -i https://mirrors.aliyun.com/pypi/simple
```

> [!IMPORTANT]
> 下载完成后，您可以通过以下命令检查magic-pdf的版本
>
> ```bash
> magic-pdf --version
> ```


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
 wget https://github.com/opendatalab/MinerU/raw/master/demo/pdfs/small_ocr.pdf -O small_ocr.pdf
 magic-pdf -p small_ocr.pdf -o ./output
```

## 8. 测试CUDA加速

如果您的显卡显存大于等于 **6GB** ，可以进行以下流程，测试CUDA解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**(请根据cuda版本选择合适的index-url，具体可参考[torch官网](https://pytorch.org/get-started/locally/))

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
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
> CUDA加速是否生效可以根据log中输出的各个阶段的耗时来简单判断，通常情况下，cuda加速后运行速度比cpu更快。
