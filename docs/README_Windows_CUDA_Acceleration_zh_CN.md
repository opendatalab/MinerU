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
pip install magic-pdf[full]==0.7.0b1 --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> ❗️下载完成后，务必通过以下命令确认magic-pdf的版本是否正确
> 
> ```bash
> magic-pdf --version
>```
> 如果版本号小于0.7.0，请到issue中向我们反馈

## 5. 下载模型
详细参考 [如何下载模型文件](how_to_download_models_zh_cn.md)  
下载后请将models目录移动到空间较大的ssd磁盘目录  
> ❗️模型下载后请务必检查模型文件是否下载完整
> 
> 请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整

## 6. 第一次运行前的配置
在仓库根目录可以获得 [magic-pdf.template.json](../magic-pdf.template.json) 配置模版文件
> ❗️务必执行以下命令将配置文件拷贝到【用户目录】下，否则程序将无法运行
>  
> windows用户目录为 "C:\Users\用户名"
```powershell
(New-Object System.Net.WebClient).DownloadFile('https://gitee.com/myhloli/MinerU/raw/master/magic-pdf.template.json', 'magic-pdf.template.json')
cp magic-pdf.template.json ~/magic-pdf.json
```

在用户目录中找到magic-pdf.json文件并配置"models-dir"为[5. 下载模型](#5-下载模型)中下载的模型权重文件所在目录
> ❗️务必正确配置模型权重文件所在目录的【绝对路径】，否则会因为找不到模型文件而导致程序无法运行
> 
> windows系统中此路径应包含盘符，且需把路径中所有的`"\"`替换为`"/"`,否则会因为转义原因导致json文件语法错误。
> 
> 例如：模型放在D盘根目录的models目录，则model-dir的值应为"D:/models"
```json
{
  "models-dir": "/tmp/models"
}
```

## 7. 第一次运行
从仓库中下载样本文件，并测试
```powershell
(New-Object System.Net.WebClient).DownloadFile('https://gitee.com/myhloli/MinerU/raw/master/demo/small_ocr.pdf', 'small_ocr.pdf')
magic-pdf -p small_ocr.pdf
```

## 8. 测试CUDA加速
如果您的显卡显存大于等于8G，可以进行以下流程，测试CUDA解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**
```bash
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```
> ❗️务必在命令中指定以下版本
> ```bash
> torch==2.3.1 torchvision==0.18.1 
> ```
> 这是我们支持的最高版本，如果不指定版本会自动安装更高版本导致程序无法运行

**2.修改【用户目录】中配置文件magic-pdf.json中"device-mode"的值**
```json
{
  "device-mode":"cuda"
}
```
**3.运行以下命令测试cuda加速效果**
```bash
magic-pdf -p small_ocr.pdf
```
> 提示：CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，`layout detection cost` 和 `mfr time` 应提速10倍以上。

## 9. 为ocr开启cuda加速
> ❗️以下操作需显卡显存大于等于16G才可进行，否则会因为显存不足导致程序崩溃或运行速度下降

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**
```bash
pip install paddlepaddle-gpu==2.6.1
```
**2.运行以下命令测试ocr加速效果**
```bash
magic-pdf -p small_ocr.pdf
```
> 提示：CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，`ocr cost`应提速10倍以上。

