# 常见问题解答

### 1.离线部署首次运行，报错urllib.error.URLError: <urlopen error [Errno 101] Network is unreachable>
    
首次运行需要在线下载一个小的语言检测模型，如果是离线部署需要手动下载该模型并放到指定目录。  
参考：https://github.com/opendatalab/MinerU/issues/121

### 2.在较新版本的mac上使用命令安装pip install magic-pdf[full-cpu] zsh: no matches found: magic-pdf[full-cpu]

在 macOS 上，默认的 shell 从 Bash 切换到了 Z shell，而 Z shell 对于某些类型的字符串匹配有特殊的处理逻辑，这可能导致no matches found错误。
可以通过在命令行禁用globbing特性，再尝试运行安装命令
```bash
setopt no_nomatch
pip install magic-pdf[full-cpu]
```

### 3.在intel cpu 的mac上 安装最新版的完整功能包 magic-pdf[full-cpu] (0.6.x) 不成功

完整功能包依赖的公式解析库unimernet限制了pytorch的最低版本为2.3.0，而pytorch官方没有为intel cpu的macOS 提供2.3.0版本的预编译包，所以会产生依赖不兼容的问题。
可以先尝试安装unimernet的老版本之后再尝试安装完整功能包的其他依赖。（为避免依赖冲突，请激活一个全新的虚拟环境）
```bash
pip install magic-pdf
pip install unimernet==0.1.0
pip install matplotlib ultralytics paddleocr==2.7.3 paddlepaddle
pip install detectron2 --extra-index-url https://myhloli.github.io/wheels/ 
```

### 4.在部分较新的M芯片macOS设备上，MPS加速开启失败

卸载torch和torchvision，重新安装nightly构建版torch和torchvision
```bash
pip uninstall torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```
参考: https://github.com/opendatalab/PDF-Extract-Kit/issues/23

### 5.使用过程中遇到paddle相关的报错FatalError: Illegal instruction is detected by the operating system.

paddlepaddle 2.6.1与部分linux系统环境存在兼容性问题。
可尝试~~降级到2.5.2~~升级到3.0.0b1使用，
```bash
pip install paddlepaddle==3.0.0b1
```
~~或卸载paddlepaddle，重新安装paddlepaddle-gpu~~

参考：https://github.com/opendatalab/MinerU/issues/224

### 6.使用过程中遇到_pickle.UnpicklingError: invalid load key, 'v'.错误

可能是由于模型文件未下载完整导致，可尝试重现下载模型文件后再试  
参考：https://github.com/opendatalab/MinerU/issues/143

### 7.模型文件应该下载到哪里/models-dir的配置应该怎么填

模型文件的路径输入是在"magic-pdf.json"中通过
```json
{
  "models-dir": "/tmp/models"
}
```
进行配置的。
这个路径是绝对路径而不是相对路径，绝对路径的获取可在models目录中通过命令 "pwd" 获取。  
参考：https://github.com/opendatalab/MinerU/issues/155#issuecomment-2230216874

### 8.命令行中 --model "model_json_path" 指的是什么？

model_json 指的是通过模型分析后生成的一种有特定格式的json文件。  
如果使用 https://github.com/opendatalab/PDF-Extract-Kit 项目生成，该文件一般在项目的output目录下。  
如果使用 MinerU 的命令行调用内置的模型分析，该文件一般在输出路径"/tmp/magic-pdf/pdf-name"下。  
参考：https://github.com/opendatalab/MinerU/issues/128

### 9.报错：Required dependency not installed, please install by "pip install magic-pdf[full-cpu] detectron2 --extra-index-url https://myhloli.github.io/wheels/"

通过更新0.6.2b1来解决
```bash
pip install magic-pdf[full]==0.6.2b1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```