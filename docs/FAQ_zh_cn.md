# 常见问题解答

### 1.离线部署首次运行，报错urllib.error.URLError: <urlopen error [Errno 101] Network is unreachable>

Fixed in 0.6.2b1



### 2.在较新版本的mac上使用命令安装pip install magic-pdf[full-cpu] zsh: no matches found: magic-pdf[full-cpu]

在 macOS 上，默认的 shell 从 Bash 切换到了 Z shell，而 Z shell 对于某些类型的字符串匹配有特殊的处理逻辑，这可能导致no matches found错误。
可以通过在命令行禁用globbing特性，再尝试运行安装命令
```bash
setopt no_nomatch
pip install magic-pdf[full-cpu]
```

### 3.在intel cpu 的mac上 安装最新版的完整功能包 magic-pdf[full-cpu] (0.6.x) 不成功

Fixed in 0.6.2b1


### 4.在部分较新的M芯片macOS设备上，MPS加速开启失败

Not support over 0.7.x

### 5.使用过程中遇到paddle相关的报错FatalError: Illegal instruction is detected by the operating system.

Fixed in 0.6.2b1


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