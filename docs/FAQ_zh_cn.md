# 常见问题解答

### 1.在较新版本的mac上使用命令安装pip install magic-pdf[full] zsh: no matches found: magic-pdf[full]

在 macOS 上，默认的 shell 从 Bash 切换到了 Z shell，而 Z shell 对于某些类型的字符串匹配有特殊的处理逻辑，这可能导致no matches found错误。
可以通过在命令行禁用globbing特性，再尝试运行安装命令
```bash
setopt no_nomatch
pip install magic-pdf[full]
```

### 2.使用过程中遇到_pickle.UnpicklingError: invalid load key, 'v'.错误

可能是由于模型文件未下载完整导致，可尝试重新下载模型文件后再试  
参考：https://github.com/opendatalab/MinerU/issues/143

### 3.模型文件应该下载到哪里/models-dir的配置应该怎么填

模型文件的路径输入是在"magic-pdf.json"中通过
```json
{
  "models-dir": "/tmp/models"
}
```
进行配置的。
这个路径是绝对路径而不是相对路径，绝对路径的获取可在models目录中通过命令 "pwd" 获取。  
参考：https://github.com/opendatalab/MinerU/issues/155#issuecomment-2230216874

### 4.在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

WSL2的Ubuntu22.04中缺少`libgl`库，可通过以下命令安装`libgl`库解决：
```bash
sudo apt-get install libgl1-mesa-glx
```
参考：https://github.com/opendatalab/MinerU/issues/388
