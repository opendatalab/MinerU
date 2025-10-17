#### 1 系统
NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
昇腾910B2
驱动 23.0.6.2
CANN 7.5.X
Miner U 2.1.9
#### 2 踩坑记录
坑1： **图形库相关的问题，总之就是动态库导致TLS的内存分配失败（OpenCV库在ARM64架构上的兼容性问题）**
⭐这个错误 ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block 是由于OpenCV库在ARM64架构上的兼容性问题导致的。从错误堆栈可以看到，问题出现在导入cv2模块时，这发生在MinerU的VLM后端初始化过程中。
解决方法：
1 安装减少内存问题的opencv版本
```
pip install --upgrade albumentations albucore simsimd# Uninstall current opencv
pip uninstall opencv-python opencv-contrib-python

# Install headless version (no GUI dependencies)
pip install opencv-python-headless

python -c "import cv2; print(cv2.__version__)"2 apt-get install一些包
```
换成清华源然后重命名为sources.list.tuna，然后挪到根目录下面
```
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ focal-security main restricted universe multiversesudo apt-get update -o Dir::Etc::sourcelist="sources.list.tuna" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install libgl1-mesa-glx -o Dir::Etc::sourcelist="sources.list.tuna" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -o Dir::Etc::sourcelist="sources.list.tuna" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install libgl1-mesa-dev libgles2-mesa-dev -o Dir::Etc::sourcelist="sources.list.tuna" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install libgomp1 -o Dir::Etc::sourcelist="sources.list.tuna" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
export OPENCV_IO_ENABLE_OPENEXR=0  export QT_QPA_PLATFORM=offscreen
```
↑这些不知道哪些好使，或者有没有好使的

3  强制覆盖conda环境自带的动态库（conda的和系统的冲突）
```
查找：find /usr/lib /lib /root/.local/conda -name "libgomp.so*" 2>/dev/null
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libstdc++.so.6:/usr/lib/aarch64-linux-gnu/libgomp.so.1"
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
```
此外，还可以把conda环境中自带的的强制挪走
```
mv $CONDA_PREFIX/lib/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.bak
mv $CONDA_PREFIX/lib/libgomp.so.1 $CONDA_PREFIX/lib/libgomp.so.1.bak
mv $CONDA_PREFIX/lib/libGLdispatch.so.0 $CONDA_PREFIX/lib/libGLdispatch.so.0.bak  # 如果有的话
simsimd包相关：
mv /root/.local/conda/envs/pdfparser/lib/python3.10/site-packages/simsimd./libgomp-947d5fa1.so.1.0.0 /root/.local/conda/envs/pdfparser/lib/python3.10/site-packages/simsimd./libgomp-947d5fa1.so.1.0.0.bak
```
或者：
降级simsimd                3.7.2
降级albumentations         1.3.1
sklean包相关：
```
# 找到 scikit-learn 内部的 libgomp 路径
SKLEARN_LIBGOMP="/root/.local/conda/envs/pdfparser/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0"

# 预加载这个特定的 libgomp 版本
export LD_PRELOAD="$SKLEARN_LIBGOMP:$LD_PRELOAD"
```
4 其他
torch / torch_npu 2.5.1
pip install "numpy<2.0" 2.0和昇腾不兼容
export MINERU_MODEL_SOURCE=modelscope
