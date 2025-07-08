# 常见问题解答

## 1.在WSL2的Ubuntu22.04中遇到报错`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

WSL2的Ubuntu22.04中缺少`libgl`库，可通过以下命令安装`libgl`库解决：

```bash
sudo apt-get install libgl1-mesa-glx
```

参考：https://github.com/opendatalab/MinerU/issues/388


## 2.在 CentOS 7 或 Ubuntu 18 系统安装MinerU时报错`ERROR: Failed building wheel for simsimd`

新版本albumentations(1.4.21)引入了依赖simsimd,由于simsimd在linux的预编译包要求glibc的版本大于等于2.28，导致部分2019年之前发布的Linux发行版无法正常安装，可通过如下命令安装:
```
conda create -n mineru python=3.11 -y
conda activate mineru
pip install -U "mineru[pipeline_old_linux]"
```

参考：https://github.com/opendatalab/MinerU/issues/1004
