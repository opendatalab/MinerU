# Frequently Asked Questions

## 1. Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2

The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:

```bash
sudo apt-get install libgl1-mesa-glx
```

Reference: https://github.com/opendatalab/MinerU/issues/388


## 2. Error when installing MinerU on CentOS 7 or Ubuntu 18: `ERROR: Failed building wheel for simsimd`

The new version of albumentations (1.4.21) introduces a dependency on simsimd. Since the pre-built package of simsimd for Linux requires a glibc version greater than or equal to 2.28, this causes installation issues on some Linux distributions released before 2019. You can resolve this issue by using the following command:
```
conda create -n mineru python=3.11 -y
conda activate mineru
pip install -U "mineru[pipeline_old_linux]"
```

Reference: https://github.com/opendatalab/MinerU/issues/1004
