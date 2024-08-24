# Frequently Asked Questions

### 1. When using the command `pip install magic-pdf[full]` on newer versions of macOS, the error `zsh: no matches found: magic-pdf[full]` occurs.

On macOS, the default shell has switched from Bash to Z shell, which has special handling logic for certain types of string matching. This can lead to the "no matches found" error. You can try disabling the globbing feature in the command line and then run the installation command again.

```bash
setopt no_nomatch
pip install magic-pdf[full]
```

### 2. Encountering the error `pickle.UnpicklingError: invalid load key, 'v'.` during use

This might be due to an incomplete download of the model file. You can try re-downloading the model file and then try again.  
Reference: https://github.com/opendatalab/MinerU/issues/143

### 3. Where should the model files be downloaded and how should the `/models-dir` configuration be set?

The path for the model files is configured in "magic-pdf.json". just like:

```json
{
  "models-dir": "/tmp/models"
}
```

This path is an absolute path, not a relative path. You can obtain the absolute path in the models directory using the "pwd" command.  
Reference: https://github.com/opendatalab/MinerU/issues/155#issuecomment-2230216874

### 4. Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2

The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:

```bash
sudo apt-get install libgl1-mesa-glx
```

Reference: https://github.com/opendatalab/MinerU/issues/388

### 5. Encountered error `ModuleNotFoundError: No module named 'fairscale'`
You need to uninstall the module and reinstall it:
```bash
pip uninstall fairscale
pip install fairscale
```
Reference: https://github.com/opendatalab/MinerU/issues/411
