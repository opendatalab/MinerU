## 安装

MinerU

```bash
# mineru已安装则跳过此步骤

git clone https://github.com/opendatalab/MinerU.git
cd MinerU

conda create -n MinerU python=3.10
conda activate MinerU
pip install .[full] --extra-index-url https://wheels.myhloli.com
```

第三方软件

```bash
cd projects/web_api
pip install poetry
portey install
```

启动服务

```bash
cd web_api
python app.py
```

接口文档
```
在浏览器打开 mineru-web接口文档.html
```
