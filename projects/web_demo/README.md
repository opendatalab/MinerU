## Mineru 本地API服务

MinerU

```
# 服务依赖mineru，请先确保mineru已安装
```

1. 打包前端界面

```bash
# 先进入前端目录
cd projects/web

# 修改配置
# 将文件vite.config.ts中的target中的IP更改为自己电脑IP

# 打包前端项目
npm install -g yarn
yarn install
yarn build
```

2. 安装服务依赖

```bash
# 先进入后端目录
cd projects/web_demo
# 安装依赖
pip3 install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 启动服务

```bash
# 进入程序目录
cd projects/web_demo/web_demo
# 启动服务
python3 app.py 或者 python app.py
# 在浏览器访问启动的地址即可访问界面
```

ps：接口文档

```
在浏览器打开 mineru-web接口文档.html
```
