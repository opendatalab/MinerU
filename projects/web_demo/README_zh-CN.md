# MinerU本地web_demo
## 功能简介
<p align="center">
  <img src="images/web_demo_1.png" width="600px" style="vertical-align:middle;">
</p>

- 支持上传pdf，并调用MinerU进行处理

- 支持对MinerU解析的Markdown结果进行在线修改

- 支持查看历史任务

## 安装部署

0. MinerU安装部署

```
# 服务依赖MinerU，请先确保MinerU已安装
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
pip3 install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple
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
https://apifox.com/apidoc/shared-b8eda098-ab9c-4cb3-9432-62be9be9c6f7
```
