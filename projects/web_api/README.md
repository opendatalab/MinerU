# 基于MinerU的PDF解析API

- MinerU的GPU镜像构建
- 基于FastAPI的PDF解析接口

## 构建方式

```
docker build -t mineru-api .
```

或者使用代理：

```
docker build --build-arg http_proxy=http://127.0.0.1:7890 --build-arg https_proxy=http://127.0.0.1:7890 -t mineru-api .
```

## 启动命令

```
docker run --rm -it --gpus=all -v ./paddleocr:/root/.paddleocr -p 8000:8000 mineru-api
```

初次调用 API 时会自动下载 paddleocr 的模型（约数十 MB），其余模型已包含在镜像中。

## 测试参数

访问地址：

```
http://localhost:8000/docs
http://127.0.0.1:8000/docs
```

## 旧版镜像地址

> 阿里云地址：docker pull registry.cn-beijing.aliyuncs.com/quincyqiang/mineru:0.1-models
>
> dockerhub地址：docker pull quincyqiang/mineru:0.1-models


## 旧版截图

### 启动命令

![](https://i-blog.csdnimg.cn/direct/bcff4f524ea5400db14421ba7cec4989.png)

具体截图请见博客：https://blog.csdn.net/yanqianglifei/article/details/141979684

### 启动日志

![](https://i-blog.csdnimg.cn/direct/4eb5657567e4415eba912179dca5c8aa.png)

### 测试参数

![](https://i-blog.csdnimg.cn/direct/8b3a2bc5908042268e8cc69756e331a2.png)

### 解析效果

![](https://i-blog.csdnimg.cn/direct/a54dcae834ae48d498fb595aca4212c3.png)
