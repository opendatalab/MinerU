基于MinerU的PDF解析API

    - MinerU的GPU镜像构建
    - 基于FastAPI的PDF解析接口

支持一键启动，已经打包到镜像中，自带模型权重，支持GPU推理加速，GPU速度相比CPU每页解析要快几十倍不等


##  启动命令：


```docker run -itd --name=mineru_server --gpus=all -p 8888:8000 quincyqiang/mineru:0.1-models```

![](https://i-blog.csdnimg.cn/direct/bcff4f524ea5400db14421ba7cec4989.png)

具体截图请见博客：https://blog.csdn.net/yanqianglifei/article/details/141979684


##   启动日志：

![](https://i-blog.csdnimg.cn/direct/4eb5657567e4415eba912179dca5c8aa.png)

##  输入参数：

访问地址：

    http://localhost:8888/docs

    http://127.0.01:8888/docs

![](https://i-blog.csdnimg.cn/direct/8b3a2bc5908042268e8cc69756e331a2.png)

##  解析效果：

![](https://i-blog.csdnimg.cn/direct/a54dcae834ae48d498fb595aca4212c3.png)



##   镜像地址：

> 阿里云地址：docker pull registry.cn-beijing.aliyuncs.com/quincyqiang/mineru:0.1-models

> dockerhub地址：docker pull quincyqiang/mineru:0.1-models

