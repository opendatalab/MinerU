# 基于MinerU的PDF解析API

    - MinerU的GPU镜像构建
    - 基于FastAPI的PDF解析接口

支持一键启动，已经打包到镜像中，自带模型权重，支持GPU推理加速，GPU速度相比CPU每页解析要快几十倍不等

## 主要功能
- 删除页眉、页脚、脚注、页码等元素，保持语义连贯
- 对多栏输出符合人类阅读顺序的文本
- 保留原文档的结构，包括标题、段落、列表等
- 提取图像、图片标题、表格、表格标题
- 自动识别文档中的公式并将公式转换成latex
- 自动识别文档中的表格并将表格转换成latex
- 乱码PDF自动检测并启用OCR
- 支持CPU和GPU环境
- 支持windows/linux/mac平台

## 具体原理
请见`PDF-Extract-Kit`:https://github.com/opendatalab/PDF-Extract-Kit/blob/main/README-zh_CN.md
PDF文档中包含大量知识信息，然而提取高质量的PDF内容并非易事。为此，我们将PDF内容提取工作进行拆解：

- 布局检测：使用`LayoutLMv3`模型进行区域检测，如图像，表格,标题,文本等；
- 公式检测：使用`YOLOv8`进行公式检测，包含行内公式和行间公式；
- 公式识别：使用`UniMERNet`进行公式识别；
- 表格识别：使用`StructEqTable`进行表格识别；
- 光学字符识别：使用`PaddleOCR`进行文本识别；
![](https://i-blog.csdnimg.cn/direct/9fe1344768ab407fba31458492454a2b.png)

## 运行环境
Docker，3090 24GB

##   镜像地址

第一步拉取镜像，建议使用阿里云地址，拉取速度比较快，

> 阿里云地址：docker pull registry.cn-beijing.aliyuncs.com/quincyqiang/mineru:0.2-models

> dockerhub地址：docker pull quincyqiang/mineru:0.2-models


##  启动命令


```docker run -itd --name=mineru_server --gpus=all -p 8888:8000 quincyqiang/mineru:0.2-models```

![](https://i-blog.csdnimg.cn/direct/bcff4f524ea5400db14421ba7cec4989.png)

具体截图请见博客：https://blog.csdn.net/yanqianglifei/article/details/141979684


##   启动日志

![](https://i-blog.csdnimg.cn/direct/4eb5657567e4415eba912179dca5c8aa.png)

##  输入参数

访问地址：

    http://localhost:8888/docs

    http://127.0.01:8888/docs

![](https://i-blog.csdnimg.cn/direct/8b3a2bc5908042268e8cc69756e331a2.png)

##  解析效果

![](https://i-blog.csdnimg.cn/direct/a54dcae834ae48d498fb595aca4212c3.png)

返回内容字段包括:dict_keys(['layout', 'info', 'content'])
其中content是一个字典列表：
```json
{
  'type': 'text', 
  'text': '现在我们知道：价值实体就是劳动；劳动量的尺度就是劳动持续时间。', 
  'page_idx': 5
}
```


