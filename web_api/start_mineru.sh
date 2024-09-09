docker run -itd --name=mineru_server --gpus=all -p 8888:8000 quincyqiang/mineru:0.1-models /bin/bash

docker run -itd --name=mineru_server --gpus=all -p 8888:8000 quincyqiang/mineru:0.3-models

docker login --username=1185918903@qq.com registry.cn-beijing.aliyuncs.com
docker tag quincyqiang/mineru:0.3-models registry.cn-beijing.aliyuncs.com/quincyqiang/gomate:0.3-models
docker push registry.cn-beijing.aliyuncs.com/quincyqiang/gomate:0.3-models