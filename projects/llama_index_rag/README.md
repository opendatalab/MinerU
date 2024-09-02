## 安装

MinerU

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU

conda create -n MinerU python=3.10
conda activate MinerU
pip install .[full] --extra-index-url https://wheels.myhloli.com
```

第三方软件

```bash
# install
pip install llama-index-vector-stores-elasticsearch==0.2.0
pip install llama-index-embeddings-dashscope==0.2.0
pip install llama-index-core==0.10.68
pip install einops==0.7.0
pip install transformers-stream-generator==0.0.5
pip install accelerate==0.33.0

# uninstall
pip uninstall transformer-engine
```

## 环境配置

```
export DASHSCOPE_API_KEY={some_key}
export ES_USER={some_es_user}
export ES_PASSWORD={some_es_password}
export ES_URL=http://{es_url}:9200
```

DASHSCOPE_API_KEY 的开通参考[文档](https://help.aliyun.com/zh/dashscope/opening-service)

## 使用

### 导入数据

```bash
python data_ingestion.py -p some.pdf  # load data from pdf

    or

python data_ingestion.py -p /opt/data/some_pdf_directory/ # load data from multiples pdf which under the directory of {some_pdf_directory}
```

### 查询

```bash
python query.py --question '{the_question_you_want_to_ask}'
```

## 示例

````bash
# 启动 es 服务
docker compose up -d

or

docker-compose up -d


# 配置环境变量
export ES_USER=elastic
export ES_PASSWORD=llama_index
export ES_URL=http://127.0.0.1:9200


# 导入数据
python data_ingestion.py example/data/declaration_of_the_rights_of_man_1789.pdf


# 查询问题
python query.py -q 'how about the rights of men'

## outputs
请基于```内的内容回答问题。"
            ```
            I. Men are born, and always continue, free and equal in respect of their rights. Civil distinctions, therefore, can be founded only on public utility.
            ```
            我的问题是：how about the rights of men。

question: how about the rights of men
answer: The statement implies that men are born free and equal in terms of their rights. Civil distinctions should only be based on public utility. However, it does not specify what those rights are. It is up to society and individual countries to determine and protect the specific rights of their citizens.

````

## 开发

`MinerU` 提供了 `RAG` 集成接口，用户可以通过指定输入单个 `pdf` 文件或者某个目录。`MinerU` 会自动解析输入文件并返回可以迭代的接口用于获取数据

### API 接口

```python
from magic_pdf.integrations.rag.type import Node

class RagPageReader:
    def get_rel_map(self) -> list[ElementRelation]:
        # 获取节点的间的关系
        pass
    ...

class RagDocumentReader:
    ...

class DataReader:
    def __init__(self, path_or_directory: str, method: str, output_dir: str):
        pass

    def get_documents_count(self) -> int:
        """获取 pdf 文档数量"""
        pass

    def get_document_result(self, idx: int) -> RagDocumentReader | None:
        """获取某个 pdf 的解析内容"""
        pass


    def get_document_filename(self, idx: int) -> Path:
        """获取某个 pdf 的具体的路径"""
        pass


```

类型定义

```python

class Node(BaseModel):
    category_type: CategoryType = Field(description='类别') # 类别
    text: str | None = Field(description='文本内容',
                             default=None)
    image_path: str | None = Field(description='图或者表格（表可能用图片形式存储）的存储路径',
                                   default=None)
    anno_id: int = Field(description='unique id', default=-1)
    latex: str | None = Field(description='公式或表格 latex 解析结果', default=None)
    html: str | None = Field(description='表格的 html 解析结果', default=None)

```

表格存储形式可能会是 图片、latex、html 三种形式之一。
anno_id 是该 Node 的在全局唯一ID。后续可以用于匹配该 Node 和其他 Node 的关系。节点的关系可以通过方法 `get_rel_map` 获取。用户可以用 `anno_id` 匹配节点之间的关系，并用于构建具备节点的关系的 rag index。

### 节点类型关系矩阵

|                | image_body | table_body |
| -------------- | ---------- | ---------- |
| image_caption  | sibling    |            |
| table_caption  |            | sibling    |
| table_footnote |            | sibling    |
