## Installation

MinerU

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU

conda create -n MinerU python=3.10
conda activate MinerU
pip install .[full] --extra-index-url https://wheels.myhloli.com
```

Third-party software

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

## Environment Configuration

```
export DASHSCOPE_API_KEY={some_key}
export ES_USER={some_es_user}
export ES_PASSWORD={some_es_password}
export ES_URL=http://{es_url}:9200
```
For instructions on obtaining a DASHSCOPE_API_KEY, refer to [documentation](https://help.aliyun.com/zh/dashscope/opening-service)

## Usage

### Data Ingestion

```bash
python data_ingestion.py -p some.pdf  # load data from pdf

    or

python data_ingestion.py -p /opt/data/some_pdf_directory/ # load data from multiples pdf which under the directory of {some_pdf_directory}
```

### Query

```bash
python query.py --question '{the_question_you_want_to_ask}'
```

## Example

````bash
# Start the es service
docker compose up -d

or

docker-compose up -d


# Set environment variables
export ES_USER=elastic
export ES_PASSWORD=llama_index
export ES_URL=http://127.0.0.1:9200
export DASHSCOPE_API_KEY={some_key}


# Ingest data
python data_ingestion.py example/data/declaration_of_the_rights_of_man_1789.pdf


# Ask a question
python query.py -q 'how about the rights of men'

## outputs
Please answer the question based on the content within ```:
            ```
            I. Men are born, and always continue, free and equal in respect of their rights. Civil distinctions, therefore, can be founded only on public utility.
            ```
            My question is：how about the rights of men。

question: how about the rights of men
answer: The statement implies that men are born free and equal in terms of their rights. Civil distinctions should only be based on public utility. However, it does not specify what those rights are. It is up to society and individual countries to determine and protect the specific rights of their citizens.

````

## Development

`MinerU` provides a `RAG` integration interface, allowing users to specify a single input `pdf` file or a directory. `MinerU` will automatically parse the input files and return an iterable interface for retrieving the data.


### API Interface

```python
from magic_pdf.integrations.rag.type import Node

class RagPageReader:
    def get_rel_map(self) -> list[ElementRelation]:
        # Retrieve the relationships between nodes
        pass
    ...

class RagDocumentReader:
    ...

class DataReader:
    def __init__(self, path_or_directory: str, method: str, output_dir: str):
        pass

    def get_documents_count(self) -> int:
        """Get the number of pdf documents"""
        pass

    def get_document_result(self, idx: int) -> RagDocumentReader | None:
        """Retrieve the parsed content of a specific pdf"""
        pass


    def get_document_filename(self, idx: int) -> Path:
        """Retrieve the path of a specific pdf"""
        pass


```

Type Definitions

```python


class Node(BaseModel):
    category_type: CategoryType = Field(description='Category') # Category
    text: str | None = Field(description='Text content', default=None)
    image_path: str | None = Field(description='Path to image or table (table may be stored as an image)', default=None)
    anno_id: int = Field(description='Unique ID', default=-1)
    latex: str | None = Field(description='LaTeX output for equations or tables', default=None)
    html: str | None = Field(description='HTML output for tables', default=None)



```

Tables can be stored in one of three formats: image, LaTeX, or HTML. 
`anno_id` is a globally unique ID for each Node. It can be used later to match this Node with other Nodes. The relationships between nodes can be retrieved using the `get_rel_map` method. Users can use `anno_id` to link nodes and construct a RAG index that includes node relationships.


### Node Relationship Matrix

|                | image_body | table_body |
| -------------- | ---------- | ---------- |
| image_caption  | sibling    |            |
| table_caption  |            | sibling    |
| table_footnote |            | sibling    |
