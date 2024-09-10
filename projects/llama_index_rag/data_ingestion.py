import os

import click
from llama_index.core.schema import TextNode
from llama_index.embeddings.dashscope import (DashScopeEmbedding,
                                              DashScopeTextEmbeddingModels,
                                              DashScopeTextEmbeddingType)
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from magic_pdf.integrations.rag.api import DataReader

es_vec_store = ElasticsearchStore(
    index_name='rag_index',
    es_url=os.getenv('ES_URL', 'http://127.0.0.1:9200'),
    es_user=os.getenv('ES_USER', 'elastic'),
    es_password=os.getenv('ES_PASSWORD', 'llama_index'),
)


# Create embeddings
# text_type=`document` to build index
def embed_node(node):
    embedder = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    )

    result_embeddings = embedder.get_text_embedding(node.text)
    node.embedding = result_embeddings
    return node


@click.command()
@click.option(
    '-p',
    '--path',
    'path',
    type=click.Path(exists=True),
    required=True,
    help='local pdf filepath or directory',
)
def cli(path):
    output_dir = '/tmp/magic_pdf/integrations/rag/'
    os.makedirs(output_dir, exist_ok=True)
    documents = DataReader(path, 'ocr', output_dir)

    # build nodes
    nodes = []

    for idx in range(documents.get_documents_count()):
        doc = documents.get_document_result(idx)
        if doc is None:  # something wrong happens when parse pdf !
            continue

        for page in iter(
                doc):  # iterate documents from initial page to last page !
            for element in iter(page):  # iterate the element from all page !
                if element.text is None:
                    continue
                nodes.append(
                    embed_node(
                        TextNode(text=element.text,
                                 metadata={'purpose': 'demo'})))
    es_vec_store.add(nodes)


if __name__ == '__main__':
    cli()
