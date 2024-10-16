import os

import click
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.dashscope import (DashScopeEmbedding,
                                              DashScopeTextEmbeddingModels,
                                              DashScopeTextEmbeddingType)
from llama_index.vector_stores.elasticsearch import (AsyncDenseVectorStrategy,
                                                     ElasticsearchStore)
# initialize qwen 7B model
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

es_vector_store = ElasticsearchStore(
    index_name='rag_index',
    es_url=os.getenv('ES_URL', 'http://127.0.0.1:9200'),
    es_user=os.getenv('ES_USER', 'elastic'),
    es_password=os.getenv('ES_PASSWORD', 'llama_index'),
    retrieval_strategy=AsyncDenseVectorStrategy(),
)


def embed_text(text):
    embedder = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    )
    return embedder.get_text_embedding(text)


def search(vector_store: ElasticsearchStore, query: str):
    query_vec = VectorStoreQuery(query_embedding=embed_text(query))
    result = vector_store.query(query_vec)
    return '\n'.join([node.text for node in result.nodes])


@click.command()
@click.option(
    '-q',
    '--question',
    'question',
    required=True,
    help='ask what you want to know!',
)
def cli(question):
    tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen-7B-Chat',
                                              revision='v1.0.5',
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('qwen/Qwen-7B-Chat',
                                                 revision='v1.0.5',
                                                 device_map='auto',
                                                 trust_remote_code=True,
                                                 fp32=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        'Qwen/Qwen-7B-Chat', revision='v1.0.5', trust_remote_code=True)

    # define a prompt template for the vectorDB-enhanced LLM generation
    def answer_question(question, context, model):
        if context == '':
            prompt = question
        else:
            prompt = f'''请基于```内的内容回答问题。"
            ```
            {context}
            ```
            我的问题是：{question}。
            '''
        history = None
        print(prompt)
        response, history = model.chat(tokenizer, prompt, history=None)
        return response

    answer = answer_question(question, search(es_vector_store, question),
                             model)
    print(f'question: {question}\n'
          f'answer: {answer}')


"""

python query.py -q 'how about the rights of men'
"""

if __name__ == '__main__':
    cli()
