import os

import chromadb
import logging

from chromadb import ClientAPI
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core import (Settings,VectorStoreIndex, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.api.models import Collection

embedding_model: HuggingFaceEmbedding
llm: Ollama
chroma_client: ClientAPI
query_engine = None

HTTP_PORT = os.getenv('HTTP_PORT', 7655)
app = Flask(__name__)
CORS(app)


@app.route('/api/chat', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    question = json['question']
    user_id = json['user_id']
    colelction = json['collection']
    logging.info("post question `%s` for user `%s`", question, user_id)

    resp = chat(question, user_id, colelction)
    data = {'answer': resp}

    return jsonify(data), 200


def init_llm():
    global llm
    llm = Ollama( model="llama2",request_timeout=300.0)
    Settings.llm = llm


def int_embed():
    global embedding_model
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embedding_model


def init_db():
    global chroma_client
    from chromadb.config import Settings

    settings = Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials="test-token_Naalin_83290830283")
    chroma_client = chromadb.HttpClient(host='localhost', port=8000, settings=settings)


def init_index(collection_name: str):
    chroma_collection: Collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # use this to set custom chunk size and splitting
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def chat(input_question, user, collection_name):
    index = init_index(collection_name)
    query_engine = init_query_engine(index)
    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response


def init_query_engine(index):
    global query_engine

    # custome prompt template
    template = (
        "Imagine you are an advanced AI expert in cyber security laws, with access to all current and relevant legal "
        "documents,"
        "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to "
        "questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references to "
        "applicable laws,"
        "precedents, or principles where appropriate:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly, starting with the phrase 'According to cyber security law,' and ensure your response is "
        "understandable to someone without a legal background."
    )
    qa_template = PromptTemplate(template)

    # build query engine with custom template
    # text_qa_template specifies custom template
    # similarity_top_k configure the retriever to return the top 3 most similar documents,
    # the default value of similarity_top_k is 2
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


if __name__ == '__main__':
    print("Start : Query Engine")
    init_llm()
    int_embed()
    init_db()
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
