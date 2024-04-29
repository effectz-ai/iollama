# import libraries
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, PromptTemplate)
from llama_index.vector_stores.chroma import ChromaVectorStore
from flask import Flask, request, jsonify

# initiate Flask app
app = Flask(__name__)

# initiate llm, embed_model and chroma client
def init_all():
    global llm, embed_model, chroma_client 
    llm = Ollama(model="llama2", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# retrieve index
def get_index(collection_name):
    collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

# retrieve query engine
def get_query_engine(index):
    global query_engine
    template = (
        "Imagine you are an advanced AI expert with access to all current and relevant"
        "documents,"
        "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to "
        "questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry"
        "Question: {query_str}\n\n"
        "Answer succinctly, starting with the phrase 'According to my knowledge,' and ensure your response is "
        "understandable to someone without a technical background."
    )
    qa_template = PromptTemplate(template)
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)
    return query_engine

# get response
def get_response(question, collection_name):
    index = get_index(collection_name)
    query_engine = get_query_engine(index)
    response = query_engine.query(question)
    return response.response

# endpoint for search
@app.route('/search', methods=['POST'])
def search():
    question = request.form['question']
    collection = request.form['collection']
    resp = get_response(question, collection)
    data = {'answer': resp}
    return jsonify(data), 200

# run app
if __name__ == '__main__':
    init_all()
    app.run(host='0.0.0.0', port=5001, debug=True)

