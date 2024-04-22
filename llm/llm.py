from llama_index.core import (Settings)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from flask import Flask
from flask_cors import CORS
from flask import jsonify
from flask import request

llm:Ollama
embedding_model:HuggingFaceEmbedding

app = Flask(__name__)
CORS(app)

# @app.route('/chat', methods=['POST'])
# def chat():
#     json = request.get_json(silent=True)
#     question = json['question']
#     chat_response = llm.chat(question)
#     return jsonify(chat_response)
# def init_llm(model: str, request_timeout: float, embed_model:str):
#     global llm
#     global embedding_model
#     llm = Ollama(model=model, request_timeout=request_timeout)
#     embedding_model = HuggingFaceEmbedding(model_name=embed_model)
#     Settings.llm = llm
#     Settings.embed_model = embedding_model

