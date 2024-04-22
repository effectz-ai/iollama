import logging
import os
import sys

import mimetypes
import time

from chromadb.config import Settings
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import textract
from chromadb import ClientAPI
from chromadb.api.models import Collection
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from llama_index.vector_stores.chroma import ChromaVectorStore
from werkzeug.utils import secure_filename

HTTP_PORT = os.getenv('HTTP_PORT', 7654)


class Config:
    UPLOAD_FOLDER = '/documents/'


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
chromaClient: ClientAPI


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_docx(file_path):
    text = ""
    try:
        import docx
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + '\n'
    except ImportError:
        pass
    return text


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        pdf_text = textract.process(file_path)
        text = pdf_text.decode('utf-8')
    return text


def extract_text_from_txt(file_path):
    text = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def extract_metadata(file):
    filename = secure_filename(file.filename)
    file_type = mimetypes.guess_type(filename)[0]
    return {'filename': filename, 'file_type': file_type}


@app.route('/addToKnowledge', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files')
    collection = request.form['collection']
    return add_to_index_db(files, collection)


@app.route('/clearCollection', methods=['POST'])
def clear_collection_from_db():
    collection = request.form['collection']
    try:
        delete_collection(collection)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'success': True}), 200


def add_to_index_db(files: list, collection_name: str):
    start_time = time.time()
    try:
        sub_fodler = os.path.join(app.config['UPLOAD_FOLDER'], collection_name)
        if not os.path.exists(sub_fodler):
            os.makedirs(sub_fodler)
        for file in files:
            if file and allowed_file(file.filename):
                file_path: str
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(sub_fodler, filename)
                    file.save(file_path, )
                    # get hash of the file
                    hash = str(file.__hash__())

                    text = ""
                    if filename.endswith('.pdf'):
                        text = extract_text_from_pdf(file_path)
                    elif filename.endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    elif filename.endswith('.txt'):
                        text = extract_text_from_txt(file_path)

                    metadata = extract_metadata(file)

                    reader = SimpleDirectoryReader(input_dir=sub_fodler, recursive=True)
                    documents = reader.load_data()
                    collection_ref: Collection = get_or_create_collection(collection_name)
                    vector_store = ChromaVectorStore(chroma_collection=collection_ref)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    VectorStoreIndex.from_documents(documents, storage_context=storage_context,
                                                    embed_model=Settings.embed_model)

                    # collection_ref.add(documents=[text], metadatas=[metadata], ids=[hash])
                except Exception as e:
                    logging.error(e)
                    return jsonify({'error': str(e)}), 400
                # finally:
                #     os.remove(file_path)  # Remove the uploaded file
    except Exception as e:
        logging.error(e)
        return jsonify({'error': str(e)}), 400
    end_time = time.time()
    return jsonify({'doc_count': len(files), 'time_taken': end_time - start_time}), 200


def init_indexer():
    Settings.embed_model = embed_model
    global chromaClient
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    settings = Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials="test-token_Naalin_83290830283")

    chromaClient = chromadb.HttpClient(host='localhost', port=8000, settings=settings)
    print(chromaClient.get_version())
    print(chromaClient.get_settings())
    get_or_create_collection(collection='test')
    print(chromaClient.list_collections())


def get_collection(collection: str) -> Collection:
    return chromaClient.get_collection(collection)


def get_or_create_collection(collection: str) -> Collection:
    return chromaClient.get_or_create_collection(collection)


def delete_collection(collection: str):
    chromaClient.delete_collection(collection)


def rename_collection(collection: str, new_name: str):
    collection = chromaClient.get_collection(collection)
    collection.rename(new_name)


def init_temp_file_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])


if __name__ == '__main__':
    print("running")
    init_indexer()
    init_temp_file_folder()
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
