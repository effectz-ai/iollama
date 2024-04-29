# import libraries
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from flask import Flask, request, jsonify
import textract
import os
import docx

# initiate Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# initialize index
def init_index():
    global embed_model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    global chroma_client 
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# update the database
def update_db(files, collection_name):
    document_list = []
    embeddings_list = []
    id_list = []
    id_itr = 0
    try:
        collection_dir = os.path.join(app.config['UPLOAD_FOLDER'], collection_name)
        if not os.path.exists(collection_dir):
            os.makedirs(collection_dir)
        for file in files:
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            if file and allowed_file(file.filename):
                id_itr += 1
                file_path = os.path.join(collection_dir, file.filename)
                file.save(file_path)
                extracted_text = extract_content(file_path)
                if extracted_text:
                    document_list.append(extracted_text)
                    embeddings_list.append(embed_model._embed(extracted_text))
                    id_list.append("id"+str(id_itr))
                    #return jsonify({'content': extracted_text)})
                else:
                    return jsonify({'error': 'Failed to extract text from file'})
            else:
                return jsonify({'error': 'Invalid file format'})
        if len(document_list) > 0:
            collection = chroma_client.get_or_create_collection(collection_name)
            collection.add(
                documents=document_list,
                ids=id_list
            )
            return jsonify({'no. of docs in the collection': collection.count()})
                    
    except Exception as e:
        return jsonify({'errors': str(e)}), 400

# extract content from documents
def extract_content(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension == '.pdf':
        text = textract.process(file_path)
        extracted_text = text.decode('utf-8')
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        extracted_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension == '.txt':
        with open(file_path, 'r') as f:
            extracted_text = f.read()
    else:
        extracted_text = None
    return extracted_text

# check for allowed file formats
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# initialize uploads folder
def init_uploads_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

# endpoint for update collection
@app.route("/update", methods=["POST"])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('file')
    collection = request.form['collection']
    return update_db(files, collection)

# endpoint for delete collection
@app.route('/delete', methods=['POST'])
def delete_collection():
    collection_name = request.form['collection']
    try:
        chroma_client.delete_collection(collection_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'success': True}), 200

# run app
if __name__ == '__main__':
    init_index()
    init_uploads_folder()
    app.run(host='0.0.0.0', port=5002, debug=True)
