from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import PromptTemplate


llm = Ollama(
    model="llama2",
    request_timeout=300.0
)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# read documents from docs folder
reader = SimpleDirectoryReader(
    input_dir="./docs-red-team",
    recursive=True,
)
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")


client = QdrantClient("http://localhost:6333")
vector_store = QdrantVectorStore(
    collection_name="iollama",
    client=client,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

#query_engine = index.as_query_engine(streaming=True)
#streaming_response = query_engine.query("what is chitra.")
#streaming_response.print_response_stream()

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question and each answer should start with code word AI Demos: {query_str}\n"
)
qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(text_qa_template=qa_template)
#query_engine.update_prompts(
#    {"response_synthesizer:text_qa_template": qa_template}
#)

while True:
    input_question = input("Enter your question (or 'exit' to quit): ")
    if input_question.lower() == 'exit':
        break

    response = query_engine.query(input_question)
    print(response)
