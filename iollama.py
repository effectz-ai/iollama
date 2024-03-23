from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext


llm = Ollama(
    model="llama2"
)
embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
service_context = ServiceContext.from_defaults(llm=llm,embed_model=embeddings)

# read documents from docs folder
reader = SimpleDirectoryReader(
    input_dir="./docs",
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
index = VectorStoreIndex.from_documents(docs, service_context=service_context, storage_context=storage_context)

query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("what is chitra.")
streaming_response.print_response_stream()

#Settings.llm = llm
#Settings.embed_model = embed_model


#response = llm.complete("What is the history of LEGO?")
#print(response)
