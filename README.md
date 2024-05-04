Build RAG Application Using a LLM Running on Local Computer with `Ollama` and `LlamaIndex`. Read more from [here](https://medium.com/rahasak/build-rag-application-using-a-llm-running-on-local-computer-with-ollama-and-llamaindex-97703153db20).
require python 3.11.9
Run below commands inside each module ( create separate env within each module )
- run `python -m venv ragindexer` to create virtual env
- ru0n `ragindexer\Scripts\activate` on windows or  `source ragindexer/bin/activate`on linux to activate environment
- pip install --no-cache-dir -r requirements.txt


1. start llm : `docker componse up -d ollama`
2. after service up run `docker exec -it ollama ollama run llama2`
3. start chromaDB : `docker componse up -d chromadb`
4. start RAGIndexer
   1. `cd ./RAGIndexer`
   2. `python -m venv ragindexer`
   3. `ragindexer\Scripts\activate`  OR  `source ragindexer/bin/activate`
   4. run rag_indexer.py file
5. start queryEngine
   1. `cd ./queryEngine`
   2. `python -m venv queryEngine`
   3. `queryEngine\Scripts\activate`  OR  `source queryEngine/bin/activate`
   4. run query_engine.py file


#    runtime: nvidia  # This is for NVIDIA GPUs, adjust for your GPU setup if needed
#    deploy:
#      resources:
#        reservations:
#          devices:
#            - driver: nvidia
#              capabilities: [ gpu ]