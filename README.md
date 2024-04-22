Build RAG Application Using a LLM Running on Local Computer with `Ollama` and `LlamaIndex`. Read more from [here](https://medium.com/rahasak/build-rag-application-using-a-llm-running-on-local-computer-with-ollama-and-llamaindex-97703153db20).
require python 3.11.9
Run below commands inside each module ( create separate env within each module )
- run `python -m venv ragindexer` to create virtual env
- ru0n `ragindexer\Scripts\activate` on windows or  `source ragindexer/bin/activate`on linux to activate environment
- pip install --no-cache-dir -r requirements.txt

#    runtime: nvidia  # This is for NVIDIA GPUs, adjust for your GPU setup if needed
#    deploy:
#      resources:
#        reservations:
#          devices:
#            - driver: nvidia
#              capabilities: [ gpu ]