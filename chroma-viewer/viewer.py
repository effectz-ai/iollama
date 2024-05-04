import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st
import argparse

# parser = argparse.ArgumentParser(description='Set the Chroma DB path to view collections')
# parser.add_argument('db')

pd.set_option('display.max_columns', 4)

def view_collections(dir):
    st.markdown("### DB Path: %s" % dir)
    settings = Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials="test-token_Naalin_83290830283")

    client = chromadb.HttpClient(host='localhost', port=8000, settings=settings)

    # This might take a while in the first execution if Chroma wants to download
    # the embedding transformer
    print(client.list_collections())

    st.header("Collections")

    for collection in client.list_collections():
        data = collection.get(include=["metadatas", "documents",'embeddings'])
        print(data)
        ids = data['ids']
        embeddings = data["embeddings"]
        metadata = data["metadatas"]
        documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        st.markdown("### Collection: **%s**" % collection.name)
        st.dataframe(df)

if __name__ == "__main__":
    try:
        # args = parser.parse_args()
        # print("Opening database: %s" % args.db)
        view_collections("h")
    except:
        pass

