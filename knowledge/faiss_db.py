from langchain_community.document_loaders import TextLoader, DirectoryLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from .prompts import *
from pathlib import Path

import configparser

# Create a ConfigParser instance
config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

cwd = Path(__file__).parent

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["name"] = record.get("name")
    metadata["description"] = record.get("description")
    metadata["URL"] = record.get("URL")

    return metadata

def load_db(DATA_CONFIG=config['DATA']):
    match DATA_CONFIG['load_method']:
        case "directory"|"DIRECTORY":
            loader = DirectoryLoader(
                cwd/'data', glob="**/*.json"
            )
        case "JSON" | "json":
            loader = JSONLoader(
                file_path=DATA_CONFIG['source'],
                jq_schema='.agents[]',
                content_key="prompt",
                metadata_func=metadata_func
            )

    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_retriever(db):
    return db.as_retriever()

# retriever
def retrieve(db, query:str):
    return get_retriever(db).invoke(query)

def retrieve_similar(
    db, query:str, DATA_CONFIG=config['DATA']):
    max_score = 0
    match DATA_CONFIG['search_method']:
        case "similar_search":
            results = db.similarity_search_with_score(query)
        case "max_marginal":
            results = db.max_marginal_relevance_search(
        query, filter=dict(page=1))
    for doc, score in results:
        if score > max_score:
            max_score = score
            data = f"Content:{doc.page_content}\nMetadata:{doc.metadata}"
        # print(f"Content: \n{doc.page_content}")
        # print(f"Metadata: \n{doc.metadata}\n\n")
    return data

if __name__=="__main__":
    default_query = "Which genie is most appropriate for asking about the impact of drug on a person's health?"
    import sys
    query = sys.argv[1]
    if not query or len(query)==0:
        query = default_query
    answer = retrieve_similar(query)
    print(answer)