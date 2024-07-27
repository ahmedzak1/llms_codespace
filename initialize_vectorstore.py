from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

def load_documents():
    loader = DirectoryLoader("./data/", glob="*.pdf")
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text = text_splitter.split_documents(documents)
    return text

def create_vectorstore(documents, embedding_model_name, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    documents = load_documents()
    split_docs = split_documents(documents)
    vectorstore = create_vectorstore(
        split_docs, 
        "sentence-transformers/all-MiniLM-L6-v2",
        "./chroma_db"
    )
    print("Vectorstore created and persisted successfully.")