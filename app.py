from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
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


def create_vectorstore(documents, embedding_model_name):
   
    # Create embeddings using the specified model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create the vector store from documents and embeddings
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    return retriever 


def create_prompt_template():
    """
    Create a PromptTemplate instance for QA.
    """
    template = """Context from PDF:
{context}

Question: {question}

Instructions: Analyze the provided context from the PDF and answer the question. Your response should be:
1. Accurate and based solely on the given context
2. Concise yet informative
3. Structured with bullet points for clarity

If the answer is not explicitly stated in the context, say so. Do not invent information.

Answer:
- """.strip()
    return PromptTemplate.from_template(template)



def initialize_llm(repo_id, max_length=64, temperature=0.5):

    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=max_length,
        temperature=temperature,
        huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
    )   


def create_and_run_qa_chain(llm, retriever, qa_prompt, question, chat_history):
    """
    Create and run a ConversationalRetrievalChain.

    Parameters:
    - llm (HuggingFaceEndpoint): The LLM instance.
    - retriever (Chroma): The retriever instance.
    - qa_prompt (PromptTemplate): The prompt template for QA.
    - question (str): The question to be asked.
    - chat_history (list): List of previous Q&A pairs.

    Returns:
    - dict: The result from running the QA chain.
    """
    # Create ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    # Run the chain
    result = qa_chain({"question": question, "chat_history": chat_history})

    return result


if __name__ == "__main__":
    # Load and process documents
    documents = load_documents()
    split_docs = split_documents(documents)
    retriever = create_vectorstore(split_docs, "sentence-transformers/all-MiniLM-L6-v2")
    qa_prompt = create_prompt_template()
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = initialize_llm(repo_id)
    # Define question and chat history
    question = "Explain Massive Multitask Language Understanding?"
    chat_history = []  # This would contain previous Q&A pairs if any

    # Run the QA chain
    result = create_and_run_qa_chain(llm, retriever, qa_prompt, question, chat_history)

    # Print the result
    print(result[answer])