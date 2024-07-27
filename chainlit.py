import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
def create_prompt_template():
    template = """Context from PDF: {context}

Question: {question}

Instructions: you are a 10+ years AI engineer, Analyze the provided context from the PDF and answer the question. Your response should be:
1. Accurate and based solely on the given context
2. Concise yet informative

If the answer is not explicitly stated in the context, say so. Do not invent information.

Answer:
- """.strip()
    return PromptTemplate.from_template(template)

def initialize_llm(repo_id, max_length=64, temperature=0.7):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=max_length,
        temperature=temperature,
        huggingfacehub_api_token=os.environ['HUGGINGFACE_API_TOKEN'],
    )

@cl.on_chat_start
async def start():
    # Load the persisted vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    qa_prompt = create_prompt_template()
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = initialize_llm(repo_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer:"]
    )
    cb.answer_reached = True
    res = await chain.acall(
        {"question": message.content, "chat_history": []}, callbacks=[cb]
    )
    answer = res["answer"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:\n"
        for source in sources:
            answer += f"- {source.metadata['source']}\n"

    await cl.Message(content=answer).send()