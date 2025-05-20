import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def create_vector_store(chunks, persist_path="vector_store"):
    embeddings = OpenAIEmbeddings()

    # Build vector store
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Save to disk
    vectorstore.save_local(persist_path)
    print(f"✅ Vector store created and saved at: {persist_path}")


def load_vector_store(persist_path="vector_store"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == "__main__":
    from data_loader import load_constitution_data  # Adjust import path as needed

    data_path = "data/Indian_Constitution.csv"
    chunks = load_constitution_data(data_path)

    create_vector_store(chunks)
    vector_store = load_vector_store()
    print(f"✅ Vector store loaded with {(vector_store.index.ntotal)} vectors.")
