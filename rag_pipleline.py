from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_qa_chain():
    # Load vector store
    vector_store = FAISS.load_local(
        "vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Init OpenAI LLM (Chat Model)
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create QA chain with verbose=True to log prompt and response
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    return qa_chain


def run_qa(qa_chain, query):
    result = qa_chain({"query": query})

    print("\n🧠 Answer:")
    print(result["result"])

    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        # Show first 300 chars and article_id if metadata is present
        meta = doc.metadata.get("source", "No source metadata")
        print(f"• Source: {meta}")
        print(f"  Content snippet: {doc.page_content[:300]}...\n")

    return result
