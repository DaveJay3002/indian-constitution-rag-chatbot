import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_callback_handler import StreamlitCallbackHandler

load_dotenv()

def build_qa_chain(callbacks=None):
    # Load FAISS vector store
    vector_store = FAISS.load_local(
        "vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Init the LLM with streaming + callback support
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0,
        callbacks=callbacks,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Indian Constitution RAG Chatbot", page_icon="📜")
st.title("📜 Indian Constitution RAG Chatbot")
st.write("Ask anything about the Indian Constitution.")

query = st.text_input("📝 Your Question:", placeholder="e.g., Which article of the Indian Constitution deals with personal liberty and protection of life?")

if query:
    with st.spinner("Thinking..."):
        # Prepare streaming output container
        stream_container = st.empty()

        # Initialize callback for streaming
        stream_handler = StreamlitCallbackHandler(stream_container)

        # Build chain with streaming handler
        qa_chain = build_qa_chain(callbacks=[stream_handler])

        # Run the query
        result = qa_chain.invoke({"query": query})

        # After stream finishes, clean up the trailing cursor
        stream_container.markdown(result["result"])

        # Show sources
        st.markdown("📚 **Sources:**")
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            snippet = doc.page_content[:300].replace("\n", " ")
            st.markdown(f"- **{source}**: {snippet}...")
