from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def build_qa_chain():
    # Load vector store - same as your existing function but no streaming/callbacks needed here
    vector_store = FAISS.load_local(
        "vector_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=False,  # no streaming for batch script
        temperature=0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain


def run_batch_questions(questions, output_file="chatbot_responses.txt"):
    qa_chain = build_qa_chain()
    with open(output_file, "w", encoding="utf-8") as f:
        for i, question in enumerate(questions, start=1):
            print(f"Processing Question {i}: {question}")
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])

            f.write(f"Q{i}: {question}\n")
            f.write(f"A{i}: {answer}\n\n")
            f.write("Sources:\n")
            for src in sources:
                source_info = src.metadata.get("source", "Unknown source")
                snippet = src.page_content[:500].replace("\n", " ").strip()
                f.write(f"- {source_info}: {snippet}...\n")
            f.write("\n" + "="*80 + "\n\n")
    print(f"Finished. Responses saved to {output_file}")


if __name__ == "__main__":
    questions = [
        "Which article of the Indian Constitution deals with personal liberty and protection of life?",
        "What is the fundamental duty of every citizen according to the Indian Constitution?",
        "Explain the procedure to amend the Constitution of India.",
        "Who appoints the Chief Justice of India?",
        "What are the Directive Principles of State Policy and their importance?",
        "How does the Indian Constitution define the term 'secularism'?",
        "What is the role of the Election Commission as per the Constitution?",
        "Explain the concept of Fundamental Rights and their limitations.",
        "Which articles in the Constitution deal with the emergency provisions?",
    ]
    run_batch_questions(questions)
