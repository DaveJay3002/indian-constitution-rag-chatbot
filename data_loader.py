import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_constitution_data(path: str) -> list[Document]:
    # Load the CSV
    df = pd.read_csv(path)

    documents = []

    # Create a Document per article, skipping empty/useless content
    for _, row in df.iterrows():
        title = str(row["article_id"]).strip()
        content = str(row["article_desc"]).strip()

        # Skip articles with no real content
        if not content or len(content.split()) < 5:
            continue

        full_text = f"{title}\n\n{content}"
        doc = Document(page_content=full_text, metadata={"source": title})
        documents.append(doc)

    # Set up splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    all_chunks = []

    # Split each article while retaining source info
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            # Skip garbage/empty chunks
            if len(chunk_text.split()) < 10:
                continue

            # Add article title explicitly at top
            chunk_content = f"{doc.metadata['source']}\n\n{chunk_text}"

            chunk_doc = Document(
                page_content=chunk_content, metadata={"source": doc.metadata["source"]}
            )
            all_chunks.append(chunk_doc)

    print(
        f"✅ Loaded and split {len(documents)} articles into {len(all_chunks)} high-quality chunks."
    )
    return all_chunks


if __name__ == "__main__":
    chunks = load_constitution_data("data/Indian_Constitution.csv")
    print(f"Total chunks created: {len(chunks)}")
    with open("chunks_output.txt", "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, 1):
            f.write(f"\n--- Chunk {idx} ---\n")
            f.write(f"Source: {chunk.metadata['source']}\n")
            f.write(f"Content:\n{chunk.page_content}\n\n")
    print("✅ All chunks have been saved to chunks_output.txt")
