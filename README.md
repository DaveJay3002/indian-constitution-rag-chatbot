# 📜 Indian Constitution RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the Indian Constitution using LangChain, OpenAI, and Streamlit.

## 🌟 Features

- Interactive Q&A about the Indian Constitution
- Real-time streaming responses
- Source citations for answers
- Efficient document chunking and retrieval
- User-friendly Streamlit interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Data Processing**: Pandas
- **Embeddings**: OpenAI Embeddings

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/DaveJay3002/indian-constitution-rag-chatbot.git
cd indian-constitution-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

4. Process the constitution data and create vector store:
```bash
python vector_store.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
indian-constitution-rag-chatbot/
├── app.py                         # Streamlit application
├── data_loader.py                 # Constitution data processing
├── vector_store.py               # FAISS vector store operations
├── streamlit_callback_handler.py  # Streaming response handler
├── explore_csv.py                # Data exploration utilities
├── data/                         # Raw data directory
│   └── Indian_Constitution.csv
└── vector_store/                 # Generated vector store files
```

## 🔍 How It Works

1. **Data Processing**: The constitution articles are loaded and split into manageable chunks
2. **Vectorization**: Text chunks are converted to embeddings and stored in FAISS
3. **Retrieval**: Relevant chunks are retrieved based on user queries
4. **Generation**: GPT-4 generates accurate answers using retrieved context
5. **Presentation**: Responses are streamed in real-time with source citations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Indian Constitution data source
- LangChain framework
- OpenAI API
- Streamlit community

## ⚠️ Note

Remember to handle the OpenAI API key securely and never commit it to version control.