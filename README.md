# Simple RAG Application

A Retrieval-Augmented Generation (RAG) application built with Streamlit, ChromaDB, and Google's Gemini API.

## Features

- 📄 **Multi-format Document Support**: Upload PDF, TXT, and DOCX files
- 🔍 **Semantic Search**: Powered by ChromaDB vector database
- 🤖 **AI-Powered Responses**: Uses Google's Gemini API for intelligent answers
- 🎨 **User-Friendly Interface**: Clean Streamlit web interface
- ⚡ **Fast Retrieval**: Efficient vector similarity search
- 🔧 **Configurable**: Customizable chunk sizes and retrieval parameters

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sarankumar1325/SIMPLE-RAG-.git
cd SIMPLE-RAG-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

## How It Works

1. **Document Upload**: Upload your documents through the web interface
2. **Processing**: Documents are chunked and converted to embeddings
3. **Storage**: Embeddings are stored in ChromaDB vector database
4. **Query**: Ask questions about your documents
5. **Retrieval**: Relevant chunks are retrieved using similarity search
6. **Generation**: Gemini API generates answers based on retrieved context

## Configuration

Customize the application by modifying `.env`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_RETRIEVAL_RESULTS`: Number of chunks to retrieve (default: 5)
- `TEMPERATURE`: Gemini API temperature (default: 0.7)

## Project Structure

```
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── document_processor.py # Document loading and chunking
├── vector_store.py       # ChromaDB integration
├── gemini_client.py      # Gemini API wrapper
├── rag_pipeline.py       # Main RAG orchestration
├── utils.py              # Utility functions
└── requirements.txt      # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

