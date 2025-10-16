# Local RAG Document Chat

KIRA - Knowledge Interface Retrieval Agent - ist a local RAG system that allows you to chat with your documents 
using open source, locally runLLMs. Think of it as a small but self-hosted and private alternative to services like 
Google NotebookLM. 

## Features

- **Private & Local** - No data leaves your machine, no need for API keys
- **Multi-format Support** - Supports .pdf and .txt files
- **Open Source** - Uses Mistral, can also use Llama 3.2 via Ollama
- **Interactive Chat** - Simple web-based UI built with Gradio
- **Semantic Search** - Find relevant information in your documents


## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- 8GB RAM at minimum, 16GB RAM recommended


## Installation

1. **Clone this repository**

```bash
git clone https://github.com/BVoermann/kira.git
cd kira
```

2. **Create virtual environment and install requirements**

*Linux*
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Windows*
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

3. **Install and set up Ollama**

Download from [Ollama](https://ollama.ai)

Then either download llama3.2 or mistral.

```bash
ollama pull llama3.2
```

```bash
ollama pull mistral
```

## Usage

1. **Start application**

*Linux*
```bash
python3 app.py
```

*Windows*
```bash
python app.py
```

2. **Open your Browser**

Navigate to `http://127.0.0.1:7860`

3. **Upload Documents**

- Select one or more PDF or TXT files
- Click "Process Documents" and wait for them to be processed
- Ask questions in the chat
- The AI will answer based on the content of the documents

## Project Structure
```
local-rag-chat/
├── app.py                    # Main Gradio interface
├── document_processor.py     # Document loading and vectorization
├── rag_engine.py            # RAG query engine with LLM
└── chroma_db/               # Vector database storage (created on first run)
```

## Configuration

### Change the LLM Model

Edit `app.py` line 19:
```python
rag_engine = RAGEngine(doc_processor.vectorstore, model_name="mistral")
```

### Adjust Chunk Size

Edit `document_processor.py` lines 34-36:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust this
    chunk_overlap=200,    # And this
    length_function=len
)
```

### Change Embedding Model

Edit `document_processor.py` line 12:
```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Adjust this
)
```


## Troubleshooting

### Ollama Connection Error
Make sure Ollama is running:
```bash
ollama list  # Should show installed models
```

### Memory Issues
- Reduce `chunk_size` in `document_processor.py`
- Use a smaller model like `mistral` instead of `llama3.2`
- Process fewer documents at once

### Slow Performance
- Use a smaller embedding model like `all-MiniLM-L6-v2`
- Reduce the number of retrieved chunks (change `k=4` in `rag_engine.py`)
