# 📚 Personal Knowledge Base Assistant

A RAG (Retrieval-Augmented Generation) system that allows you to upload documents and ask questions about them using AI.

## 🎯 What Does This Project Do?

This is an intelligent document Q&A system that:
1. **Uploads** your PDF/TXT documents
2. **Processes** them into searchable chunks
3. **Stores** them in a vector database (ChromaDB)
4. **Retrieves** relevant information when you ask questions
5. **Generates** precise answers using an LLM (Large Language Model)

## 🏗️ Project Structure

```
personal_knowledge_assistant/
├── app.py                      # Streamlit web interface
├── main.py                     # Command-line interface (CLI)
├── config.py                   # All configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Document storage
│   ├── raw/                   # Original uploaded documents
│   └── processed/             # Processed/chunked documents (cache)
├── logs/                       # Application logs
├── src/                        # Source code modules
│   ├── document_processor.py  # Load and chunk documents
│   ├── embeddings_manager.py  # Handle text embeddings
│   ├── retriever.py           # Search documents
│   ├── rag_chain.py          # Main RAG logic with LLM
│   └── utils.py              # Logging and utilities
├── temp_uploads/              # Temporary file uploads (Streamlit)
└── vectorstore/               # ChromaDB vector database
    └── chroma/               # Stored embeddings
```

## 📄 File Descriptions

### Core Application Files

#### `app.py` - Web Interface
- **Purpose**: Streamlit web UI for document upload and Q&A
- **What it does**:
  - Provides file upload interface for PDF/TXT files
  - Saves uploaded files temporarily
  - Processes and stores documents in ChromaDB
  - Displays Q&A interface
  - Shows answers with proper formatting
- **Key Features**:
  - Sidebar for document upload
  - Real-time status messages
  - Chat-style interface
  - Success/error notifications

#### `main.py` - Command Line Interface
- **Purpose**: CLI tool for batch operations and testing
- **What it does**:
  - Add documents via command line: `python main.py --add path/to/docs`
  - Query via command line: `python main.py --query "Your question"`
  - Useful for automation and scripting
- **Use Cases**:
  - Batch document processing
  - Testing without UI
  - Integration with other scripts

#### `config.py` - Configuration Hub
- **Purpose**: Centralized configuration for entire project
- **What it does**:
  - Defines all file paths (data, logs, vectorstore)
  - Sets embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
  - Configures LLM settings (`mistral-community/Mistral-7B-Instruct-v0.1`)
  - Sets chunking parameters (size: 500, overlap: 50)
  - Retrieval settings (k=3 documents)
  - Loads environment variables (HF API key)
- **Key Settings**:
  ```python
  EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
  HF_MODEL_FOR_QA = "mistral-community/Mistral-7B-Instruct-v0.1"
  CHUNK_SIZE = 500
  CHUNK_OVERLAP = 50
  RETRIEVAL_K = 3
  ```

### Source Code Modules (`src/`)

#### `document_processor.py` - Document Processing Pipeline
- **Purpose**: Load and split documents into chunks
- **What it does**:
  1. Loads documents from directory (PDF, TXT)
  2. Uses LangChain document loaders:
     - `PyPDFLoader` for PDFs
     - `TextLoader` for text files
  3. Splits documents into chunks using `RecursiveCharacterTextSplitter`
  4. Handles errors gracefully (skips empty docs)
  5. Returns list of chunked documents
- **Key Functions**:
  - `process_documents(path)`: Main entry point
  - Automatically detects file types
  - Configurable chunk size and overlap

#### `embeddings_manager.py` - Text Embeddings Handler
- **Purpose**: Convert text into vector embeddings
- **What it does**:
  - Initializes HuggingFace embedding model
  - Uses `sentence-transformers/all-MiniLM-L6-v2`
  - Creates 384-dimensional vectors from text
  - Runs locally (no API calls for embeddings)
  - Cached for reuse across application
- **Why Embeddings?**:
  - Convert text to numbers (vectors)
  - Similar texts have similar vectors
  - Enables semantic search (meaning-based, not keyword)

#### `retriever.py` - Document Search
- **Purpose**: Search and retrieve relevant documents
- **What it does**:
  - Loads ChromaDB vector store
  - Searches for most relevant chunks given a query
  - Returns top k=3 most similar documents
  - Uses cosine similarity for matching
- **Key Functions**:
  - `retrieve_documents(query)`: Returns relevant docs
  - Works with embeddings to find semantic matches

#### `rag_chain.py` - RAG Logic with LLM
- **Purpose**: Core RAG (Retrieval-Augmented Generation) pipeline
- **What it does**:
  1. **Retrieves** relevant documents from vector store
  2. **Formats** documents into context
  3. **Creates** prompt with context + question
  4. **Sends** to LLM (Mistral-7B via HuggingFace API)
  5. **Returns** AI-generated answer
- **Key Components**:
  - `create_rag_chain()`: Builds the RAG pipeline
  - `answer_question(question)`: Main entry point
- **LangChain LCEL Pipeline**:
  ```python
  retriever → format_docs → prompt → LLM → answer
  ```
- **Prompt Template**:
  ```
  Context: [retrieved documents]
  Question: [user question]
  Answer: [LLM generates precise answer]
  ```

#### `utils.py` - Utilities
- **Purpose**: Logging and helper functions
- **What it does**:
  - Sets up logging to file and console
  - Provides formatted log messages
  - Date-stamped log files
  - Error tracking

## 🔧 How It Works (Technical Flow)

### 1. Document Upload Flow
```
User uploads file (app.py)
    ↓
Save to temp_uploads/
    ↓
document_processor.py loads & chunks
    ↓
embeddings_manager.py creates vectors
    ↓
Store in ChromaDB vectorstore/
```

### 2. Question Answering Flow
```
User asks question (app.py)
    ↓
rag_chain.py receives question
    ↓
retriever.py searches ChromaDB
    ↓
Top 3 relevant chunks retrieved
    ↓
Format into prompt with context
    ↓
Send to HuggingFace LLM (Mistral-7B)
    ↓
LLM generates precise answer
    ↓
Display to user
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Environment Variables
Create a `.env` file:
```
HF_TOKEN=your_huggingface_api_token
```
Get your token from: https://huggingface.co/settings/tokens

### Run Web Interface
```bash
streamlit run app.py
```

### Run CLI
```bash
# Add documents
python main.py --add path/to/documents/

# Ask a question
python main.py --query "What is Day 2 about?"
```

## 🛠️ Technologies Used

- **LangChain**: RAG framework
- **ChromaDB**: Vector database
- **HuggingFace**: 
  - Embeddings (sentence-transformers)
  - LLM (Mistral-7B-Instruct)
- **Streamlit**: Web UI
- **PyPDF**: PDF processing
- **Python 3.12+**

## 📊 Key Features

✅ **Upload PDF/TXT documents**  
✅ **Semantic search** (meaning-based, not keywords)  
✅ **AI-powered answers** using Mistral-7B LLM  
✅ **Persistent storage** (ChromaDB)  
✅ **Web UI** (Streamlit)  
✅ **CLI support** for automation  
✅ **Detailed logging**  
✅ **Configurable settings**  

## 🎯 Configuration Tips

### Adjust Chunk Size
In `config.py`:
- **Smaller chunks (300)**: More precise, but may lose context
- **Larger chunks (1000)**: More context, but less specific

### Adjust Retrieval Count (k)
- **k=1**: Fast, single best match
- **k=3**: Balanced (default)
- **k=5**: More comprehensive, but more tokens

### Change LLM Model
In `config.py`, change `HF_MODEL_FOR_QA`:
- `"mistral-community/Mistral-7B-Instruct-v0.1"` (default)
- `"meta-llama/Llama-2-7b-chat-hf"` (requires approval)
- `"google/flan-t5-large"` (smaller, faster)

## 📝 Troubleshooting

### Issue: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Issue: "HuggingFace API key not found"
- Create `.env` file
- Add: `HF_TOKEN=your_token_here`

### Issue: "ChromaDB not found"
- Upload documents first
- Check `vectorstore/chroma/` directory exists

## 🔮 Future Enhancements

- [ ] Support for more file types (DOCX, HTML)
- [ ] Multiple collections (different projects)
- [ ] Chat history persistence
- [ ] Source citation in answers
- [ ] Admin dashboard
- [ ] API endpoint
- [ ] Docker deployment

## � Deployment Guide

### Option 1: Streamlit Cloud (Recommended - FREE)
**Best for**: Quick sharing, demos

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/personal-knowledge-assistant.git
   git push -u origin main
   ```

2. **Create Streamlit Cloud Account**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app" → Select your repository
   - Set main file: `app.py`

3. **Add Secrets**
   - Dashboard → Advanced settings → Secrets
   - Paste:
     ```
     HF_TOKEN = "your_huggingface_token"
     ```

✅ **Your app is live!** Share the URL

---

### Option 2: Docker + Railway (FREE tier)
**Best for**: Production-ready, easy scaling

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt langchain-chroma
   COPY . .
   CMD ["streamlit", "run", "app.py", "--server.port=8501"]
   ```

2. **Deploy to Railway**
   - https://railway.app → New Project
   - Deploy from GitHub
   - Add `HF_TOKEN` environment variable
   - Done! ✅

---

### Option 3: Google Cloud Run (Pay-per-use)
```bash
gcloud run deploy knowledge-assistant \
  --source . \
  --set-env-vars HF_TOKEN=your_token
```

---

### Option 4: DigitalOcean VPS ($5-12/month)
- Traditional VPS deployment with Nginx reverse proxy
- Full control, easily scalable

---

## 🔒 Security Checklist Before GitHub

✅ **Already Done:**
- `.gitignore` excludes `.env` and sensitive files
- `.env.example` provided as template
- API keys loaded from environment only

⚠️ **Verify Before Pushing:**
```bash
# Make sure .env is NOT committed
git rm --cached .env 2>/dev/null || true

# Check for hardcoded secrets
grep -r "sk_\|password\|secret" . --include="*.py" || echo "✓ No hardcoded secrets found"
```

---

## 📊 Platform Comparison

| Platform | Cost | Setup | Best For |
|----------|------|-------|----------|
| Streamlit Cloud | FREE | 5 min | Demos |
| Railway | FREE tier | 10 min | Small projects |
| Cloud Run | Pay-per-use | 15 min | Serverless |
| DigitalOcean | $5/mo | 20 min | Full control |

---

## 📄 License

MIT License - feel free to use and modify!

---

**Built with ❤️ using LangChain, HuggingFace, and Streamlit**
