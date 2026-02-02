"""
Configuration Settings for Personal Knowledge Base Assistant

This file centralizes all configuration:
- Paths (where files are stored)
- Model settings (embedding model, LLM)
- Database settings (vector store configuration)
- Processing settings (chunk size, overlap)
- API keys (loaded from .env)

Benefits:
- Easy to modify settings without touching code
- Different configs for dev/prod
- All settings in one place (DRY principle)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# LOAD ENVIRONMENT VARIABLES FROM .env FILE
# =============================================================================
# This loads your HuggingFace API key from .env
# Make sure your .env file has: HUGGINGFACE_API_KEY=your_key_here

load_dotenv()

# =============================================================================
# PROJECT PATHS
# =============================================================================
# Define where files are stored - makes it easy to find things

PROJECT_ROOT = Path(__file__).parent
"""Root directory of the project"""

DATA_DIR = PROJECT_ROOT / "data"
"""Where to store documents"""

DATA_RAW_DIR = DATA_DIR / "raw"
"""Where original documents go"""

DATA_PROCESSED_DIR = DATA_DIR / "processed"
"""Where processed/chunked documents go (optional cache)"""

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
"""Where ChromaDB stores vector embeddings"""

CHROMA_DB_PATH = VECTORSTORE_DIR / "chroma_db"
"""Full path to ChromaDB directory"""

LOGS_DIR = PROJECT_ROOT / "logs"
"""Where to store application logs"""

# Create directories if they don't exist
for directory in [DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, VECTORSTORE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# Settings for converting text to vectors

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""
Which embedding model to use.

Options:
1. "sentence-transformers/all-MiniLM-L6-v2" (RECOMMENDED)
   - Fast, lightweight (22MB)
   - Good quality for general use
   - Works offline
   
2. "sentence-transformers/all-mpnet-base-v2"
   - Better quality but slower
   - 420MB download
   - Better for semantic search
   
3. "BAAI/bge-large-en-v1.5"
   - State-of-the-art (uses Alibaba's BGE)
   - Large model, needs GPU for speed
   - Best quality

For this project, all-MiniLM-L6-v2 is perfect - fast & good quality.
"""

EMBEDDING_DEVICE = "cpu"
"""
Which device to use for embeddings.
Options: "cpu" or "cuda" (if you have GPU)

Note: HuggingFace embeddings run locally, not via API
"""

EMBEDDING_BATCH_SIZE = 32
"""
How many documents to embed at once.
Larger = faster but uses more memory.
If you get memory errors, reduce this.
"""

# =============================================================================
# VECTOR STORE CONFIGURATION (CHROMADB)
# =============================================================================
# Settings for storing and searching vectors

VECTORSTORE_TYPE = "chroma"
"""Type of vector store: 'chroma' or 'faiss'"""

CHROMA_COLLECTION_NAME = "documents"
"""Name of the Chroma collection to store documents"""

CHROMA_PERSIST_DIRECTORY = str(CHROMA_DB_PATH)
"""Where ChromaDB saves its data persistently"""

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION
# =============================================================================
# Settings for splitting documents into chunks

CHUNK_SIZE = 500
"""
How many characters per chunk.

Why 500?
- Too small (100): Too many chunks, loses context
- Just right (500): Good balance of context & specificity
- Too large (2000): Too much text, less relevant results

Rule of thumb: 300-1000 depending on your content
"""

CHUNK_OVERLAP = 50
"""
How many characters to overlap between chunks.

Example:
Chunk 1: [0-500]
Chunk 2: [450-950] <- 50 character overlap
Chunk 3: [900-1400]

Why overlap?
- Prevents important info from being at chunk boundaries
- Ensures context continuity
- Typical range: 20-100 characters
"""

TEXT_SPLITTER_TYPE = "recursive"
"""
How to split documents.

Options:
1. "recursive" (RECOMMENDED)
   - Splits by sentences first, then paragraphs
   - Preserves context
   - Better for general text
   
2. "character"
   - Simple character-based split
   - Less intelligent
   
3. "token"
   - Splits by tokens (if using specific LLM)
   - Useful for API limits
"""

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
# Settings for searching and retrieving documents

RETRIEVAL_K = 3
"""
How many documents to retrieve for each query.

Why 3?
- Too few (1): Might miss relevant info
- Just right (3): Good balance of relevance & brevity
- Too many (10): Overwhelms LLM with too much text

Adjust based on your needs:
- Quick answers: k=1 or k=2
- Comprehensive: k=5 or k=6
"""

RETRIEVAL_METHOD = "similarity"
"""
How to retrieve documents.

Options:
1. "similarity" (RECOMMENDED)
   - Find most similar chunks
   - Fast and effective
   
2. "mmr" (Maximal Marginal Relevance)
   - Balances relevance + diversity
   - Avoids redundant results
   - Slower but better quality
"""

RETRIEVAL_SCORE_THRESHOLD = 0.5
"""
Minimum similarity score to include a result (0.0 to 1.0).
- 0.5: Include any reasonably relevant results
- 0.7: Only highly relevant results
- 0.9: Only extremely relevant results

Lower = more permissive, Higher = stricter
"""

# =============================================================================
# LLM CONFIGURATION - CHAT HUGGINGFACE WITH HUGGINGFACE ENDPOINT
# =============================================================================
# Using ChatHuggingFace with HuggingFace Inference Endpoint for LLM

# Note: We're NOT using OpenAI API (too expensive)
# Instead, we'll use HuggingFace ChatHuggingFace with Inference API for free

USE_LOCAL_LLM = False
"""
If True: Use local LLM (free but slower)
If False: Skip LLM integration (just retrieval)

Options if True:
- "ollama": Requires Ollama installed locally
- "huggingface": Use HuggingFace Inference API with ChatHuggingFace
"""

HF_API_KEY = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
"""
Your HuggingFace API key from .env file.
Get it from: https://huggingface.co/settings/tokens
Can use either HF_TOKEN or HUGGINGFACE_API_KEY in .env

This is required for ChatHuggingFace to work with HuggingFace Endpoint.
"""

if not HF_API_KEY:
    print("⚠️  WARNING: HF_TOKEN not found in .env file!")
    print("   You can still use retrieval without it.")
    print("   For LLM features with ChatHuggingFace, add: HF_TOKEN=your_key_here to .env file")

HF_MODEL_FOR_QA = "Qwen/Qwen2.5-72B-Instruct"
"""
Which HuggingFace model to use for question answering with ChatHuggingFace.

Using Qwen2.5-72B-Instruct for excellent conversational abilities (2026)
Alternatives:
- mistralai/Mistral-7B-Instruct-v0.2 (lighter, faster)
- google/flan-t5-xxl (good quality)
"""

HF_ENDPOINT_URL = "https://api-inference.huggingface.co/models"
"""
HuggingFace Inference Endpoint URL.
Used by ChatHuggingFace for API communication.
Default: https://api-inference.huggingface.co/models
"""

# LLM PARAMETERS
LLM_TEMPERATURE = 0.3
"""
Temperature for LLM generation (0.0 to 1.0).
- 0.0: Deterministic, always same output
- 0.3: Lower (more focused, less random) - Good for Q&A
- 0.7: Higher (more creative, more random) - Good for brainstorming
"""

LLM_MAX_NEW_TOKENS = 256
"""
Maximum tokens in LLM response.
Lower = faster responses, Higher = more detailed answers
Good range: 256-512 for Q&A
"""

# =============================================================================
# RAG CHAIN CONFIGURATION
# =============================================================================
# Settings for the complete RAG pipeline

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. Only answer using information from the provided documents
2. If the answer is not in the documents, say "I couldn't find this information in the documents"
3. Always cite your sources (which document the info came from)
4. Be concise but comprehensive
5. If multiple documents mention the same thing, mention all of them"""
# The system prompt tells the LLM how to behave.
# Customize this for your use case!
# Examples:
# - For student notes: "You are a tutor answering based on lecture notes"
# - For research: "You are a research assistant..."
# - For legal docs: "You are a legal assistant..."

QUESTION_TEMPLATE = """Based on the following documents:

{context}

Answer this question: {question}

Provide a clear, concise answer with citations."""
# Template for how questions are formatted when sent to LLM.
# {context} = retrieved documents
# {question} = user's question

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Settings for logging (debugging and tracking)

LOG_LEVEL = "INFO"
"""
Logging level.
Options: DEBUG (most verbose), INFO, WARNING, ERROR, CRITICAL (least)
"""

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""Format for log messages"""

LOG_FILE = LOGS_DIR / "assistant.log"
"""Where to save log file"""

# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================
# Which document formats we can process

SUPPORTED_FILE_TYPES = {
    ".pdf": "PDF documents",
    ".txt": "Text files",
    ".docx": "Word documents",
    ".md": "Markdown files",
}

"""
Supported file formats for upload.
You can add more here if needed.
"""

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================
# Settings for the web interface

STREAMLIT_PAGE_TITLE = "Personal Knowledge Base Assistant"
STREAMLIT_PAGE_ICON = "📚"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

"""Streamlit app settings - controls how the web UI looks"""

# =============================================================================
# DEVELOPMENT VS PRODUCTION
# =============================================================================
# Different settings for development and production

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
"""Current environment: 'development' or 'production'"""

DEBUG_MODE = ENVIRONMENT == "development"
"""Enable debug mode (more verbose output, etc.)"""

if DEBUG_MODE:
    print("\n" + "="*60)
    print("🔧 DEVELOPMENT MODE ACTIVATED")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_RAW_DIR}")
    print(f"Vector Store: {CHROMA_DB_PATH}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"HF API Key: {'✓ Found' if HF_API_KEY else '✗ Not found'}")
    print("="*60 + "\n")

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

CACHE_EMBEDDINGS = True
"""Cache embeddings to avoid recomputing. Saves time & resources."""

USE_GPU_FOR_EMBEDDINGS = False
"""Use GPU if available (faster but uses more VRAM). Set True if you have GPU."""

# =============================================================================
# OPTIONAL: ADVANCED RETRIEVAL SETTINGS
# =============================================================================
# These are for advanced use cases - you don't need to change these now

ENABLE_METADATA_FILTERING = True
"""Allow filtering by metadata (e.g., filename, date)"""

ENABLE_RERANKING = False
"""Use reranking for better results (slower but more accurate)"""

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
"""Reranker model for improved relevance (if ENABLE_RERANKING=True)"""

# =============================================================================
# SUMMARY: KEY THINGS TO MODIFY
# =============================================================================
"""
When you use this project, these are the most common changes:

1. CHUNK_SIZE: Increase for longer documents, decrease for shorter
2. RETRIEVAL_K: More results = more thorough but slower
3. EMBEDDING_MODEL_NAME: Different model = different quality
4. SYSTEM_PROMPT: Customize how LLM responds
5. Supported file types: Add more formats if needed
"""
