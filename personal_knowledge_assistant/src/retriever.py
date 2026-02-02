"""
Retriever - Use Chroma vector store to store and retrieve documents
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma

from config import VECTORSTORE_DIR
from src.embeddings_manager import embeddings


# Chroma persists to this directory
CHROMA_PATH = VECTORSTORE_DIR / "chroma"
