"""
Retriever - Use Chroma vector store to store and retrieve documents
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VECTORSTORE_DIR
from src.embeddings_manager import embeddings


# Chroma persists to this directory
CHROMA_PATH = VECTORSTORE_DIR / "chroma"
