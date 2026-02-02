"""
Embeddings Manager - Use HuggingFace embeddings for vector search

This file shows how to use HuggingFaceEmbeddings from LangChain
to convert text into vectors for similarity search.
"""

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME
from src.utils import logger


# Create embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
logger.info(f"Loaded embeddings: {EMBEDDING_MODEL_NAME}")
