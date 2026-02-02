"""
Document Processor - Load and chunk documents for RAG
"""

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import logger, clean_text


def process_documents(directory: Path) -> List[Document]:
    """Load and chunk documents from directory."""
    
    directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    logger.info(f"Processing documents from: {directory}")
    
    # Load all PDFs and TXTs using LangChain's DirectoryLoader
    all_docs = []
    
    try:
        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(directory),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        logger.warning(f"Could not load PDFs: {e}")
    
    try:
        # Load TXTs
        txt_loader = DirectoryLoader(
            str(directory),
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        all_docs.extend(txt_loader.load())
    except Exception as e:
        logger.warning(f"Could not load TXTs: {e}")
    
    if not all_docs:
        logger.warning("No documents found")
        return []
    
    logger.info(f"Loaded {len(all_docs)} documents")
    
    # Filter out empty documents
    non_empty_docs = [doc for doc in all_docs if doc.page_content.strip()]
    if len(non_empty_docs) < len(all_docs):
        logger.warning(f"Skipped {len(all_docs) - len(non_empty_docs)} empty documents")
    
    if not non_empty_docs:
        logger.error("All loaded documents are empty - cannot create chunks")
        return []
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(non_empty_docs)
    
    # Clean text
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    
    logger.info(f"✓ Created {len(chunks)} chunks")
    return chunks
