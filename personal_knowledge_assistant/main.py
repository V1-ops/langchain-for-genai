"""
Main CLI - Command-line interface for RAG system
"""

import argparse
from pathlib import Path

from src.document_processor import process_documents
from src.retriever import CHROMA_PATH
from src.rag_chain import answer_question
from langchain_community.vectorstores import Chroma
from src.embeddings_manager import embeddings
from src.utils import logger


def add_documents(docs_path: str):
    """Add documents to vector store."""
    
    docs_path = Path(docs_path)
    
    logger.info(f"Processing documents from: {docs_path}")
    
    # Load and chunk documents
    chunks = process_documents(docs_path)
    
    if not chunks:
        logger.error("No documents processed")
        return
    
    # Store in Chroma
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )
    
    logger.info(f"✓ Stored {len(chunks)} chunks in vector store")


def query(question: str):
    """Ask a question."""
    
    if not CHROMA_PATH.exists():
        logger.error("Vector store not found. Add documents first with --add")
        return
    
    logger.info(f"Question: {question}")
    
    answer = answer_question(question)
    
    print(f"\nAnswer: {answer}\n")


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Base Assistant"
    )
    
    parser.add_argument(
        "--add",
        type=str,
        help="Path to documents directory"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask"
    )
    
    args = parser.parse_args()
    
    if args.add:
        add_documents(args.add)
    elif args.query:
        query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
