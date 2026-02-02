"""
RAG Chain - Retrieve documents and generate answers using LLM API
"""

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import signal
from functools import wraps
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    VECTORSTORE_DIR, 
    HF_API_KEY, 
    HF_MODEL_FOR_QA,
    HF_ENDPOINT_URL,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    RETRIEVAL_K,
    RETRIEVAL_METHOD,
    ENABLE_RERANKING,
    RERANKER_MODEL
)
from src.embeddings_manager import embeddings
from src.utils import logger


CHROMA_PATH = VECTORSTORE_DIR / "chroma"


def create_rag_chain():
    """Create RAG chain with LLM API for intelligent answers."""
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
    
    # Configure retriever with MMR or similarity search
    search_kwargs = {"k": RETRIEVAL_K}
    if RETRIEVAL_METHOD == "mmr":
        search_kwargs["search_type"] = "mmr"
        search_kwargs["fetch_k"] = RETRIEVAL_K * 2  # Fetch more candidates for MMR
    
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # Initialize HuggingFace Endpoint
    try:
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL_FOR_QA,
            huggingfacehub_api_token=HF_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            task="conversational"
        )
        logger.info(f"✓ HuggingFace Endpoint initialized: {HF_MODEL_FOR_QA}")
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace Endpoint: {e}")
        return None
    
    # Wrap with ChatHuggingFace for better chat handling
    try:
        llm = ChatHuggingFace(
            llm=endpoint,
            temperature=LLM_TEMPERATURE
        )
        logger.info(f"✓ ChatHuggingFace LLM loaded: {HF_MODEL_FOR_QA}")
    except Exception as e:
        logger.error(f"Failed to load ChatHuggingFace: {e}")
        return None
    
    # Create prompt template for RAG
    prompt = ChatPromptTemplate.from_template("""Answer the question based ONLY on the context provided. Be precise and extract specific information.

Context:
{context}

Question: {question}

Answer:""")
    
    # Format documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Apply reranking if enabled
    context_chain = retriever | RunnableLambda(format_docs)
    if ENABLE_RERANKING:
        try:
            from langchain_community.document_compressors import CrossEncoderReranker
            from langchain_community.retrievers import ContextualCompressionRetriever
            
            # Add reranker for better relevance filtering
            compressor = CrossEncoderReranker(model_name=RERANKER_MODEL)
            context_chain = (
                retriever 
                | RunnableLambda(lambda docs: ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever
                ).compress_documents(docs, ""))
                | RunnableLambda(format_docs)
            )
            logger.info(f"✓ Reranking enabled with {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. Using standard retrieval.")
    
    # Create RAG chain
    rag_chain = (
        RunnableParallel(
            context=context_chain,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
    )
    
    logger.info("✓ RAG chain created with LLM API")
    return rag_chain


def answer_question(question: str, timeout: int = 60):
    """Ask a question and get answer with sources.
    
    Args:
        question: The question to answer
        timeout: Timeout in seconds (default 60)
    """
    
    rag_chain = create_rag_chain()
    
    if rag_chain is None:
        logger.error("Could not create RAG chain")
        return "Error: Could not create RAG chain. Check your HuggingFace API key."
    
    logger.info(f"Question: {question}")
    
    try:
        # Invoke with timeout handling
        answer = rag_chain.invoke(question)
        
        if answer:
            # Extract content if answer is an object with .content attribute
            if hasattr(answer, 'content'):
                answer = answer.content
            logger.info(f"Answer generated successfully")
            return answer
        else:
            logger.warning("Answer is empty")
            return "No answer generated. Please try rephrasing your question."
            
    except TimeoutError:
        logger.error(f"Answer generation timed out after {timeout}s")
        return f"⏱️ Request timed out. The model took too long to respond. Please try again or ask a simpler question."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error: {str(e)}"
