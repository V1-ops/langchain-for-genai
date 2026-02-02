"""
RAG Chain - Retrieve documents and generate answers using LLM API
"""

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    VECTORSTORE_DIR, 
    HF_API_KEY, 
    HF_MODEL_FOR_QA,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    RETRIEVAL_K,
    RETRIEVAL_METHOD,
)
from src.embeddings_manager import embeddings
from src.utils import logger


CHROMA_PATH = VECTORSTORE_DIR / "chroma"


def create_rag_chain():
    """Create RAG chain with LLM API for intelligent answers."""
    
    # Validate API key
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not set. Add HF_TOKEN to your .env file.")
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )
    
    # Check if vector store has documents
    collection = vector_store._collection
    doc_count = collection.count()
    if doc_count == 0:
        raise ValueError("No documents in vector store. Please upload documents first.")
    
    # Configure retriever with MMR or similarity search
    if RETRIEVAL_METHOD == "mmr":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_K * 2}
        )
    else:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
    
    # Initialize HuggingFace Endpoint directly (no wrapper)
    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_FOR_QA,
        huggingfacehub_api_token=HF_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS
    )
    
    # Create prompt template for RAG
    prompt = PromptTemplate.from_template("""Answer the question based ONLY on the context provided.

Context:
{context}

Question: {question}

Answer:""")
    
    # Format documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create RAG chain
    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | prompt
        | llm
    )
    
    logger.info(f"✓ RAG chain created with {doc_count} documents")
    return rag_chain

def answer_question(question: str):
    """Ask a question and get answer.
    
    Args:
        question: The question to answer
    """
    
    try:
        logger.info(f"Creating RAG chain...")
        rag_chain = create_rag_chain()
        
        logger.info(f"Invoking with question: {question}")
        answer = rag_chain.invoke(question)
        
        logger.info(f"Answer type: {type(answer)}, value: {answer}")
        
        # Handle different response types
        if answer is None:
            return "No answer generated. Please try again."
        
        if hasattr(answer, 'content'):
            logger.info("Extracting from content attribute")
            result = answer.content
        elif isinstance(answer, str):
            result = answer
        else:
            result = str(answer)
        
        # Clean up result
        result = result.strip() if result else "No answer generated"
        
        if not result or result.lower() == "answer:":
            return "Unable to generate a valid answer. The model may not have sufficient context."
        
        logger.info(f"Answer generated: {result[:100]}...")
        return result
        
    except StopIteration:
        logger.error("StopIteration: LLM returned empty response")
        return "Error: LLM returned an empty response. Try a different question or check the documents."
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        tb = traceback.format_exc()
        logger.error(f"Error generating answer: {error_msg}\n{tb}")
        return f"Error: {error_msg if error_msg else 'Unknown error - check logs'}"
