"""
LangChain Retrievers - Theory and Practice

==========================================
WHAT ARE RETRIEVERS?
==========================================

Retrievers are interfaces that return documents given an unstructured query. They are more 
general than vector stores - while every vector store can be turned into a retriever, not 
every retriever is a vector store.

Key Characteristics:
- Standard interface: get_relevant_documents(query) -> List[Document]
- Abstract away the implementation details of document retrieval
- Can be based on various retrieval strategies (semantic, keyword, hybrid)
- Essential component in RAG (Retrieval Augmented Generation) pipelines
- Chainable and composable with other LangChain components

Purpose:
Retrievers serve as the "memory" or "knowledge base" for LLM applications, enabling them 
to access external information beyond their training data.


==========================================
RETRIEVERS VS VECTOR STORES
==========================================

VECTOR STORES:
- Specific storage mechanism for embeddings
- Focus on similarity search using vector math
- Lower-level component
- Methods: similarity_search(), similarity_search_with_score()
- Can be converted to retrievers using .as_retriever()

RETRIEVERS:
- Abstract interface for document retrieval
- Can use various strategies (not just similarity)
- Higher-level abstraction
- Standard method: get_relevant_documents()
- Can wrap vector stores, search APIs, databases, etc.

Analogy: Vector store is like a library catalog system, while retriever is the librarian 
who knows different strategies to find the books you need.


==========================================
TYPES OF RETRIEVERS IN LANGCHAIN
==========================================

1. VECTOR STORE-BASED RETRIEVERS
   - VectorStoreRetriever: Direct wrapper around vector stores
   - Multi-Vector Retriever: Stores multiple vectors per document
   - Parent Document Retriever: Retrieves larger parent documents

2. SEARCH-BASED RETRIEVERS
   - BM25 Retriever: Classic keyword-based ranking algorithm
   - TF-IDF Retriever: Term frequency-inverse document frequency
   - SVM Retriever: Support Vector Machine-based retrieval

3. ADVANCED RETRIEVERS
   - Ensemble Retriever: Combines multiple retrievers
   - Contextual Compression Retriever: Compresses retrieved documents
   - Self-Query Retriever: Generates structured queries from natural language
   - Time-Weighted Retriever: Weights recent documents higher
   - Multi-Query Retriever: Generates multiple queries for better recall

4. EXTERNAL API RETRIEVERS
   - Wikipedia Retriever
   - Arxiv Retriever
   - PubMed Retriever
   - Web Search Retrievers (Google, Bing, DuckDuckGo)


==========================================
RETRIEVER PARAMETERS
==========================================

Common parameters when creating retrievers from vector stores:

1. search_type: 
   - "similarity" (default): Standard similarity search
   - "mmr": Maximal Marginal Relevance (diverse results)
   - "similarity_score_threshold": Filter by minimum similarity score

2. search_kwargs:
   - k: Number of documents to retrieve
   - fetch_k: Number of documents to fetch before filtering (for MMR)
   - lambda_mult: Diversity parameter for MMR (0=diverse, 1=similar)
   - score_threshold: Minimum similarity score (for threshold search)
   - filter: Metadata filters


==========================================
CODE EXAMPLES
==========================================
"""

from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
import numpy as np

# Note: Advanced retrievers commented out - may not be available in current version
# Uncomment and install if needed: pip install langchain langchain-experimental
# from langchain.retrievers.ensemble import EnsembleRetriever
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor

# Placeholder classes for examples (remove if above imports work)
EnsembleRetriever = None
ContextualCompressionRetriever = None
MultiQueryRetriever = None
LLMChainExtractor = None

# ==========================================
# Example 1: Basic Vector Store Retriever
# ==========================================

def example_1_basic_retriever():
    """Create a basic retriever from a vector store"""
    
    print("=" * 60)
    print("Example 1: Basic Vector Store Retriever")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning enables computers to learn from data without explicit programming.",
        "Neural networks are inspired by the human brain's structure.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses multiple layers to progressively extract higher-level features.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Computer vision enables machines to interpret visual information.",
        "Transfer learning applies knowledge from one task to another."
    ]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="basic_retriever"
    )
    
    # Convert to retriever (default: similarity search with k=4)
    retriever = vectorstore.as_retriever()
    
    # Retrieve documents
    query = "How do computers learn?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Example 2: Retriever with Custom Parameters
# ==========================================

def example_2_custom_parameters():
    """Configure retriever with custom search parameters"""
    
    print("\n" + "=" * 60)
    print("Example 2: Custom Retriever Parameters")
    print("=" * 60)
    
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "The Great Wall of China is one of the world's most famous landmarks.",
        "The Taj Mahal is a white marble mausoleum in India.",
        "The Colosseum is an ancient amphitheater in Rome, Italy.",
        "Machu Picchu is an Incan citadel in Peru.",
        "The Statue of Liberty stands in New York Harbor.",
        "The Pyramids of Giza are ancient monuments in Egypt.",
        "Christ the Redeemer overlooks Rio de Janeiro, Brazil."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="landmarks"
    )
    
    # Create retriever with custom parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Return top 3 results
    )
    
    query = "Tell me about monuments in Europe"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print(f"\nTop 3 relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Example 3: MMR (Maximal Marginal Relevance) Retriever
# ==========================================

def example_3_mmr_retriever():
    """Use MMR for diverse document retrieval"""
    
    print("\n" + "=" * 60)
    print("Example 3: MMR Retriever (Diverse Results)")
    print("=" * 60)
    
    documents = [
        "Python was created by Guido van Rossum in 1991.",
        "Python is known for its simple and readable syntax.",
        "Python is widely used in data science and machine learning.",
        "Python has extensive libraries for various tasks.",
        "Java is a statically-typed programming language.",
        "JavaScript is the language of the web.",
        "C++ offers high performance and low-level control.",
        "Rust provides memory safety without garbage collection."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embeddings
    )
    
    query = "Tell me about Python"
    
    # Standard similarity retriever
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # MMR retriever for diverse results
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 8,  # Fetch more candidates
            "lambda_mult": 0.5  # Balance between relevance and diversity
        }
    )
    
    print(f"\nQuery: {query}\n")
    
    print("--- Similarity Retriever Results ---")
    sim_docs = similarity_retriever.get_relevant_documents(query)
    for i, doc in enumerate(sim_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    print("\n--- MMR Retriever Results (More Diverse) ---")
    mmr_docs = mmr_retriever.get_relevant_documents(query)
    for i, doc in enumerate(mmr_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return mmr_retriever


# ==========================================
# Example 4: Similarity Score Threshold Retriever
# ==========================================

def example_4_score_threshold():
    """Filter documents by minimum similarity score"""
    
    print("\n" + "=" * 60)
    print("Example 4: Similarity Score Threshold")
    print("=" * 60)
    
    documents = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models can diagnose diseases.",
        "AI helps in drug discovery and development.",
        "Pizza is a popular Italian dish.",
        "The weather today is sunny and warm.",
        "Deep learning improves medical imaging analysis."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="threshold_test"
    )
    
    # Retriever with score threshold
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,  # Only return docs with similarity > 0.5
            "k": 10  # Max number to consider
        }
    )
    
    query = "AI in medical field"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print(f"\nDocuments above threshold (0.5):")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Example 5: BM25 Retriever (Keyword-Based)
# ==========================================

def example_5_bm25_retriever():
    """Traditional keyword-based retrieval using BM25 algorithm"""
    
    print("\n" + "=" * 60)
    print("Example 5: BM25 Retriever (Keyword Search)")
    print("=" * 60)
    
    # Create documents
    texts = [
        "Python is great for web development with Django and Flask.",
        "Machine learning models require large datasets for training.",
        "Natural language processing enables chatbots and virtual assistants.",
        "JavaScript frameworks like React and Vue are popular for frontend.",
        "Database management systems store and organize data efficiently.",
        "Cloud computing provides scalable infrastructure for applications."
    ]
    
    docs = [Document(page_content=text) for text in texts]
    
    # Create BM25 retriever (doesn't use embeddings)
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 3  # Return top 3 results
    
    query = "web development frameworks"
    results = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print("\nBM25 Results (keyword-based):")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Example 6: Ensemble Retriever (Hybrid Search)
# ==========================================

def example_6_ensemble_retriever():
    """Combine multiple retrievers for better results"""
    
    print("\n" + "=" * 60)
    print("Example 6: Ensemble Retriever (Hybrid Search)")
    print("=" * 60)
    
    if EnsembleRetriever is None:
        print("\nEnsembleRetriever not available. Install with:")
        print("pip install langchain")
        return None
    
    texts = [
        "Python is a versatile programming language used in many domains.",
        "Machine learning algorithms can predict outcomes from data.",
        "Web scraping extracts data from websites automatically.",
        "APIs enable communication between different software systems.",
        "Docker containers provide isolated environments for applications.",
        "Version control systems like Git track code changes over time."
    ]
    
    # Create documents for BM25
    docs = [Document(page_content=text) for text in texts]
    
    # BM25 Retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    
    # Semantic Retriever (embedding-based)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Ensemble Retriever (combines both)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]  # Equal weight to both
    )
    
    query = "How to extract information from web pages?"
    
    print(f"\nQuery: {query}\n")
    
    print("--- BM25 Only ---")
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    for i, doc in enumerate(bm25_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    print("\n--- Semantic Only ---")
    semantic_docs = semantic_retriever.get_relevant_documents(query)
    for i, doc in enumerate(semantic_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    print("\n--- Ensemble (Hybrid) ---")
    ensemble_docs = ensemble_retriever.get_relevant_documents(query)
    for i, doc in enumerate(ensemble_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return ensemble_retriever


# ==========================================
# Example 7: Retriever with Metadata Filtering
# ==========================================

def example_7_metadata_filtering():
    """Filter documents by metadata during retrieval"""
    
    print("\n" + "=" * 60)
    print("Example 7: Metadata Filtering")
    print("=" * 60)
    
    texts = [
        "Python 3.9 introduced new syntax features.",
        "Java 17 is the latest LTS release.",
        "JavaScript ES2021 added logical assignment operators.",
        "Python 3.10 added structural pattern matching.",
        "TypeScript 4.5 improved template string types.",
        "Python 3.11 improved performance significantly."
    ]
    
    metadatas = [
        {"language": "Python", "year": 2020, "version": "3.9"},
        {"language": "Java", "year": 2021, "version": "17"},
        {"language": "JavaScript", "year": 2021, "version": "ES2021"},
        {"language": "Python", "year": 2021, "version": "3.10"},
        {"language": "TypeScript", "year": 2021, "version": "4.5"},
        {"language": "Python", "year": 2022, "version": "3.11"}
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        collection_name="metadata_filter"
    )
    
    # Retriever with metadata filter
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"language": "Python"}  # Only Python documents
        }
    )
    
    query = "What are the new features?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print("\nPython documents only:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    return retriever


# ==========================================
# Example 8: Multi-Query Retriever
# ==========================================

def example_8_multi_query_retriever():
    """Generate multiple queries for better recall (requires LLM)"""
    
    print("\n" + "=" * 60)
    print("Example 8: Multi-Query Retriever")
    print("=" * 60)
    
    if MultiQueryRetriever is None:
        print("\nMultiQueryRetriever not available. Install with:")
        print("pip install langchain")
        return None
    
    texts = [
        "Climate change affects global temperatures and weather patterns.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "Deforestation contributes to greenhouse gas emissions.",
        "Electric vehicles reduce carbon footprint compared to gasoline cars.",
        "Recycling helps reduce waste and conserve natural resources.",
        "Ocean acidification threatens marine ecosystems."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    try:
        # Requires OpenAI API key
        llm = ChatOpenAI(temperature=0)
        
        # Multi-query retriever generates multiple perspectives
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        query = "How can we reduce environmental impact?"
        
        print(f"\nQuery: {query}")
        print("\nGenerating multiple query perspectives...")
        docs = multi_query_retriever.get_relevant_documents(query)
        
        print(f"\nRetrieved {len(docs)} unique documents:")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content}")
        
        return multi_query_retriever
        
    except Exception as e:
        print(f"\nMulti-Query Retriever requires OpenAI API key: {e}")
        print("Using base retriever instead...")
        docs = base_retriever.get_relevant_documents(
            "How can we reduce environmental impact?"
        )
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content}")
        return base_retriever


# ==========================================
# Example 9: Contextual Compression Retriever
# ==========================================

def example_9_contextual_compression():
    """Compress retrieved documents to relevant parts only"""
    
    print("\n" + "=" * 60)
    print("Example 9: Contextual Compression Retriever")
    print("=" * 60)
    
    if ContextualCompressionRetriever is None or LLMChainExtractor is None:
        print("\nContextual Compression components not available. Install with:")
        print("pip install langchain")
        return None
    
    texts = [
        """Python is a high-level, interpreted programming language. Created by Guido van Rossum 
        and first released in 1991, Python emphasizes code readability with its notable use of 
        significant whitespace. It supports multiple programming paradigms including procedural, 
        object-oriented, and functional programming.""",
        
        """JavaScript is a programming language that is one of the core technologies of the World 
        Wide Web. It enables interactive web pages and is an essential part of web applications. 
        The vast majority of websites use it for client-side page behavior.""",
        
        """Machine learning is a branch of artificial intelligence based on the idea that systems 
        can learn from data, identify patterns and make decisions with minimal human intervention. 
        It is used in various applications from email filtering to computer vision."""
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    query = "What is Python used for?"
    
    print(f"\nQuery: {query}\n")
    print("--- Without Compression ---")
    docs = base_retriever.get_relevant_documents(query)
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:150]}...")
    
    try:
        # Requires OpenAI API key
        llm = ChatOpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        print("\n--- With Compression (relevant parts only) ---")
        compressed_docs = compression_retriever.get_relevant_documents(query)
        for i, doc in enumerate(compressed_docs, 1):
            print(f"{i}. {doc.page_content}")
        
        return compression_retriever
        
    except Exception as e:
        print(f"\nContextual Compression requires OpenAI API key: {e}")
        return base_retriever


# ==========================================
# Example 10: Custom Retriever Class
# ==========================================

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class SimpleCustomRetriever(BaseRetriever):
    """Custom retriever that returns documents based on simple keyword matching"""
    
    documents: List[Document]
    k: int = 4
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents containing any query keywords"""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            score = sum(word in content for word in query_words)
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:self.k]]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (optional)"""
        return self._get_relevant_documents(query)


def example_10_custom_retriever():
    """Create and use a custom retriever"""
    
    print("\n" + "=" * 60)
    print("Example 10: Custom Retriever Implementation")
    print("=" * 60)
    
    docs = [
        Document(page_content="Python is great for data science and machine learning."),
        Document(page_content="JavaScript is used for web development."),
        Document(page_content="Data visualization helps understand complex datasets."),
        Document(page_content="Machine learning models require training data."),
        Document(page_content="Web scraping extracts data from websites.")
    ]
    
    # Create custom retriever
    retriever = SimpleCustomRetriever(documents=docs, k=3)
    
    query = "data machine learning"
    results = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print("\nCustom Retriever Results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LANGCHAIN RETRIEVERS - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    try:
        example_1_basic_retriever()
        example_2_custom_parameters()
        example_3_mmr_retriever()
        example_4_score_threshold()
        example_5_bm25_retriever()
        example_6_ensemble_retriever()
        example_7_metadata_filtering()
        example_8_multi_query_retriever()  # May need OpenAI API key
        example_9_contextual_compression()  # May need OpenAI API key
        example_10_custom_retriever()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to install required packages:")
        print("pip install langchain langchain-community chromadb faiss-cpu")
        print("pip install sentence-transformers rank-bm25")


"""
==========================================
KEY TAKEAWAYS
==========================================

1. Retrievers provide a standard interface for document retrieval
2. They can be based on various strategies (semantic, keyword, hybrid)
3. Vector store retrievers are most common but not the only option
4. MMR provides diverse results, avoiding redundancy
5. Ensemble retrievers combine multiple strategies for better results
6. Metadata filtering narrows down search space
7. Contextual compression reduces token usage by extracting relevant parts
8. Custom retrievers can implement domain-specific logic

==========================================
BEST PRACTICES
==========================================

1. Use similarity search for semantic understanding
2. Use BM25/keyword search for exact term matching
3. Combine both in ensemble retriever for hybrid search
4. Set appropriate k value (typically 3-5 for QA, higher for exploration)
5. Use MMR when you want diverse results
6. Apply metadata filters to reduce search space
7. Use score thresholds to filter low-quality matches
8. Consider contextual compression for long documents

==========================================
COMMON USE CASES
==========================================

1. Question Answering: Retrieve relevant context for LLM
2. Semantic Search: Find similar documents
3. RAG Systems: Augment generation with retrieved knowledge
4. Chatbots with Memory: Retrieve relevant conversation history
5. Document Analysis: Find related documents for analysis
6. Recommendation Systems: Retrieve similar items

==========================================
REQUIRED PACKAGES
==========================================

pip install langchain langchain-community langchain-openai
pip install chromadb faiss-cpu sentence-transformers
pip install rank-bm25 scikit-learn
pip install langchain-text-splitters

Note: Some examples require OpenAI API key (Multi-Query, Contextual Compression)
"""
