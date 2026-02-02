"""
Vector Stores in LangChain - Theory and Practice

==========================================
WHAT ARE VECTOR STORES?
==========================================

Vector stores are specialized storage systems designed to store and retrieve high-dimensional 
vector embeddings efficiently. They are crucial components in modern AI applications, 
particularly for semantic search, recommendation systems, and retrieval-augmented generation (RAG).

Key Characteristics:
- Store numerical vector representations (embeddings) of data
- Enable similarity search based on vector distance metrics (cosine, euclidean, etc.)
- Optimize for fast nearest neighbor searches in high-dimensional spaces
- Support metadata filtering alongside vector similarity

Use Cases:
1. Semantic Search: Find documents similar in meaning, not just keywords
2. Question Answering: Retrieve relevant context for answering questions
3. Recommendation Systems: Find similar items based on embeddings
4. Memory for AI Agents: Store and retrieve conversation history efficiently


==========================================
VECTOR STORES VS VECTOR DATABASES
==========================================

VECTOR STORES:
--------------
- Lightweight, embedded solutions (e.g., Chroma, FAISS)
- Often run in-memory or with minimal setup
- Great for prototyping and small-to-medium scale applications
- Limited querying capabilities beyond similarity search
- Easy to integrate directly into applications
- Examples: Chroma, FAISS, Annoy

Pros:
✓ Quick setup and easy integration
✓ Low overhead for development
✓ No separate server required
✓ Good for local development

Cons:
✗ Limited scalability
✗ Basic querying features
✗ No built-in distributed architecture
✗ Limited persistence options


VECTOR DATABASES:
-----------------
- Full-featured database systems (e.g., Pinecone, Weaviate, Qdrant)
- Production-ready with enterprise features
- Built for scale with distributed architecture
- Advanced filtering, hybrid search, and analytics
- CRUD operations with ACID compliance (in some)
- Multi-tenancy and access control

Pros:
✓ Highly scalable (millions/billions of vectors)
✓ Production-ready with HA and fault tolerance
✓ Advanced querying and filtering
✓ Better performance at scale
✓ Built-in monitoring and analytics

Cons:
✗ More complex setup
✗ Requires infrastructure/hosting
✗ Higher operational costs
✗ Steeper learning curve


WHEN TO USE WHAT:
-----------------
Use Vector Stores when:
- Prototyping or building MVPs
- Small to medium datasets (< 1M vectors)
- Running locally or in single-server environments
- Simple similarity search is sufficient

Use Vector Databases when:
- Production applications at scale
- Large datasets (> 1M vectors)
- Need advanced filtering and hybrid search
- Require high availability and fault tolerance
- Multi-user or multi-tenant applications


==========================================
CHROMA DB - PRACTICAL EXAMPLES
==========================================
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ==========================================
# Example 1: Basic Chroma Setup with Local Embeddings
# ==========================================

def example_1_basic_chroma():
    """Create a simple vector store with Chroma using HuggingFace embeddings"""
    
    print("=" * 60)
    print("Example 1: Basic Chroma Vector Store")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Vector databases store high-dimensional embeddings.",
        "LangChain makes it easy to build LLM applications."
    ]
    
    # Create embeddings model (using free HuggingFace model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create Chroma vector store from texts
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="basic_collection"
    )
    
    # Perform similarity search
    query = "What is machine learning?"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print("\nTop 2 similar documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
    
    return vectorstore


# ==========================================
# Example 2: Persistent Chroma with Metadata
# ==========================================

def example_2_persistent_chroma():
    """Create a persistent Chroma vector store with metadata filtering"""
    
    print("\n" + "=" * 60)
    print("Example 2: Persistent Chroma with Metadata")
    print("=" * 60)
    
    # Documents with metadata
    texts = [
        "Python is great for machine learning.",
        "JavaScript is used for web development.",
        "Java is popular in enterprise applications.",
        "Rust offers memory safety without garbage collection.",
        "Go is excellent for building microservices."
    ]
    
    metadatas = [
        {"language": "Python", "domain": "ML", "year": 2023},
        {"language": "JavaScript", "domain": "Web", "year": 2023},
        {"language": "Java", "domain": "Enterprise", "year": 2023},
        {"language": "Rust", "domain": "Systems", "year": 2023},
        {"language": "Go", "domain": "Backend", "year": 2023}
    ]
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create persistent vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db",  # Persists to disk
        collection_name="programming_languages"
    )
    
    # Search with metadata filter
    print("\nSearch: Languages for ML")
    results = vectorstore.similarity_search(
        "machine learning",
        k=2,
        filter={"domain": "ML"}
    )
    
    for doc in results:
        print(f"- {doc.page_content} | Metadata: {doc.metadata}")
    
    return vectorstore


# ==========================================
# Example 3: Chroma with Documents and Text Splitting
# ==========================================

def example_3_chroma_with_documents():
    """Load documents, split them, and store in Chroma"""
    
    print("\n" + "=" * 60)
    print("Example 3: Chroma with Document Loading")
    print("=" * 60)
    
    # Sample long text
    long_text = """
    Artificial Intelligence (AI) is revolutionizing various industries. 
    Machine learning, a subset of AI, enables computers to learn from data.
    Deep learning uses neural networks with multiple layers to process information.
    Natural Language Processing (NLP) helps machines understand human language.
    Computer vision allows AI to interpret and understand visual information.
    Reinforcement learning trains agents to make decisions through trial and error.
    """
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separator="\n"
    )
    
    chunks = text_splitter.split_text(long_text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store from chunks
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="ai_documents"
    )
    
    # Similarity search with scores
    query = "How do machines understand language?"
    results = vectorstore.similarity_search_with_score(query, k=2)
    
    print(f"\nQuery: {query}")
    print("\nResults with similarity scores:")
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content.strip()}\n")
    
    return vectorstore


# ==========================================
# Example 4: Chroma as Retriever for RAG
# ==========================================

def example_4_chroma_as_retriever():
    """Use Chroma as a retriever for RAG applications"""
    
    print("\n" + "=" * 60)
    print("Example 4: Chroma as Retriever for RAG")
    print("=" * 60)
    
    # Knowledge base documents
    knowledge_base = [
        "LangChain is a framework for developing applications powered by language models.",
        "Vector stores enable efficient similarity search over embeddings.",
        "RAG (Retrieval Augmented Generation) combines retrieval with generation.",
        "Chroma is an open-source embedding database.",
        "Embeddings are numerical representations of text in vector space."
    ]
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_texts(
        texts=knowledge_base,
        embedding=embeddings,
        collection_name="rag_knowledge_base"
    )
    
    # Convert to retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Retrieve relevant documents
    query = "What is RAG?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nQuery: {query}")
    print("\nRetrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")
    
    return retriever


# ==========================================
# Example 5: Advanced Chroma - MMR Search
# ==========================================

def example_5_mmr_search():
    """Maximal Marginal Relevance for diverse results"""
    
    print("\n" + "=" * 60)
    print("Example 5: MMR Search for Diverse Results")
    print("=" * 60)
    
    documents = [
        "Python is a high-level programming language.",
        "Python is widely used in data science and AI.",
        "Python has a simple and readable syntax.",
        "Java is a statically-typed programming language.",
        "JavaScript runs in web browsers.",
        "C++ offers high performance for system programming."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="mmr_collection"
    )
    
    query = "Tell me about Python"
    
    # Standard similarity search (may return similar results)
    print("\n--- Standard Similarity Search ---")
    sim_results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(sim_results, 1):
        print(f"{i}. {doc.page_content}")
    
    # MMR search (returns diverse results)
    print("\n--- MMR Search (More Diverse) ---")
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=6  # Fetch more candidates for diversity
    )
    for i, doc in enumerate(mmr_results, 1):
        print(f"{i}. {doc.page_content}")
    
    return vectorstore


# ==========================================
# Example 6: Updating and Deleting from Chroma
# ==========================================

def example_6_crud_operations():
    """Demonstrate CRUD operations with Chroma"""
    
    print("\n" + "=" * 60)
    print("Example 6: CRUD Operations in Chroma")
    print("=" * 60)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = Chroma(
        collection_name="crud_collection",
        embedding_function=embeddings
    )
    
    # Add documents
    texts = ["First document", "Second document", "Third document"]
    ids = ["id1", "id2", "id3"]
    
    vectorstore.add_texts(texts=texts, ids=ids)
    print(f"\nAdded {len(texts)} documents")
    
    # Search to verify
    results = vectorstore.similarity_search("document", k=5)
    print(f"Total documents: {len(results)}")
    
    # Delete a document
    vectorstore.delete(ids=["id2"])
    print("\nDeleted document with id 'id2'")
    
    # Search again
    results = vectorstore.similarity_search("document", k=5)
    print(f"Total documents after deletion: {len(results)}")
    for doc in results:
        print(f"- {doc.page_content}")
    
    return vectorstore


# ==========================================
# Example 7: Using OpenAI Embeddings (if available)
# ==========================================

def example_7_openai_embeddings():
    """Example with OpenAI embeddings (requires API key)"""
    
    print("\n" + "=" * 60)
    print("Example 7: Chroma with OpenAI Embeddings")
    print("=" * 60)
    
    try:
        # Requires OPENAI_API_KEY environment variable
        embeddings = OpenAIEmbeddings()
        
        documents = [
            "OpenAI's GPT models are powerful language models.",
            "Vector embeddings capture semantic meaning.",
            "Chroma supports various embedding providers."
        ]
        
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=embeddings,
            collection_name="openai_collection"
        )
        
        query = "What are language models?"
        results = vectorstore.similarity_search(query, k=2)
        
        print(f"\nQuery: {query}")
        print("\nResults:")
        for doc in results:
            print(f"- {doc.page_content}")
        
        return vectorstore
        
    except Exception as e:
        print(f"\nOpenAI embeddings not available: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")
        return None


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CHROMA DB EXAMPLES - LANGCHAIN")
    print("=" * 60)
    
    # Run examples
    try:
        example_1_basic_chroma()
        example_2_persistent_chroma()
        example_3_chroma_with_documents()
        example_4_chroma_as_retriever()
        example_5_mmr_search()
        example_6_crud_operations()
        # example_7_openai_embeddings()  # Uncomment if you have OpenAI API key
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure to install required packages:")
        print("pip install langchain chromadb sentence-transformers")


"""
==========================================
KEY TAKEAWAYS
==========================================

1. Vector stores are essential for semantic search and RAG applications
2. Chroma is an excellent choice for prototyping and medium-scale applications
3. For production at scale, consider vector databases like Pinecone or Weaviate
4. Use metadata filtering to narrow down search results
5. MMR search provides more diverse results than similarity search
6. Chroma supports persistence, making it suitable for applications beyond POCs

==========================================
ADDITIONAL RESOURCES
==========================================

- Chroma Documentation: https://docs.trychroma.com/
- LangChain Vector Stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
- Embedding Models: https://huggingface.co/models?pipeline_tag=sentence-similarity

==========================================
REQUIRED PACKAGES
==========================================

pip install langchain langchain-community chromadb sentence-transformers
pip install langchain-openai  # Optional, for OpenAI embeddings
"""
