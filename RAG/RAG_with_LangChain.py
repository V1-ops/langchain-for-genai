"""
RAG Implementation with LangChain - Practical Examples

This file demonstrates how to implement RAG (Retrieval Augmented Generation) 
using LangChain framework with working code examples.

Prerequisites:
pip install langchain langchain-community langchain-openai
pip install chromadb faiss-cpu sentence-transformers
pip install pypdf python-dotenv
"""

from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os


# ==========================================
# Example 1: Basic RAG Pipeline (No LLM)
# ==========================================

def example_1_basic_rag_retrieval():
    """
    Demonstrate the retrieval part of RAG without actual LLM generation.
    This shows how to build the knowledge base and retrieve relevant documents.
    """
    
    print("=" * 70)
    print("Example 1: Basic RAG - Retrieval Only")
    print("=" * 70)
    
    # Step 1: Create sample documents (knowledge base)
    documents = [
        """Machine learning is a branch of artificial intelligence (AI) that focuses on 
        building systems that can learn from and make decisions based on data. ML algorithms 
        use computational methods to learn information directly from data without relying 
        on predetermined equations.""",
        
        """Deep learning is a subset of machine learning that uses neural networks with 
        multiple layers (deep neural networks) to progressively extract higher-level features 
        from raw input. It has been particularly successful in image and speech recognition.""",
        
        """Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
        between computers and human language. It involves programming computers to process 
        and analyze large amounts of natural language data.""",
        
        """Computer vision is a field of AI that trains computers to interpret and understand 
        the visual world. Using digital images from cameras and videos and deep learning models, 
        machines can accurately identify and classify objects.""",
        
        """Reinforcement learning is an area of machine learning where an agent learns to 
        make decisions by performing actions in an environment to achieve maximum cumulative 
        reward. It's inspired by behavioral psychology."""
    ]
    
    # Step 2: Create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Step 3: Create vector store
    print("\nBuilding vector store...")
    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embeddings
    )
    
    # Step 4: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Retrieve top 2 documents
    )
    
    # Step 5: Query and retrieve
    query = "What is machine learning and how does it work?"
    print(f"\nQuery: {query}")
    
    retrieved_docs = retriever.get_relevant_documents(query)
    
    print(f"\nRetrieved {len(retrieved_docs)} documents:\n")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"Document {i}:")
        print(doc.page_content[:200] + "...")
        print()
    
    return vectorstore, retriever


# ==========================================
# Example 2: RAG with Document Chunking
# ==========================================

def example_2_rag_with_chunking():
    """
    Demonstrate proper document chunking for RAG.
    Large documents need to be split into manageable chunks.
    """
    
    print("\n" + "=" * 70)
    print("Example 2: RAG with Text Chunking")
    print("=" * 70)
    
    # Long document that needs chunking
    long_document = """
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
    programming languages in the world.
    
    Python supports multiple programming paradigms, including procedural, object-oriented, and functional
    programming. It has a comprehensive standard library that provides tools suited to many tasks.
    
    Python is widely used in web development, with frameworks like Django and Flask. These frameworks
    provide the structure and tools needed to build robust web applications quickly.
    
    In data science, Python has become the language of choice. Libraries like NumPy, Pandas, and 
    Matplotlib provide powerful tools for data manipulation, analysis, and visualization.
    
    Machine learning and artificial intelligence applications heavily rely on Python. Libraries such
    as TensorFlow, PyTorch, and Scikit-learn make it easy to build and deploy ML models.
    
    Python's simplicity makes it an excellent choice for beginners, while its power and flexibility
    make it suitable for large-scale applications. Companies like Google, Netflix, and NASA use Python.
    """
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Characters per chunk
        chunk_overlap=50,  # Overlap between chunks
        separators=["\n\n", "\n", " ", ""]  # Split priorities
    )
    
    # Split document
    chunks = text_splitter.split_text(long_document)
    
    print(f"\nOriginal document length: {len(long_document)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk.strip())
    
    # Create vector store with chunks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    
    # Retrieve relevant chunks
    query = "What is Python used for in data science?"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.get_relevant_documents(query)
    
    print(f"\n\nQuery: {query}")
    print("\nRelevant chunks retrieved:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content.strip()}")
    
    return vectorstore


# ==========================================
# Example 3: RAG with Metadata
# ==========================================

def example_3_rag_with_metadata():
    """
    Add metadata to documents for better filtering and context.
    """
    
    print("\n" + "=" * 70)
    print("Example 3: RAG with Metadata Filtering")
    print("=" * 70)
    
    # Documents with metadata
    documents = [
        Document(
            page_content="Python 3.9 introduced the merge operator for dictionaries.",
            metadata={"language": "Python", "version": "3.9", "category": "feature"}
        ),
        Document(
            page_content="JavaScript ES2020 added BigInt for arbitrary precision integers.",
            metadata={"language": "JavaScript", "version": "ES2020", "category": "feature"}
        ),
        Document(
            page_content="Python's pip is a package installer for Python packages.",
            metadata={"language": "Python", "version": "all", "category": "tool"}
        ),
        Document(
            page_content="npm is the package manager for JavaScript and Node.js.",
            metadata={"language": "JavaScript", "version": "all", "category": "tool"}
        ),
        Document(
            page_content="Python 3.10 introduced structural pattern matching.",
            metadata={"language": "Python", "version": "3.10", "category": "feature"}
        )
    ]
    
    # Create vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="metadata_example"
    )
    
    # Query with metadata filter
    print("\nQuery: New language features")
    print("Filter: Only Python documents")
    
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"language": "Python"}
        }
    )
    
    results = retriever.get_relevant_documents("new language features")
    
    print(f"\nFound {len(results)} Python documents:")
    for doc in results:
        print(f"\n- {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")
    
    return vectorstore


# ==========================================
# Example 4: Complete RAG Pipeline with LLM
# ==========================================

def example_4_complete_rag_with_llm():
    """
    Complete RAG pipeline: Retrieval + Generation with LLM.
    Note: Requires OpenAI API key or uses HuggingFace alternative.
    """
    
    print("\n" + "=" * 70)
    print("Example 4: Complete RAG Pipeline with LLM")
    print("=" * 70)
    
    # Knowledge base
    documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "RAG combines retrieval systems with generative AI to provide accurate, contextual responses.",
        "Vector stores enable efficient similarity search over large document collections.",
        "Embeddings are numerical representations that capture the semantic meaning of text.",
        "Prompt engineering is crucial for getting high-quality outputs from LLMs."
    ]
    
    # Setup embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nNote: OPENAI_API_KEY not found in environment.")
        print("This example will show the prompt template but won't generate actual responses.")
        print("To run with OpenAI, set: export OPENAI_API_KEY='your-key-here'\n")
        
        # Demonstrate the RAG prompt structure
        query = "What is RAG and why is it useful?"
        docs = retriever.get_relevant_documents(query)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print("Retrieved Context:")
        print("-" * 70)
        print(context)
        print("-" * 70)
        
        print("\n\nRAG Prompt Template:")
        print("-" * 70)
        prompt_text = f"""Answer the question based only on the following context:

{context}

Question: {query}

Answer: """
        print(prompt_text)
        print("-" * 70)
        
        return None
    
    # If OpenAI key is available, run full RAG pipeline
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Define prompt template
        template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run query
        query = "What is RAG and why is it useful?"
        print(f"\nQuery: {query}")
        
        response = rag_chain.invoke(query)
        
        print(f"\nRAG Response:\n{response}")
        
        return rag_chain
        
    except Exception as e:
        print(f"\nError with OpenAI: {e}")
        return None


# ==========================================
# Example 5: RAG with Custom Prompt Template
# ==========================================

def example_5_custom_prompts():
    """
    Demonstrate different prompt templates for RAG.
    """
    
    print("\n" + "=" * 70)
    print("Example 5: RAG with Custom Prompt Templates")
    print("=" * 70)
    
    # Sample context from retrieval
    context = """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    """
    
    question = "What is machine learning?"
    
    # Template 1: Basic RAG
    basic_template = f"""Context: {context}

Question: {question}

Answer based on the context:"""
    
    # Template 2: With Instructions
    instructed_template = f"""You are a helpful AI assistant. Answer the question based ONLY on the 
following context. If the answer cannot be found in the context, say "I don't have enough 
information to answer this question."

Context: {context}

Question: {question}

Answer:"""
    
    # Template 3: With Citations
    citation_template = f"""Answer the question using the provided context. Include direct quotes 
from the context to support your answer.

Context: {context}

Question: {question}

Answer (with quotes):"""
    
    # Template 4: Step-by-Step
    cot_template = f"""Answer the question by reasoning through it step-by-step using the context.

Context: {context}

Question: {question}

Let's think step by step:
1."""
    
    print("\n1. Basic RAG Template:")
    print("-" * 70)
    print(basic_template)
    
    print("\n\n2. Instructed Template:")
    print("-" * 70)
    print(instructed_template)
    
    print("\n\n3. Citation Template:")
    print("-" * 70)
    print(citation_template)
    
    print("\n\n4. Chain-of-Thought Template:")
    print("-" * 70)
    print(cot_template)


# ==========================================
# Example 6: RAG with Conversational Memory
# ==========================================

def example_6_conversational_rag():
    """
    RAG with conversation history for chatbot applications.
    """
    
    print("\n" + "=" * 70)
    print("Example 6: Conversational RAG")
    print("=" * 70)
    
    # Knowledge base
    documents = [
        "Python was created by Guido van Rossum in 1991.",
        "Python is known for its simple and readable syntax.",
        "Python is widely used in data science, web development, and automation.",
        "Popular Python frameworks include Django, Flask, and FastAPI."
    ]
    
    # Setup vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Simulate conversation
    conversation_history = []
    
    queries = [
        "Who created Python?",
        "When was it created?",
        "What is it used for?"
    ]
    
    print("\nSimulated Conversation:")
    print("-" * 70)
    
    for query in queries:
        # Retrieve context
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Build prompt with history
        history_text = "\n".join([
            f"User: {q}\nAssistant: {a}" 
            for q, a in conversation_history
        ])
        
        prompt = f"""Previous conversation:
{history_text if history_text else "None"}

Current context from knowledge base:
{context}

Current question: {query}

Answer: """
        
        print(f"\nUser: {query}")
        print(f"Retrieved Context: {context[:100]}...")
        
        # Simulated response (in real app, this would be LLM generated)
        simulated_answer = f"Based on the context: {context.split('.')[0]}."
        print(f"Assistant: {simulated_answer}")
        
        # Add to history
        conversation_history.append((query, simulated_answer))


# ==========================================
# Example 7: Advanced RAG with Reranking
# ==========================================

def example_7_rag_with_reranking():
    """
    Retrieve more documents, then rerank by relevance.
    """
    
    print("\n" + "=" * 70)
    print("Example 7: RAG with Reranking")
    print("=" * 70)
    
    documents = [
        "Paris is the capital city of France.",
        "The Eiffel Tower is located in Paris.",
        "France is known for its cuisine and wine.",
        "Paris has world-class museums like the Louvre.",
        "The Seine River flows through Paris.",
        "Lyon is another major city in France.",
        "French is the official language of France."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    
    query = "Tell me about Paris landmarks"
    
    # Initial broad retrieval
    print(f"\nQuery: {query}")
    print("\nStep 1: Initial Retrieval (top 5)")
    
    initial_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    initial_docs = initial_retriever.get_relevant_documents(query)
    
    for i, doc in enumerate(initial_docs, 1):
        print(f"{i}. {doc.page_content}")
    
    # Simple reranking based on keyword matching
    print("\nStep 2: Reranking based on 'Paris' keyword")
    
    def simple_rerank(docs, query_keywords):
        scored_docs = []
        for doc in docs:
            score = sum(keyword.lower() in doc.page_content.lower() 
                       for keyword in query_keywords)
            scored_docs.append((score, doc))
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs]
    
    reranked_docs = simple_rerank(initial_docs, ["Paris", "landmark", "tower"])
    
    print("\nTop 3 after reranking:")
    for i, doc in enumerate(reranked_docs[:3], 1):
        print(f"{i}. {doc.page_content}")


# ==========================================
# Example 8: RAG Evaluation
# ==========================================

def example_8_rag_evaluation():
    """
    Demonstrate how to evaluate RAG system performance.
    """
    
    print("\n" + "=" * 70)
    print("Example 8: RAG Evaluation")
    print("=" * 70)
    
    # Test dataset
    test_data = [
        {
            "question": "What is machine learning?",
            "relevant_docs": [0, 1],  # Indices of relevant documents
            "ground_truth": "Machine learning is a method of data analysis..."
        },
        {
            "question": "What is deep learning?",
            "relevant_docs": [1, 2],
            "ground_truth": "Deep learning is a subset of machine learning..."
        }
    ]
    
    documents = [
        "Machine learning is a method of data analysis that automates model building.",
        "Deep learning uses neural networks with multiple layers.",
        "Neural networks are inspired by biological neural networks.",
        "Artificial intelligence encompasses machine learning and deep learning."
    ]
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    print("\nEvaluation Metrics:\n")
    
    for i, test in enumerate(test_data, 1):
        print(f"Test Case {i}: {test['question']}")
        
        # Retrieve documents
        retrieved = retriever.get_relevant_documents(test['question'])
        retrieved_indices = [documents.index(doc.page_content) for doc in retrieved]
        
        # Calculate metrics
        relevant_set = set(test['relevant_docs'])
        retrieved_set = set(retrieved_indices)
        
        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Retrieved docs: {retrieved_indices}")
        print(f"  Relevant docs: {list(relevant_set)}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1 Score: {f1:.2f}\n")


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG IMPLEMENTATION WITH LANGCHAIN - PRACTICAL EXAMPLES")
    print("=" * 70)
    
    try:
        # Run examples
        example_1_basic_rag_retrieval()
        example_2_rag_with_chunking()
        example_3_rag_with_metadata()
        example_4_complete_rag_with_llm()
        example_5_custom_prompts()
        example_6_conversational_rag()
        example_7_rag_with_reranking()
        example_8_rag_evaluation()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Experiment with different embedding models")
        print("2. Try different chunk sizes and overlaps")
        print("3. Test with your own documents")
        print("4. Add OpenAI API key for full LLM integration")
        print("5. Implement evaluation on your use case")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure required packages are installed:")
        print("pip install langchain langchain-community chromadb faiss-cpu")
        print("pip install sentence-transformers langchain-openai")


"""
==========================================
KEY TAKEAWAYS
==========================================

1. RAG combines retrieval with generation for accurate responses
2. Document chunking is crucial for effective retrieval
3. Metadata enables powerful filtering capabilities
4. Prompt engineering determines output quality
5. Evaluation is essential for improving RAG systems

==========================================
PRODUCTION CONSIDERATIONS
==========================================

1. Choose appropriate chunk size (test different values)
2. Use persistent vector stores for production
3. Implement caching for frequent queries
4. Add error handling and fallbacks
5. Monitor retrieval quality and latency
6. Regular reindexing for updated information
7. Use appropriate embedding models for your domain

==========================================
ADVANCED TOPICS TO EXPLORE
==========================================

1. Hybrid search (keyword + semantic)
2. Multi-query retrieval
3. Contextual compression
4. Parent-child document chunking
5. Graph-enhanced RAG
6. Agentic RAG workflows
"""
