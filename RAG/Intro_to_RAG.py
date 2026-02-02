"""
RAG (Retrieval Augmented Generation) - Complete Theory Guide

==========================================
WHAT IS RAG?
==========================================

RAG (Retrieval Augmented Generation) is a technique that enhances Large Language Models (LLMs) 
by combining them with information retrieval systems. Instead of relying solely on the knowledge 
encoded in the model's parameters during training, RAG allows the model to access external 
knowledge sources dynamically at inference time.

Think of it like an open-book exam vs a closed-book exam:
- Traditional LLM: Closed-book exam (relies only on memorized knowledge)
- RAG: Open-book exam (can reference external materials when answering)


==========================================
WHY DO WE NEED RAG?
==========================================

1. KNOWLEDGE CUTOFF PROBLEM
   - LLMs are trained on data up to a specific date
   - They don't know about events after their training cutoff
   - RAG allows access to up-to-date information

2. HALLUCINATION REDUCTION
   - LLMs can generate plausible but incorrect information
   - RAG grounds responses in actual retrieved documents
   - Reduces made-up facts by providing verifiable sources

3. DOMAIN-SPECIFIC KNOWLEDGE
   - Training custom LLMs is expensive and time-consuming
   - RAG allows instant access to proprietary/specialized knowledge
   - No need to retrain models for specific use cases

4. COST EFFICIENCY
   - Fine-tuning large models is resource-intensive
   - RAG is more cost-effective for adding new knowledge
   - Can update knowledge base without retraining

5. TRANSPARENCY & ATTRIBUTION
   - RAG provides source documents for answers
   - Users can verify information from original sources
   - Improves trust and accountability

6. DYNAMIC KNOWLEDGE
   - Knowledge base can be updated in real-time
   - Add/remove documents as needed
   - Adapt to changing information quickly


==========================================
HOW RAG WORKS - THE COMPLETE PROCESS
==========================================

                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Query         │
                    │   Embedding     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Similarity    │
                    │   Search in     │
                    │   Vector Store  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Retrieve      │
                    │   Relevant      │
                    │   Documents     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Augment       │
                    │   Prompt with   │
                    │   Context       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Generate      │
                    │   Response      │
                    │   (LLM)         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Return        │
                    │   Answer        │
                    └─────────────────┘


DETAILED STEPS:

1. DOCUMENT INGESTION (One-time setup)
   a) Load documents (PDFs, web pages, databases, etc.)
   b) Split documents into chunks (typically 500-1000 tokens)
   c) Convert chunks to embeddings (numerical vectors)
   d) Store embeddings in vector database

2. QUERY PROCESSING (At runtime)
   a) User submits a question
   b) Convert question to embedding using same model
   c) Search vector database for similar embeddings
   d) Retrieve top-k most relevant document chunks

3. CONTEXT AUGMENTATION
   a) Take retrieved documents
   b) Combine with user's original query
   c) Format as a prompt for the LLM
   d) Include instructions for using the context

4. GENERATION
   a) Send augmented prompt to LLM
   b) LLM generates response based on provided context
   c) Response is grounded in retrieved information

5. POST-PROCESSING (Optional)
   a) Fact-checking against source documents
   b) Adding citations/references
   c) Filtering out hallucinations


==========================================
KEY COMPONENTS OF RAG
==========================================

1. DOCUMENT LOADERS
   - Load data from various sources (PDFs, web, databases)
   - Examples: PyPDFLoader, WebBaseLoader, CSVLoader
   - Purpose: Ingest raw data into the system

2. TEXT SPLITTERS
   - Break large documents into smaller chunks
   - Methods: Character-based, token-based, semantic
   - Purpose: Create manageable pieces for embedding

3. EMBEDDING MODELS
   - Convert text to numerical vectors
   - Examples: OpenAI Embeddings, HuggingFace models
   - Purpose: Enable semantic similarity search

4. VECTOR STORES
   - Store and index embeddings efficiently
   - Examples: Chroma, FAISS, Pinecone, Weaviate
   - Purpose: Fast similarity search at scale

5. RETRIEVERS
   - Interface for fetching relevant documents
   - Strategies: Similarity, MMR, threshold-based
   - Purpose: Find most relevant context for query

6. LLMS (Large Language Models)
   - Generate natural language responses
   - Examples: GPT-4, Claude, Llama, Gemini
   - Purpose: Create coherent answers using retrieved context

7. PROMPT TEMPLATES
   - Structure the input to LLMs
   - Include instructions and formatting
   - Purpose: Ensure consistent, high-quality outputs


==========================================
RAG ARCHITECTURE PATTERNS
==========================================

1. NAIVE RAG (Basic)
   Query → Retrieve → Generate
   - Simplest implementation
   - Direct retrieval and generation
   - Good for simple use cases

2. ADVANCED RAG
   - Pre-retrieval optimization (query rewriting)
   - Post-retrieval filtering and reranking
   - Better accuracy and relevance

3. MODULAR RAG
   - Multiple specialized retrievers
   - Hybrid search (keyword + semantic)
   - Ensemble methods for robustness

4. SELF-REFLECTIVE RAG
   - LLM evaluates retrieved documents
   - Decides if more retrieval needed
   - Iterative refinement process

5. AGENTIC RAG
   - Agent decides when to retrieve
   - Multiple tools and data sources
   - Complex multi-step reasoning


==========================================
CHUNKING STRATEGIES
==========================================

Why Chunking Matters:
- LLMs have context window limits
- Smaller chunks = more precise retrieval
- Larger chunks = more context but less precise

Common Strategies:

1. FIXED-SIZE CHUNKING
   - Split by character count or tokens
   - Simple but may break context
   - Example: 500 characters per chunk

2. RECURSIVE CHUNKING
   - Split by paragraphs, then sentences
   - Preserves natural boundaries
   - LangChain's RecursiveCharacterTextSplitter

3. SEMANTIC CHUNKING
   - Split based on meaning/topics
   - Uses embeddings to find boundaries
   - More intelligent but computationally expensive

4. DOCUMENT-BASED CHUNKING
   - Keep document structure (sections, chapters)
   - Preserve hierarchical information
   - Good for structured documents

Best Practices:
- Chunk size: 512-1024 tokens
- Overlap: 10-20% between chunks
- Preserve complete sentences
- Include metadata (source, page number, date)


==========================================
EMBEDDING MODELS EXPLAINED
==========================================

What are Embeddings?
- Dense vector representations of text
- Capture semantic meaning
- Similar meanings → similar vectors
- Typically 384, 768, or 1536 dimensions

Popular Models:

1. OpenAI text-embedding-ada-002
   - 1536 dimensions
   - High quality, general purpose
   - Paid API

2. Sentence-Transformers (HuggingFace)
   - Various sizes and dimensions
   - Free and open-source
   - Can run locally
   - Example: all-MiniLM-L6-v2 (384-dim)

3. Cohere Embeddings
   - Optimized for search
   - Paid API
   - Good performance

4. BGE (BAAI General Embedding)
   - State-of-the-art open-source
   - Multiple size variants
   - Excellent for retrieval

Key Considerations:
- Same model for indexing and querying
- Dimension affects storage and speed
- Domain-specific models for specialized tasks


==========================================
VECTOR DATABASE OPTIONS
==========================================

OPEN-SOURCE / LOCAL:
- Chroma: Easy to use, good for prototyping
- FAISS: Fast, optimized by Facebook
- Annoy: Spotify's approximate nearest neighbor
- Hnswlib: High performance, memory-based

CLOUD / MANAGED:
- Pinecone: Managed, scalable, easy to use
- Weaviate: Open-source with cloud option
- Qdrant: High performance, Rust-based
- Milvus: Highly scalable, enterprise-grade


==========================================
RETRIEVAL STRATEGIES
==========================================

1. SIMILARITY SEARCH
   - Find k most similar documents
   - Based on vector distance (cosine, euclidean)
   - Most common approach

2. MMR (Maximal Marginal Relevance)
   - Balance relevance and diversity
   - Avoid redundant results
   - Good for exploratory queries

3. THRESHOLD-BASED
   - Only return documents above similarity score
   - Filters out low-quality matches
   - More reliable but may return fewer results

4. HYBRID SEARCH
   - Combine keyword and semantic search
   - BM25 + vector similarity
   - Best of both worlds

5. METADATA FILTERING
   - Filter by document properties
   - Date ranges, categories, sources
   - Narrow search space


==========================================
PROMPT ENGINEERING FOR RAG
==========================================

Basic RAG Prompt Template:

```
Context: {retrieved_documents}

Question: {user_question}

Instructions: Answer the question based only on the provided context. 
If the context doesn't contain relevant information, say "I don't have 
enough information to answer this question."

Answer:
```

Advanced Techniques:

1. EXPLICIT INSTRUCTIONS
   - Tell model to use only provided context
   - Request citations or quotes
   - Define output format

2. FEW-SHOT EXAMPLES
   - Show example Q&A pairs
   - Demonstrate desired behavior
   - Improve consistency

3. CHAIN-OF-THOUGHT
   - Ask model to reason step-by-step
   - Better for complex questions
   - More transparent reasoning

4. SYSTEM PROMPTS
   - Define role and behavior
   - Set constraints and guidelines
   - Establish tone and style


==========================================
RAG EVALUATION METRICS
==========================================

1. RETRIEVAL METRICS
   - Precision@k: Relevant docs in top k
   - Recall@k: % of relevant docs retrieved
   - MRR (Mean Reciprocal Rank): Position of first relevant doc
   - NDCG (Normalized Discounted Cumulative Gain): Ranking quality

2. GENERATION METRICS
   - Faithfulness: Answer grounded in context?
   - Relevance: Answer addresses the question?
   - Coherence: Is answer well-structured?
   - Correctness: Factually accurate?

3. END-TO-END METRICS
   - Human evaluation (gold standard)
   - RAGAS: Automated RAG evaluation framework
   - LLM-as-a-judge: Use another LLM to evaluate


==========================================
COMMON RAG CHALLENGES & SOLUTIONS
==========================================

1. POOR RETRIEVAL QUALITY
   Problem: Irrelevant documents retrieved
   Solutions:
   - Improve chunking strategy
   - Use better embedding models
   - Add metadata filtering
   - Implement hybrid search

2. CONTEXT WINDOW LIMITATIONS
   Problem: Too many/large documents for LLM
   Solutions:
   - Retrieve fewer documents
   - Use reranking to prioritize
   - Implement context compression
   - Use models with larger windows

3. HALLUCINATIONS
   Problem: LLM invents information
   Solutions:
   - Stronger prompt instructions
   - Post-generation fact-checking
   - Lower temperature settings
   - Use citation mechanisms

4. OUTDATED INFORMATION
   Problem: Knowledge base not current
   Solutions:
   - Regular reindexing schedule
   - Incremental updates
   - Timestamp-based filtering
   - Automated data pipelines

5. SLOW RESPONSE TIME
   Problem: High latency in RAG pipeline
   Solutions:
   - Use approximate nearest neighbor search
   - Cache frequent queries
   - Optimize chunk sizes
   - Use faster embedding models


==========================================
RAG VS FINE-TUNING
==========================================

RAG:
✓ Fast to implement
✓ Easy to update knowledge
✓ Transparent and explainable
✓ Cost-effective
✗ Dependent on retrieval quality
✗ Context window limitations

FINE-TUNING:
✓ Knowledge internalized in model
✓ No retrieval overhead
✓ Consistent behavior
✗ Expensive and time-consuming
✗ Hard to update knowledge
✗ Requires significant data and compute

HYBRID APPROACH:
- Fine-tune for style/format/domain language
- Use RAG for factual knowledge
- Best of both worlds


==========================================
REAL-WORLD RAG APPLICATIONS
==========================================

1. CUSTOMER SUPPORT CHATBOTS
   - Knowledge base: Documentation, FAQs, policies
   - Benefits: Consistent, accurate responses

2. LEGAL DOCUMENT ANALYSIS
   - Knowledge base: Case law, contracts, regulations
   - Benefits: Quick research, precedent finding

3. MEDICAL INFORMATION SYSTEMS
   - Knowledge base: Research papers, clinical guidelines
   - Benefits: Evidence-based recommendations

4. ENTERPRISE SEARCH
   - Knowledge base: Company documents, emails, wikis
   - Benefits: Find information across silos

5. EDUCATION & TUTORING
   - Knowledge base: Textbooks, course materials
   - Benefits: Personalized learning assistance

6. RESEARCH ASSISTANCE
   - Knowledge base: Academic papers, articles
   - Benefits: Literature review, summarization

7. CODE DOCUMENTATION
   - Knowledge base: Code repos, API docs
   - Benefits: Contextual code help


==========================================
ADVANCED RAG TECHNIQUES
==========================================

1. QUERY TRANSFORMATION
   - Rewrite user query for better retrieval
   - Generate multiple query variants
   - Use HyDE (Hypothetical Document Embeddings)

2. DOCUMENT RERANKING
   - Initial broad retrieval (e.g., top 20)
   - Rerank with cross-encoder model
   - Return top k after reranking

3. ITERATIVE RETRIEVAL
   - Retrieve, generate partial answer
   - Use partial answer to refine retrieval
   - Iterate until satisfactory answer

4. PARENT-CHILD CHUNKING
   - Store small chunks for precise retrieval
   - Return larger parent document for context
   - Best of both worlds

5. MULTI-HOP REASONING
   - Answer requires multiple documents
   - Chain together multiple retrieval steps
   - Build answer incrementally


==========================================
RAG IMPLEMENTATION CHECKLIST
==========================================

Phase 1: Data Preparation
□ Identify data sources
□ Load documents
□ Clean and preprocess
□ Choose chunking strategy
□ Implement splitting logic

Phase 2: Indexing
□ Select embedding model
□ Choose vector database
□ Generate embeddings
□ Store in vector DB
□ Add metadata

Phase 3: Retrieval Setup
□ Configure retriever
□ Set retrieval parameters (k, threshold)
□ Test retrieval quality
□ Implement fallbacks

Phase 4: Generation
□ Select LLM
□ Design prompt template
□ Set generation parameters
□ Test responses

Phase 5: Evaluation
□ Create test dataset
□ Measure retrieval metrics
□ Evaluate generation quality
□ Iterate and improve

Phase 6: Deployment
□ Optimize performance
□ Add monitoring/logging
□ Implement caching
□ Set up error handling


==========================================
BEST PRACTICES
==========================================

1. START SIMPLE
   - Begin with naive RAG
   - Add complexity as needed
   - Measure improvement at each step

2. FOCUS ON RETRIEVAL QUALITY
   - Retrieval is often the bottleneck
   - Better documents = better answers
   - Test different strategies

3. OPTIMIZE CHUNKING
   - Experiment with chunk sizes
   - Use overlap between chunks
   - Preserve context boundaries

4. USE METADATA EFFECTIVELY
   - Add source, date, category
   - Enable filtering options
   - Improve relevance

5. MONITOR AND ITERATE
   - Log queries and results
   - Collect user feedback
   - Continuously improve

6. HANDLE EDGE CASES
   - No relevant documents found
   - Conflicting information
   - Ambiguous queries

7. ENSURE DATA QUALITY
   - Regular updates
   - Remove duplicates
   - Validate sources


==========================================
FUTURE OF RAG
==========================================

Emerging Trends:

1. MULTIMODAL RAG
   - Text + images + audio
   - Unified retrieval across modalities

2. GRAPH-ENHANCED RAG
   - Knowledge graphs + vector search
   - Relationship-aware retrieval

3. AGENTIC WORKFLOWS
   - Autonomous decision-making
   - Dynamic tool selection

4. REAL-TIME LEARNING
   - Continuous knowledge updates
   - Adaptive retrieval strategies

5. PRIVACY-PRESERVING RAG
   - Federated learning approaches
   - Secure multi-party computation


==========================================
RESOURCES FOR LEARNING
==========================================

Documentation:
- LangChain: https://python.langchain.com/
- LlamaIndex: https://docs.llamaindex.ai/
- Haystack: https://haystack.deepset.ai/

Papers:
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

Courses & Tutorials:
- DeepLearning.AI: LangChain courses
- Coursera: NLP Specialization
- YouTube: Numerous RAG tutorials

Tools & Frameworks:
- LangChain: Full-featured RAG framework
- LlamaIndex: Data framework for LLMs
- Haystack: NLP framework with RAG support


==========================================
CONCLUSION
==========================================

RAG is a powerful technique that bridges the gap between static LLM knowledge 
and dynamic, domain-specific information needs. By combining retrieval systems 
with generation capabilities, RAG enables:

- More accurate and up-to-date responses
- Domain specialization without fine-tuning
- Transparent and verifiable answers
- Cost-effective knowledge integration

Key Takeaways:
1. RAG augments LLMs with external knowledge
2. Quality retrieval is crucial for success
3. Proper chunking and embedding are foundational
4. Prompt engineering guides effective generation
5. Continuous evaluation and iteration are essential

RAG is not a silver bullet, but when implemented correctly, it significantly 
enhances the capabilities of LLM applications, making them more reliable, 
accurate, and valuable for real-world use cases.
"""

# Simple conceptual example (no actual implementation)
def conceptual_rag_example():
    """
    Conceptual RAG Pipeline (Pseudocode)
    """
    
    # Step 1: Prepare Knowledge Base (one-time)
    documents = load_documents("./data/")
    chunks = split_into_chunks(documents, chunk_size=500)
    embeddings = create_embeddings(chunks)
    vector_store = store_in_database(embeddings)
    
    # Step 2: Process User Query (runtime)
    user_query = "What is machine learning?"
    query_embedding = create_embedding(user_query)
    
    # Step 3: Retrieve Relevant Documents
    relevant_docs = vector_store.similarity_search(query_embedding, k=3)
    
    # Step 4: Augment Prompt
    context = "\n".join([doc.content for doc in relevant_docs])
    prompt = f"""
    Context: {context}
    
    Question: {user_query}
    
    Answer based on the context provided:
    """
    
    # Step 5: Generate Response
    response = llm.generate(prompt)
    
    return response


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 70)
    print("This file contains comprehensive theory about RAG.")
    print("See RAG_with_LangChain.py for practical implementation examples.")
    print("=" * 70)
