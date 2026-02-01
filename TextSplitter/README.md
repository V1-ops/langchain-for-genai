# Text Splitters in LangChain

## What are Text Splitters?

Text splitters are essential components in LangChain that break down large text documents into smaller, manageable chunks. This is crucial when working with Large Language Models (LLMs) that have token limitations and when building applications like:
- **Retrieval-Augmented Generation (RAG)** systems
- **Question-Answering** applications
- **Semantic search** engines
- **Document summarization** tools

## Why Use Text Splitters?

### 1. **Token Limitations**
LLMs have maximum context window sizes (e.g., 4K, 8K, 16K tokens). Large documents must be split to fit within these limits.

### 2. **Improved Retrieval**
Smaller chunks allow for more precise semantic matching when searching for relevant content in vector databases.

### 3. **Better Embeddings**
Embedding models work better with focused, coherent chunks rather than entire documents.

### 4. **Cost Optimization**
Processing smaller chunks reduces API costs and improves response times.

### 5. **Maintaining Context**
Proper splitting with overlap ensures important context isn't lost between chunks.

---

## Text Splitting Techniques

### 1. Character-Based Text Splitting (`1_.py`)

**File:** `1_.py`

**Theory:**
The `CharacterTextSplitter` is the simplest form of text splitting that divides text based purely on character count. It splits text into chunks of a specified size, optionally with overlap between consecutive chunks.

**How It Works:**
```python
CharacterTextSplitter(
    chunk_size=200,      # Maximum characters per chunk
    chunk_overlap=0,     # Characters to overlap between chunks
    separator=""         # Character(s) to split on
)
```

**Use Cases:**
- Simple text processing where document structure doesn't matter
- Uniform chunk sizes for batch processing
- Quick prototyping and testing

**Limitations:**
- ❌ **Ignores natural text boundaries** (sentences, paragraphs)
- ❌ **May break words mid-sentence**
- ❌ **Loses semantic coherence**
- ❌ **Not suitable for structured content**
- ❌ **Poor handling of code or formatted text**

---

### 2. Recursive Character Text Splitting (`2_.py`)

**File:** `2_.py`

**Theory:**
The `RecursiveCharacterTextSplitter` is a more intelligent splitter that tries to keep related pieces of text together. It recursively tries different separators in order of priority, attempting to split on natural boundaries first (paragraphs, then sentences, then words, then characters).

**How It Works:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]  # Priority order
)
```

**Separator Priority:**
1. `\n\n` - Paragraph breaks (highest priority)
2. `\n` - Line breaks
3. ` ` (space) - Word boundaries
4. `""` - Character level (last resort)

**Use Cases:**
- **Natural language documents** (articles, essays, books)
- **Maintaining semantic coherence**
- **General-purpose text splitting**
- **Content that needs context preservation**

**Advantages:**
- ✅ Preserves paragraph and sentence structure
- ✅ More semantically meaningful chunks
- ✅ Better for question-answering systems
- ✅ Flexible separator configuration

**Limitations:**
- ❌ Still treats all text the same (no special handling)
- ❌ Not optimized for code or structured formats
- ❌ May split long sentences awkwardly
- ❌ Chunk sizes can vary significantly

---

### 3. Language-Aware Document Splitting (`3_.py`)

**File:** `3_.py`

**Theory:**
This approach uses `RecursiveCharacterTextSplitter.from_language()` to split text based on programming language syntax. It understands code structure like classes, functions, and blocks, ensuring splits happen at logical boundaries rather than arbitrary points.

**How It Works:**
```python
RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)
```

**Supported Languages:**
- Python, JavaScript, TypeScript
- Java, C++, C#, Go
- Ruby, PHP, Rust
- Markdown, HTML, LaTeX, and more

**Use Cases:**
- **Code documentation** generation
- **Code search** and retrieval systems
- **AI-powered code assistants**
- **Technical documentation** with code blocks
- **API reference** splitting

**Advantages:**
- ✅ Respects code structure (functions, classes)
- ✅ Maintains syntactic validity
- ✅ Preserves logical code blocks
- ✅ Better for code understanding tasks
- ✅ Reduces broken references

**Limitations:**
- ❌ Language-specific (must specify correct language)
- ❌ May struggle with mixed-language files
- ❌ Large functions/classes might exceed chunk size
- ❌ Comments might be separated from code

---

### 4. Semantic Meaning-Based Text Splitting

**Theory:**
Semantic text splitters use embeddings and similarity measures to split text based on semantic meaning rather than just character count or syntactic structure. They create chunks that are semantically coherent by analyzing the meaning of sentences and grouping similar content together.

**How It Works:**
```python
from langchain_text_splitters import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Initialize with embedding model
embeddings = OpenAIEmbeddings()

# Create semantic chunker
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"  # or "standard_deviation", "interquartile"
)

chunks = splitter.split_text(text)
```

**Breakpoint Threshold Types:**
1. **Percentile** - Split when similarity drops below certain percentile
2. **Standard Deviation** - Split when similarity changes by more than X standard deviations
3. **Interquartile** - Split based on interquartile range of similarities

**How It Determines Splits:**
1. Embeds each sentence in the text
2. Calculates cosine similarity between consecutive sentences
3. Identifies breakpoints where similarity drops significantly
4. Groups sentences with high similarity into chunks

**Use Cases:**
- **Topic-based segmentation** of long articles
- **RAG systems** where semantic coherence is critical
- **Content summarization** requiring thematic consistency
- **Knowledge base** construction
- **Research paper** processing where logical flow matters
- **Conversation history** splitting by topic

**Advantages:**
- ✅ **Semantically coherent chunks** - keeps related ideas together
- ✅ **Natural topic boundaries** - respects conceptual shifts
- ✅ **Better for RAG** - improves retrieval relevance
- ✅ **Context preservation** - maintains logical flow within chunks
- ✅ **Language agnostic** - works across languages with appropriate embeddings
- ✅ **Adaptive chunk sizes** - adjusts based on content complexity

**Limitations:**
- ❌ **Computationally expensive** - requires embedding every sentence
- ❌ **Requires embedding model** - needs API access or local model
- ❌ **Slower processing** - significantly slower than character-based splitters
- ❌ **Variable chunk sizes** - unpredictable chunk lengths
- ❌ **May create very large/small chunks** - less control over size
- ❌ **Embedding quality dependent** - results vary with embedding model
- ❌ **Cost implications** - embedding API calls can be expensive
- ❌ **Memory intensive** - stores embeddings for all sentences
- ❌ **Not suitable for code** - focuses on natural language semantics

**When to Use:**
- Quality of retrieval is more important than speed
- You have access to good embedding models
- Content has clear semantic structure and topics
- Working with long-form content (articles, research papers, books)
- Cost and latency are acceptable trade-offs

**When NOT to Use:**
- Need fast, real-time processing
- Working with code or structured data
- Limited computational resources
- Budget constraints on API calls
- Simple use cases where basic splitters suffice

---

## General Limitations of Text Splitters

### 1. **Loss of Context**
Even with overlap, important context spanning multiple chunks may be lost.

### 2. **Chunk Size Trade-offs**
- **Too small:** Loses context, increases processing overhead
- **Too large:** May exceed token limits, reduces retrieval precision

### 3. **Semantic Boundaries**
No splitter perfectly understands semantic meaning; they rely on syntactic patterns.

### 4. **Complex Structures**
Tables, lists, nested structures, and cross-references may be split awkwardly.

### 5. **Token vs Character Mismatch**
Splitters use character counts, but LLMs use token counts (1 token ≈ 4 characters, varies by language).

### 6. **Metadata Loss**
Document metadata (titles, headers, sections) may be lost during splitting.

### 7. **Language and Encoding**
Performance varies significantly across languages (especially non-Latin scripts).

---

## Best Practices

1. **Use Overlap Wisely**
   - Set `chunk_overlap` to 10-20% of `chunk_size` to maintain context

2. **Choose the Right Splitter**
   - Text documents → `RecursiveCharacterTextSplitter`
   - Code → Language-specific splitter
   - Simple cases → `CharacterTextSplitter`

3. **Experiment with Chunk Sizes**
   - Start with 500-1000 characters for text
   - Adjust based on your use case and embedding model

4. **Test Retrieval Quality**
   - Evaluate how well chunks retrieve relevant information
   - Iterate on splitter configuration

5. **Consider Document Structure**
   - Preserve hierarchical information when possible
   - Use custom separators for domain-specific formats

---

## Summary

| Splitter | Best For | Preserves Structure | Semantic Awareness | Complexity | Speed |
|----------|----------|---------------------|-------------------|------------|-------|
| CharacterTextSplitter | Quick prototyping | ❌ No | ❌ No | Low | ⚡⚡⚡ Fast |
| RecursiveCharacterTextSplitter | Natural language | ✅ Yes | ❌ No | Medium | ⚡⚡ Fast |
| Language-aware Splitter | Source code | ✅✅ Yes++ | ❌ No | Medium | ⚡⚡ Fast |
| Semantic Chunker | Topic-based content | ✅ Yes | ✅✅ Yes++ | High | 🐌 Slow |

**Choosing the Right Splitter:**
- Need speed and simplicity? → **CharacterTextSplitter**
- Processing natural text? → **RecursiveCharacterTextSplitter**
- Splitting source code? → **Language-aware Splitter**
- Need semantic coherence for RAG? → **Semantic Chunker**

Each text splitter has its place in the LangChain ecosystem. Understanding their strengths and limitations helps you choose the right tool for your specific use case.
