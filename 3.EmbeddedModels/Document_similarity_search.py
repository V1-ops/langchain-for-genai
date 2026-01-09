# HuggingFaceEmbeddings - LangChain wrapper for embedding models
# Why: Converts text into numerical vectors for semantic similarity comparison
from langchain_huggingface import HuggingFaceEmbeddings

# load_dotenv() - Loads environment variables from .env file
# Why: In case API tokens are needed (though this model runs locally)
from dotenv import load_dotenv

# cosine_similarity - Measures similarity between vectors
# Why: Calculates how similar two embeddings are (1.0 = identical, 0 = unrelated, -1 = opposite)
#      Cosine similarity is the standard metric for comparing text embeddings
from sklearn.metrics.pairwise import cosine_similarity

# numpy - Numerical computing library
# Why: Used for array operations and numerical computations with embeddings
import numpy as np 

# load_dotenv() - Executes loading of environment variables
# Why: Makes any necessary tokens available
load_dotenv()

# HuggingFaceEmbeddings() - Initializes the sentence transformer model
# Why: Creates an embedding model that runs locally to convert text to vectors
# Parameters:
#   - model_name: "all-MiniLM-L6-v2" is optimized for semantic similarity tasks
#     Why: Fast, lightweight, and produces high-quality semantic embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# documents - List of text documents to search through
# Why: This is our knowledge base / document corpus that we want to search
#      In real applications, this could be thousands of documents
documents=[
    "LangChain is a framework for developing applications powered by language models.",
    "Embeddings are numerical representations of text that capture semantic meaning.",
    "Document similarity search allows you to find documents similar to a given query.",
    "chat models are fine-tuned for conversational tasks."
]

# query - The search query from user
# Why: This is what we want to find similar documents for
query= "What is Document similarity search ?"

# embedding.embed_documents() - Converts multiple documents into vectors
# Why: Generates embeddings for all documents in our corpus at once
#      Returns a list of vectors, one for each document
# Note: Use embed_documents() for batch processing of multiple texts (more efficient)
doc_embeddings = embedding.embed_documents(documents)

# embedding.embed_query() - Converts the search query into a vector
# Why: Generates embedding for the query so we can compare it with document embeddings
#      Query and documents must be in the same vector space for comparison
query_embedding = embedding.embed_query(query)

# cosine_similarity() - Calculates similarity scores between query and all documents
# Why: Compares the query vector with each document vector using cosine similarity
#      Returns scores from -1 to 1 (higher = more similar)
# Parameters:
#   - [query_embedding]: Query vector wrapped in a list (required format)
#   - doc_embeddings: All document vectors to compare against
#   - [0]: Extracts the first (and only) row of results as a 1D array
scores=cosine_similarity([query_embedding], doc_embeddings)[0]

# sorted() - Sorts documents by similarity score
# Why: Finds the document with the highest similarity score
# Process:
#   - enumerate(scores): Creates (index, score) pairs
#   - list(): Converts to list
#   - sorted(..., key=lambda x:x[1]): Sorts by score (x[1])
#   - [-1]: Gets the last element (highest score)
index,score=sorted(list(enumerate(scores)), key =lambda x:x[1])[-1]

# print() - Displays the search query
# Why: Shows what the user was searching for
print(query)

# print() - Displays the most similar document
# Why: Returns the document from our corpus that best matches the query
#      documents[index] retrieves the document at the position with highest score
print(documents[index])

# print() - Displays the similarity score
# Why: Shows how confident the match is (closer to 1.0 = better match)
print("Similarity socre is :",score)