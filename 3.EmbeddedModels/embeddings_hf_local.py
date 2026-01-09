# HuggingFaceEmbeddings - LangChain wrapper for Hugging Face embedding models
# Why: Converts text into numerical vectors (embeddings) that represent semantic meaning
#      These vectors enable mathematical comparison of text similarity
from langchain_huggingface import HuggingFaceEmbeddings

# HuggingFaceEmbeddings() - Initializes the embedding model locally
# Why: Loads a pre-trained model that can convert text to vector representations
# Parameters:
#   - model_name: "sentence-transformers/all-MiniLM-L6-v2" is a lightweight, efficient model
#     Why: It's small (22MB), fast, and produces good quality embeddings for semantic search
#          Perfect for local execution without heavy computational requirements
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text to be converted into embedding
# Why: Demonstrates how to convert a sentence into a numerical representation
text = "LangChain is a framework for developing applications powered by language models."

# embedding.embed_query() - Converts a single text string into a vector
# Why: Generates a numerical representation (list of floats) of the text
#      This vector captures the semantic meaning - similar texts will have similar vectors
#      Returns a 384-dimensional vector (for this model)
# Note: Use embed_query() for search queries, embed_documents() for document corpus
vector=embedding.embed_query(text)

# print() - Displays the embedding vector
# Why: Shows the numerical representation (e.g., [0.123, -0.456, 0.789, ...])
#      This vector can be used for similarity comparisons, clustering, or as input to ML models
print(vector)
