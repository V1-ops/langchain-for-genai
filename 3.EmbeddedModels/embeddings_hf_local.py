from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "LangChain is a framework for developing applications powered by language models."

vector=embedding.embed_query(text)
print(vector)
