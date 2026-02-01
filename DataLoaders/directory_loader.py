from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# DirectoryLoader loads all files from a specified directory
# It requires:
# 1. path: the directory path to load files from
# 2. glob: pattern to match files (e.g., "**/*.txt" for all text files)
# 3. loader_cls: the loader class to use for each file (e.g., TextLoader)
# 4. loader_kwargs: keyword arguments to pass to the loader (optional)

loader = DirectoryLoader(
    path=r"C:\Users\manpr\OneDrive\Desktop\LangChain\DataLoaders",
    glob="**/*.txt",  # Load all .txt files recursively
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf8"}
)

# Load all documents
docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print(f"Type of docs: {type(docs)}")

# Print the first document
if docs:
    print(f"\nFirst document content preview:")
    print(docs[0].page_content[:200])  # First 200 characters
