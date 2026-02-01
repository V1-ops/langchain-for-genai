from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\manpr\Downloads\GenAI_Plan_Jan_2_to_13.pdf")

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0])