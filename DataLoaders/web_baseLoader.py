from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.linkedin.com/feed/?highlightedUpdateType=COMMENTS_BY_YOUR_NETWORK&highlightedUpdateUrn=urn%3Ali%3Aactivity%3A7422520131546890240")
docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0])
