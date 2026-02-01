from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
prompt = PromptTemplate(
    template=" Summarize the following text:\n{text}",
    input_variables=["text"]
)

loader = TextLoader(r"C:\Users\manpr\OneDrive\Desktop\LangChain\DataLoaders\cricket.txt",encoding="utf8")

docs=loader.load()

print(type(docs))
print(docs[0])
chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})
print(result)