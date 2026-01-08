'''
Provider: Hugging Face

Framework: LangChain

Model: Qwen 2.5

Abstraction:  LangChain ChatModel
'''
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a short note on LangChain.")
print(result.content)