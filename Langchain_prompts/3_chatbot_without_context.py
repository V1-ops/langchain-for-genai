from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,

)
model = ChatHuggingFace(llm=llm)

while True:
    user_input= input("you:")
    if user_input =='exit' :
        break
    result=model.invoke(user_input)
    print("Ai:",result.content)
    
    # The problem in above chatbot is that it does not have any history , it cannot remember previous conversation.

