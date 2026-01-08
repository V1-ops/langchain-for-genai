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
chat_history =[]

while True:
    user_input= input("you:")
    chat_history.append(user_input)
    if user_input =='exit' :
        break
    result=model.invoke(chat_history )
    chat_history.append(result.content)
    print("Ai:",result.content)
print(chat_history )
# example of a sample conversation stored in chat_history
#['which is bigger 2 or 3 ', 'The number 3 is bigger than the number 2.', 'then multiply the bigger with 10 ', 'Since 3 is the bigger number, you would multiply it by 10:\n\n\\[ 3 \\times 10 = 30 \\]\n\nSo, the result is 30.', 'exit']
# Now this chatbot can remeber the previous conversation as we are passing the chat_history to the model.invoke() function, but a new problem rises here , the chat_history is just a list of strings ,where it is not clear which string is from user and which string is from AI . To solve this problem langchain introduced the concept of Messages like HumanMessage and AIMessage to clearly differentiate between user and AI messages in the chat history. This will be implemented in the next file.