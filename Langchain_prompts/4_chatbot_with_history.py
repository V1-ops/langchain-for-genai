# ChatHuggingFace - LangChain wrapper for Hugging Face chat models
# Why: Provides standardized chat interface with conversation history support
# HuggingFaceEndpoint - Connects to Hugging Face Inference API
# Why: Enables remote access to large language models without local resources
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load_dotenv() - Loads environment variables from .env file
# Why: Securely manages API keys needed for Hugging Face API authentication
from dotenv import load_dotenv

# load_dotenv() - Executes the loading of environment variables
# Why: Makes HUGGINGFACEHUB_API_TOKEN available for authentication
load_dotenv()

# HuggingFaceEndpoint() - Initializes connection to Hugging Face model
# Why: Sets up the base LLM with specific parameters for text generation
# Parameters explained:
#   - repo_id: Qwen 2.5 72B is a powerful multilingual instruction-following model
#   - task: text-generation enables chat/completion capabilities
#   - temperature: 0.7 balances creativity with coherence
#   - max_new_tokens: 256 limits response length for faster responses
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,

)

# ChatHuggingFace() - Wraps the endpoint for chat functionality
# Why: Enables conversation with context by passing message history
model = ChatHuggingFace(llm=llm)

# chat_history - List to store all conversation messages
# Why: Maintains context so the AI can reference previous messages in the conversation
chat_history =[]

# while True - Infinite loop for continuous conversation
# Why: Keeps the chatbot running until user types 'exit'
while True:
    # input() - Gets user input from console
    # Why: Captures what the user wants to ask the AI
    user_input= input("you:")
    
    # chat_history.append() - Adds user message to history
    # Why: Stores user input so the model can see what was asked
    chat_history.append(user_input)
    
    # if condition - Checks for exit command
    # Why: Provides a way for user to end the conversation
    if user_input =='exit' :
        break
    
    # model.invoke() - Sends entire chat history to the model
    # Why: By passing the full history, the model can understand context and reference
    #      previous messages (e.g., "multiply the bigger one" references earlier conversation)
    result=model.invoke(chat_history )
    
    # chat_history.append() - Adds AI response to history
    # Why: Stores the AI's response so it can be referenced in future turns
    chat_history.append(result.content)
    
    # print() - Displays the AI's response
    # Why: Shows the user what the AI responded
    print("Ai:",result.content)

# print() - Displays full conversation history
# Why: Useful for debugging and seeing the complete conversation flow
print(chat_history )
# example of a sample conversation stored in chat_history
#['which is bigger 2 or 3 ', 'The number 3 is bigger than the number 2.', 'then multiply the bigger with 10 ', 'Since 3 is the bigger number, you would multiply it by 10:\n\n\\[ 3 \\times 10 = 30 \\]\n\nSo, the result is 30.', 'exit']
# Now this chatbot can remeber the previous conversation as we are passing the chat_history to the model.invoke() function, but a new problem rises here , the chat_history is just a list of strings ,where it is not clear which string is from user and which string is from AI . To solve this problem langchain introduced the concept of Messages like HumanMessage and AIMessage to clearly differentiate between user and AI messages in the chat history. This will be implemented in the next file.