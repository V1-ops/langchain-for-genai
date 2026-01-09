
# ChatHuggingFace - LangChain wrapper for HF chat models
# HuggingFaceEndpoint - Connects to Hugging Face Inference API
# Why: Enables structured chat interactions with API-hosted models
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# SystemMessage, HumanMessage, AIMessage - Explicit message type classes
# Why: These classes solve the problem from 4_chatbot_with_history.py where plain strings
#      didn't differentiate between user and AI messages. Now each message has a clear role.
from langchain_core.messages import SystemMessage , HumanMessage, AIMessage

# load_dotenv() - Loads environment variables from .env file
# Why: Securely manages API keys without hardcoding
from dotenv import load_dotenv

# load_dotenv() - Executes loading of environment variables
# Why: Makes HUGGINGFACEHUB_API_TOKEN available for authentication
load_dotenv()

# HuggingFaceEndpoint() - Initializes connection to HF Inference API
# Why: Sets up the base LLM with specific generation parameters
# Parameters:
#   - repo_id: Qwen 2.5 72B is a powerful instruction-following model
#   - task: text-generation enables chat/completion capabilities
#   - temperature: 0.7 balances creativity with coherence
#   - max_new_tokens: 256 limits response length for faster responses
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=256,

)
# ChatHuggingFace() - Wraps the endpoint for chat-based interactions
# Why: Provides invoke() method and handles message objects properly
model = ChatHuggingFace(llm=llm)

# chat_history - List initialized with SystemMessage
# Why: SystemMessage sets the AI's behavior for the entire conversation
#      Unlike 4_chatbot_with_history.py (plain list), this explicitly defines the assistant's role
#      All subsequent messages reference this system instruction
chat_history =[SystemMessage(content='You are a helpful assistant.')]


# while True - Infinite loop for continuous conversation
# Why: Keeps the chatbot running until user types 'exit'
while True:
    
    # input() - Gets user input from console
    # Why: Captures what the user wants to ask the AI
    user_input= input("you:")
    
    # if condition - Checks for exit command FIRST (case-insensitive, strips whitespace)
    # Why: Must check before appending to history to properly exit
    #      .lower().strip() handles "exit", "EXIT", "Exit", " exit ", etc.
    if user_input.lower().strip() == 'exit':
        break
    
    # HumanMessage - Creates a message object for user input
    # chat_history.append() - Adds user message to history
    # Why: HumanMessage explicitly marks this as user input (not AI or system)
    #      This is the KEY IMPROVEMENT over plain strings - the model knows who said what
    chat_history.append(HumanMessage(content=user_input))
    
    
    # model.invoke() - Sends entire chat history with proper message types to the model
    # Why: By passing messages with explicit types (System/Human/AI), the model understands:
    #      1. What's the system instruction
    #      2. What the user asked
    #      3. What the AI previously responded
    #      This is much better than 4_chatbot_with_history.py where everything was just strings
    result=model.invoke(chat_history )
    
    
    # AIMessage - Creates a message object for AI's response
    # chat_history.append() - Adds AI response to history
    # Why: AIMessage explicitly marks this as AI output
    #      Maintains proper conversation structure: [System, Human, AI, Human, AI, ...]
    chat_history.append(AIMessage(content=result.content))
    
    
    # print() - Displays the AI's response
    # Why: Shows the user what the AI responded
    print("Ai:",result.content)

# print() - Displays full conversation history with message types
# Why: Shows the structured conversation format with explicit roles
#      Example output: [SystemMessage(...), HumanMessage(...), AIMessage(...), ...]
#      Compare this to 4_chatbot_with_history.py which just showed strings
print(chat_history )