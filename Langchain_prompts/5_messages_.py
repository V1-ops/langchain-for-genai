# ChatHuggingFace - LangChain wrapper for HF chat models
# HuggingFaceEndpoint - Connects to Hugging Face Inference API
# Why: Enables structured chat with proper message handling
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# SystemMessage, HumanMessage, AIMessage - Explicit message type classes
# Why: Differentiates between system instructions, user input, and AI responses
#      Better than plain strings as it makes conversation structure clear to the model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

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
#   - max_new_tokens: 256 limits response length
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256

)

# ChatHuggingFace() - Wraps the endpoint for chat-based interactions
# Why: Provides invoke() method and handles message objects properly
model = ChatHuggingFace(llm=llm)

# messages - List of message objects representing the conversation
# Why: Using SystemMessage/HumanMessage instead of strings provides:
#      1. Clear differentiation between system prompts and user queries
#      2. Better context understanding for the model
#      3. Proper conversation structure for multi-turn chats
messages=[
# SystemMessage - Sets the AI's behavior/personality
# Why: Defines how the assistant should respond throughout the conversation
SystemMessage(content='You are a helpful assistant.'),
# HumanMessage - Represents the user's question
# Why: Clearly marks this as user input, helping the model understand its role
HumanMessage(content='Explain about Lanchain in brief.')
]

# model.invoke() - Sends structured messages to the model
# Why: Passing message objects (not plain text) enables better context awareness
result=model.invoke(messages)

# AIMessage - Creates a message object for the AI's response
# messages.append() - Adds AI response to conversation history
# Why: Maintains structured history so the model can reference previous turns
#      result.content extracts the actual text from the response object
messages.append(AIMessage(content=result.content))

# print() - Displays the complete conversation with message types
# Why: Shows the structured format: [SystemMessage, HumanMessage, AIMessage]
print(messages)
