'''
Provider: Hugging Face

Framework: LangChain

Model: Qwen 2.5

Abstraction:  LangChain ChatModel
'''
# load_dotenv() - Used to load environment variables from .env file
# Why: Securely manages API keys and credentials without hardcoding them in the code
from dotenv import load_dotenv

# ChatHuggingFace - LangChain wrapper for Hugging Face chat models
# Why: Provides a standardized interface for chat-based interactions
# HuggingFaceEndpoint - Connects to Hugging Face Inference API endpoints
# Why: Enables remote model inference without downloading large models locally
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# load_dotenv() - Loads environment variables from .env file into the environment
# Why: Makes API keys available to the application securely
load_dotenv()  

# HuggingFaceEndpoint() - Creates a connection to the Hugging Face Inference API
# Why: Allows us to use powerful models hosted on Hugging Face servers
# Parameters:
#   - repo_id: Specifies which model to use (Qwen 2.5 72B Instruct model)
#   - task: Defines the task type (text-generation for chat/completion)
#   - temperature: Controls randomness (0.7 for balanced creativity/coherence)
#   - max_new_tokens: Limits response length (256 tokens for concise answers)
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
)

# ChatHuggingFace() - Wraps the HuggingFace endpoint for chat-based interactions
# Why: Provides a LangChain-compatible interface with chat-specific methods like invoke()
model = ChatHuggingFace(llm=llm)

# model.invoke() - Sends a prompt to the model and gets a response
# Why: This is the main method to interact with the chat model for one-off queries
# Returns: A message object containing the model's response
result = model.invoke("Write a short note on LangChain.")

# result.content - Extracts the text content from the response message
# Why: The invoke() method returns a message object, we need .content to get the actual text
print(result.content)