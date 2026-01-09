# Purely hugging face implementation of chat model inference
'''
Provider: Hugging Face

API: Hugging Face Inference API

Model: Qwen 2.5 (72B Instruct)

Abstraction:  No LangChain
'''
# InferenceClient - Direct Hugging Face API client without LangChain wrapper
# Why: Provides direct access to HF Inference API with more control and less abstraction
#      Useful when you want to use pure HF without LangChain dependencies
from huggingface_hub import InferenceClient

# load_dotenv() - Loads environment variables from .env file
# Why: Securely manages API tokens without hardcoding them
from dotenv import load_dotenv

# os - Operating system interface for environment variable access
# Why: Used to retrieve the HF_TOKEN from environment variables
import os

# load_dotenv() - Executes loading of environment variables
# Why: Makes HF_TOKEN available in the environment for authentication
load_dotenv()

# os.getenv() - Retrieves environment variable value
# Why: Gets the Hugging Face API token needed for authentication with the Inference API
# Note: HF_TOKEN must be set in .env file
hf_token = os.getenv("HF_TOKEN")

# InferenceClient() - Initializes the Hugging Face Inference client
# Why: Creates authenticated connection to HF Inference API for making model requests
# Parameters:
#   - token: Authentication token for API access
client = InferenceClient(token=hf_token)

# Variable to store user query
# Why: Separates the question from the API call for better code organization
question = "What is the capital of India ?"


# client.chat_completion() - Makes a chat completion request to the model
# Why: Sends messages to the model and gets conversational responses
# This is the core function that communicates with the LLM
# Parameters:
#   - messages: List of message dicts with 'role' and 'content' (OpenAI-compatible format)
#   - model: Specifies which HF model to use (Qwen 2.5 72B for high-quality responses)
#   - max_tokens: Limits response length to 256 tokens for faster responses
response = client.chat_completion(
    messages=[
        {"role": "user", "content": question}
    ],
    model="Qwen/Qwen2.5-72B-Instruct",
    max_tokens=256
)

# response.choices[0].message.content - Extracts the actual text response
# Why: The API returns a complex response object; we need to navigate to the content
#      - choices[0]: Gets the first (and typically only) response option
#      - .message: Accesses the message object
#      - .content: Gets the actual text content of the response
answer = response.choices[0].message.content

# print() - Displays the answer to the user
# Why: Shows the model's response in a formatted way
print(f"Answer: {answer}")
