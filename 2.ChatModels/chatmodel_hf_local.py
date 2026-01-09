# ChatHuggingFace - LangChain wrapper for Hugging Face chat models
# Why: Provides standardized LangChain interface for chat interactions
# HuggingFacePipeline - Enables running HF models locally using transformers pipeline
# Why: Allows loading and running models on your local machine instead of using API
#      Useful for offline usage, privacy, or avoiding API costs
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

# load_dotenv() - Loads environment variables from .env file
# Why: Even for local models, may need tokens for downloading model weights from HF Hub
from dotenv import load_dotenv

# os - Operating system interface
# Why: Can be used for environment variables or file system operations
import os 

# load_dotenv() - Executes loading of environment variables
# Why: Makes HF token available if needed for model downloads
load_dotenv()

# HuggingFacePipeline.from_model_id() - Downloads and loads a model locally
# Why: Creates a local inference pipeline that runs the model on your machine
#      This is different from API-based approaches - model runs entirely locally
# Parameters:
#   - model_id: "Qwen2.5-0.5B-Instruct" is a smaller model (0.5B parameters)
#     Why: Smaller models can run on consumer hardware without GPU requirements
#          72B model would require significant GPU memory
#   - task: "text-generation" enables text completion/chat capabilities
#   - pipeline_kwargs: Additional parameters for the transformers pipeline
#     * max_new_tokens: 256 - Limits response length
#     * do_sample: False - Uses greedy decoding for deterministic outputs
#       Why: False gives consistent answers, True adds randomness
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model for local execution
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 256, "do_sample": False},
)

# ChatHuggingFace() - Wraps the local pipeline for chat-based interactions
# Why: Converts the raw text generation pipeline into a chat interface
#      Provides methods like invoke() for easier interaction
model = ChatHuggingFace(llm=llm)

# Variable to store the user's question
# Why: Separates data from logic for better code organization
question = "What is the capital of India ?"

# model.invoke() - Sends question to the locally running model
# Why: Main method to get responses from the model
#      Unlike API calls, this runs entirely on your local machine
answer = model.invoke(question)

# print() - Displays the model's response
# Why: Shows the complete response object (includes content and metadata)
print(answer)