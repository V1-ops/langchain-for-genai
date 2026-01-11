# Import required modules for LangChain and HuggingFace integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict 

# Load environment variables from .env file (for API keys and credentials)
load_dotenv()

# Create HuggingFace endpoint with model configuration
# This initializes the Qwen2.5-72B model with specified parameters
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # Model repository ID
    task="text-generation",                # Task type for the model
    temperature=0.5,                       # Controls randomness (0.5 = moderate randomness)
    max_new_tokens=256,                    # Maximum number of tokens to generate
)

# Create ChatHuggingFace model wrapper with the endpoint
model = ChatHuggingFace(llm=llm)

# Define a TypedDict for structured output
# This specifies the structure of the response we want from the LLM
class Review(TypedDict):
    summary: str  # Brief summary of the review
    sentiment: str  # Sentiment classification (e.g., Positive, Negative, Neutral)

# Create a structured model that enforces the Review TypedDict output schema
structured_model = model.with_structured_output(Review)

# Invoke the model with sample review text
# The model will parse the input and return a structured response matching the Review schema
result = structured_model.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.")

# Print the structured result
print(result)