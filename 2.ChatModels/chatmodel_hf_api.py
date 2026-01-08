from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()
# Get the Hugging Face token from environment
hf_token = os.getenv("HF_TOKEN")

# Initialize the Inference Client
client = InferenceClient(token=hf_token)

# Ask the question
question = "What is the capital of India ?"


# Use chat completion with Qwen model
response = client.chat_completion(
    messages=[
        {"role": "user", "content": question}
    ],
    model="Qwen/Qwen2.5-72B-Instruct",
    max_tokens=256
)
answer = response.choices[0].message.content
print(f"Answer: {answer}")
