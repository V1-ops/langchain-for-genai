# ===== PYDANTIC BASEMODEL SYNTAX FOR STRUCTURED OUTPUT =====
# This file demonstrates the CORRECT Pydantic syntax for structured outputs
# However, this approach is NOT APPLICABLE for HuggingFace models!
# HuggingFace only supports TypedDict, not Pydantic BaseModel
# This code will work with: OpenAI (ChatOpenAI), Anthropic (ChatAnthropic), etc.
# This code will FAIL with: HuggingFace (ChatHuggingFace) ❌

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field  # BaseModel: provides runtime type validation
from typing import Optional, Literal   # Optional: can be None, Literal: restricted values 


# Load environment variables from .env file (for API keys and credentials)
load_dotenv()

# Create HuggingFace endpoint with model configuration
# This initializes the Qwen2.5-72B model with specified parameters
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # Model repository ID
    task="text-generation",  # Task type for the model
    temperature=0.5,  # Controls randomness (0.5 = moderate randomness)
    max_new_tokens=256,  # Maximum number of tokens to generate
)

model = ChatHuggingFace(llm=llm)

# ===== PYDANTIC SYNTAX EXPLANATION =====
# BaseModel: Pydantic class that provides runtime type validation and conversion
# Field(): Allows adding metadata, descriptions, constraints, and defaults
# This syntax works with OpenAI, Anthropic, etc. but NOT with HuggingFace

class Review(BaseModel):
    # Field with description - tells the LLM what to extract
    # list[str]: expects a list of strings
    key_themes: list[str] = Field(description="Write down all the key themes mentioned in the review")
    
    # Simple string field with description
    summary: str = Field(description="A brief summary of the review")
    
    # Literal type: restricts values to only "pos" or "neg"
    # This enforces type checking at runtime
    sentiment: Literal["pos", "neg"] = Field(description="A brief sentiment of the review, either pos or neg")

    # Optional field: can be None (default value provided)
    # Field(default=None) means this field is optional
    pros: Optional[list[str]] = Field(default=None, description="List down the pros mentioned in the review")
    
    # Optional list of strings
    cons: Optional[list[str]] = Field(default=None, description="List down the cons mentioned in the review")
    
    # Optional string field
    name: Optional[str] = Field(default=None, description="Name of the person who reviewed the product")

# ===== IMPORTANT NOTE =====
# The above Pydantic syntax is correct and follows best practices
# HOWEVER, this code will FAIL when run because:
# - HuggingFace's ChatHuggingFace does NOT support Pydantic BaseModel
# - HuggingFace only accepts TypedDict for structured output
# - You will get: NotImplementedError: Pydantic schema is not supported for function calling
# 
# To use Pydantic BaseModel, switch to a provider that supports it:
# - from langchain_openai import ChatOpenAI (use gpt-4o or gpt-4-turbo)
# - from langchain_anthropic import ChatAnthropic (use claude-3 models)


# This line will throw an error when using HuggingFace:
# NotImplementedError: Pydantic schema is not supported for function calling
structured_model = model.with_structured_output(Review)  # ❌ NOT SUPPORTED by HuggingFace

# Attempt to invoke - this will fail with HuggingFace
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:

Insanely powerful processor (great for gaming and productivity)

Stunning 200MP camera with incredible zoom capabilities

Long battery life with fast charging

S-Pen support is unique and useful

Cons:

Bulky and heavy—not great for one-handed use

Bloatware still exists in One UI

Expensive compared to competitors
Reviewed by Vanshdeep singh                             """)


print(result.name)

