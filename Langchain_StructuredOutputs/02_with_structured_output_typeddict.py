# Using Annotated TypedDict for structured output with detailed field descriptions
# Annotated provides inline documentation for each field in the structured output

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

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

# Create ChatHuggingFace model wrapper with the endpoint
model = ChatHuggingFace(llm=llm)

# Define a TypedDict with Annotated fields for structured output
# Annotated allows adding descriptions/instructions for each field to guide the LLM
# The format is: Annotated[data_type, "instruction string for LLM"]
class Review (TypedDict):
    # List of key themes/topics mentioned in the review with instructions
    # Annotated[list[str], ...] means: extract as a list of strings, and the string is an instruction for the LLM
    key_themes :Annotated[list[str],"Write down all the key themes mentioned in  the review"]
    # Brief summary of the entire review
    # The string "A brief summary..." is a prompt instruction for what the LLM should extract
    summary : Annotated[str,"A brief summary of the review "] 
    # Sentiment classification with specific values
    # The string tells the LLM to classify sentiment as one of three categories
    sentiment : Annotated[str, "A brief sentiment of the review , either Positive , Negative or Neutral"]
    # Optional list of pros (advantages/positives)
    # The instruction guides the LLM to identify and list positive aspects
    pros: Optional[Annotated[list[str],"List down the pros mentioned in the review"]]
    # Optional list of cons (disadvantages/negatives)
    # The instruction guides the LLM to identify and list negative aspects
    cons: Optional[Annotated[list[str],"List down the cons mentioned in the review"]]
    # Optional name of the reviewer/person who wrote the review
    # The instruction tells the LLM to extract the name if mentioned
    name: Optional[Annotated[str,"Name of the person who reviewed the product"]]


# Create a structured model that enforces the Review TypedDict output schema
# This ensures the LLM response matches the defined structure
structured_model= model.with_structured_output(Review)

# Invoke the model with a detailed product review
# The model will parse the review text and extract information into the Review structure
result =structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
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


print(result)
# Print the complete structured result (contains all fields: key_themes, summary, sentiment, pros, cons, name)

print(result['pros'])
# Print just the pros field from the result

print(result['cons'])
# Print just the cons field from the result

print(result['name'])
# Print just the name field from the result


# ===== WHY PYDANTIC IS BETTER THAN TYPEDDICT =====
# While TypedDict with Annotated provides good guidance for LLMs to understand the desired output structure,
# the LLM can still make mistakes and deviate from the defined schema at runtime.
# 
# For example, even though we defined sentiment as a string with instructions "either Positive, Negative or Neutral",
# the LLM might return "Somewhat Positive" or "Mixed" instead of the exact values.
# Similarly, if pros/cons should be lists, the LLM might return a single string or a dict instead.
#
# This is where Pydantic comes in:
# - Pydantic enforces strict type validation and converts responses to the exact specified types
# - It validates the response structure at runtime and raises errors if the LLM deviates
# - It can apply custom validators and transformations to ensure data integrity
# - It provides better error messages when the LLM produces invalid data
# 
# In summary: TypedDict = guidance (soft constraints), Pydantic = enforcement (hard constraints)
