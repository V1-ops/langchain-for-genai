from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables (e.g., HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Initialize the LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)

# Wrap the LLM in ChatHuggingFace for chat functionality
model = ChatHuggingFace (llm=llm)

# Create JsonOutputParser - parses JSON strings into Python dictionaries
# DRAWBACK: JsonOutputParser does NOT enforce schema validation
# The model can return any JSON structure - we cannot guarantee specific fields or types
parser = JsonOutputParser()

# Create prompt template with format instructions from the parser
template = PromptTemplate(
    template = "Create a fictional person with name, age, and city. Return ONLY valid JSON, nothing else.\n{format_instructions}",
    input_variables=[],
    partial_variables = {"format_instructions":parser.get_format_instructions()}
)

# Format the prompt
prompt = template.format_prompt()

# Invoke the model with the prompt
result = model.invoke(prompt)
print("Raw output:", result.content)
print("\n" + "="*50 + "\n")

# Parse the JSON string into a Python dictionary
final_result = parser.parse(result.content)
print("Parsed JSON:", final_result)
print("Type of parsed result:", type(final_result))

# Another way to write this using chain 
# chain = template |model | parser
# result = chain.invoke({})
# print(result)


"""
COMPARISON: Raw Output vs Parsed Output

Raw output: 
- Type: STRING (str)
- Format: JSON-formatted string with proper indentation
- Example: '{\n  "name": "Elena Martinez",\n  "age": 29,\n  "city": "Barcelona"\n}'
- Usage: You would need to manually parse it with json.loads() to access individual fields
- Cannot directly access fields: result.content['name'] would cause an error

==================================================

Parsed JSON: 
- Type: DICTIONARY (dict)
- Format: Python dictionary object
- Example: {'name': 'Elena Martinez', 'age': 29, 'city': 'Barcelona'}
- Usage: Directly access fields like final_result['name'], final_result['age']
- Can iterate over keys/values: for key, value in final_result.items()
- Ready to use in your Python code without additional parsing

KEY BENEFIT: JsonOutputParser automatically converts the JSON string into a usable Python dictionary,
saving you from manually calling json.loads() and handling potential parsing errors.
"""
"""Raw output: {
  "name": "Elena Martinez",
  "age": 29,
  "city": "Barcelona"
}

==================================================

Parsed JSON: {'name': 'Elena Martinez', 'age': 29, 'city': 'Barcelona'}
"""

