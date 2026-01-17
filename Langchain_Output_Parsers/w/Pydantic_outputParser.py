from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation", 
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace (llm=llm)

class Person(BaseModel):
    name: str = Field(description ="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    place: str = Field(description ="The place where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional {place} person.\n"
        "{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format_prompt(place="Indian")
print(prompt)
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)