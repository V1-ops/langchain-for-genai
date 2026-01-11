from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id ="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace (llm=llm)
# 1st prompt  
template1 = PromptTemplate(
    template = "Make a detailed report on the {topic}",
    input_variables=["topic"]

)

# 2nd prompt 
template2 = PromptTemplate(
    template = "write a 5 line summary of the following report: {report}",
    input_variables=["report"]

)

prompt1 = template1.format_prompt(topic = "GenAI")

result1 = model.invoke(prompt1)
prompt2 = template2.format_prompt (report = result1.content)
result2 = model.invoke (prompt2)
print("Detailed Report:\n", result1.content)
print("\n5 Line Summary:\n", result2.content)
