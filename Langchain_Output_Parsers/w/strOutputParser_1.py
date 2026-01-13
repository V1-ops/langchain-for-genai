from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = template1 | model | parser |template2 | model | parser 

result=chain.invoke({"topic": "Dussehra Festival"})
print(result)


"""Without using String Output Parser
content='Dussehra, also known as Vijayadashami, is a major Hindu festival...' additional_kwargs={'id': 'chatcmpl-123', 'model': 'Qwen/Qwen2.5-72B-Instruct', 'object': 'chat.completion', 'usage': {'completion_tokens': 145, 'prompt_tokens': 234, 'total_tokens': 379}} response_metadata={'token_usage': {'completion_tokens': 145, 'prompt_tokens': 234, 'total_tokens': 379}, 'model_name': 'Qwen/Qwen2.5-72B-Instruct', 'finish_reason': 'stop'} id='run-abc123-xyz'
"""

"""
Using String Output Parser
Dussehra, or Vijayadashami, is a major Indian festival celebrating the triumph of good over evil. It commemorates Lord Rama's victory over the demon king Ravana and Goddess Durga's defeat of the buffalo demon Mahishasura. The festival is observed with great fervor and devotion across India, marking the tenth day of the month of Ashwin. It symbolizes the ultimate victory of righteousness and is steeped in rich historical and mythological significance.


"""
