from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template= "Tell five interesting facts about {topic}.",
    input_variables = ["topic"]
)
parser = StrOutputParser()
chain = prompt | model | parser
result=chain.invoke({"topic": "Paris "})
print(result)
chain.get_graph().print_ascii()

""" +-------------+       
     | PromptInput |
     +-------------+
            *
            *
            *
    +----------------+
    | PromptTemplate |
    +----------------+
            *
            *
            *
   +-----------------+
   | ChatHuggingFace |
   +-----------------+
            *
            *
            *
   +-----------------+
   | StrOutputParser |
   +-----------------+
            *
            *
            *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
"""
