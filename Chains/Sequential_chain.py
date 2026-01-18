from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint 
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

prompt1 = PromptTemplate(
    template = "Give a detailed explanation about {topic}.",
    input_variables = ["topic"]
)
prompt2= PromptTemplate(
    template = "Generate a 5 pointer summary about the following text:\n{text}",
    input_variables = ["text"]
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 
result = chain.invoke({"topic":"Sunlight"})
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
