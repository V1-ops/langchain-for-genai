from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
class feedback(BaseModel):
    sentiment :Literal['Positive' ,'Negative']= Field(description="Classify the sentiment of the feedback into Positive or Negative")
parser2 = PydanticOutputParser(pydantic_object= feedback)
prompt1 = PromptTemplate(
    template = """Classify the sentiment of the following feedback as either Positive or Negative.

Text: {feedback}

Return your answer in the following JSON format:
{format_instructions}

Only return valid JSON, nothing else.""",
    input_variables = ["feedback"],
    partial_variables = {"format_instructions": parser2.get_format_instructions()}
)
classification_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response for this positive review:\n{feedback}",
    input_variables = ["feedback"]
    
)
prompt3 = PromptTemplate(
    template = "Write an appropriate response for this negative review:\n{feedback}",
    input_variables = ["feedback"]
    
)
Branched_chain = RunnableBranch(
    (lambda x:x.sentiment=='Positive', prompt2 | model | parser),
    (lambda x:x.sentiment=='Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Sentiment not found")
)
chain = classification_chain | Branched_chain
result=chain.invoke({"feedback": "The product quality is excellent and exceeded my expectations!"})
print(result)
chain.get_graph().print_ascii()
"""Here's a possible response:

"Thank you so much for taking the time to share your wonderful experience with us! We're thrilled to hear that you enjoyed [service/product] and appreciate your kind words. We're committed to providing the best possible service, and your feedback means a lot to us. We look forward to serving you again in the future!"
    +-------------+      
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
+----------------------+
| PydanticOutputParser |
+----------------------+
            *
            *
            *
       +--------+
       | Branch |
       +--------+
            *
            *
            *
    +--------------+
    | BranchOutput |
    +--------------+
"""
