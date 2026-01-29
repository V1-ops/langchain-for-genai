from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="Tell me a joke about {text}",
    input_variables=["text"]

)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=256
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = RunnableSequence(prompt, model, parser)
result = chain.invoke({"text": "Tree"})
print(result)