from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from dotenv import load_dotenv
import os 
load_dotenv()
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model for local execution
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 256, "do_sample": False},
)

model = ChatHuggingFace(llm=llm)
question = "What is the capital of India ?"
answer =model.invoke(question)
print(answer)