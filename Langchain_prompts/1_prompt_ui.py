from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st 

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=50,
)
model = ChatHuggingFace(llm=llm)

st.header('Research Assistant')

user_input=st.text_input('Enter your Query...')
if st.button('Summarize'):
    result=model.invoke(user_input)
    st.write(result.content)
# Static Prompt example 
