from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256
)

model = ChatHuggingFace(llm=llm)

st.header('Research Assistant')

paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)
# Load the prompt template from the JSON file
template= load_prompt('template.json')


# renaming for clarity 
selected_paper = paper_input
selected_style = style_input
selected_length = length_input

prompt = template.format(
    paper_input=selected_paper, 
    style_input=selected_style,   
    length_input=selected_length  
    # LEFT SIDE (keyword argument) = placeholder name in template string
    # RIGHT SIDE (value) = Python variable holding actual data

)

if st.button('Generate Explanation'):
    result = model.invoke(prompt)
    st.write(result.content)     

#refer 2.1 for any variable related doubt 
