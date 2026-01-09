from langchain_core.prompts import ChatPromptTemplate
#Dynamic chat prompt for multi-turn conversations

# ChatPromptTemplate - Builds a chat prompt with role-tagged messages and placeholders
# Why: Lets you define reusable chat prompts that can be parameterized with variables
chat_template = ChatPromptTemplate([
    ("system","You are a helpful {domain} assistant."),
    ("human", "Tell me something about {topic}.")
])

# format_messages() - Fills the placeholders with runtime values and returns message objects
# Why: Produces a list of role-aware messages you can pass to chat models
prompt= chat_template.format_messages(
    domain= "AI",
    topic  = "LangChain vs LangGraph"

)

print(prompt)