from langchain_core.prompts import PromptTemplate
# prompt template
template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
    input_variables=["paper_input", "style_input", "length_input"]
    #This list tells PromptTemplate: "These are the placeholders I'm using in my template string"
)

template.save('template.json')  #Saves the prompt template to a JSON file