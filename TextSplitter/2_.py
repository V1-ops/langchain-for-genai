# text-structured Based Text Splitting
from langchain_text_splitters import  RecursiveCharacterTextSplitter
text = """Education is the process of acquiring knowledge, skills, values, and habits that help individuals grow intellectually, socially, and emotionally. It is not limited to schools and textbooks; learning happens through experiences, interactions, observation, and self-reflection. Education shapes the way a person thinks, solves problems, and understands the world. It builds the foundation for communication, decision-making, creativity, and critical thinking, which are essential for personal development and responsible citizenship.

Beyond individual growth, education plays a major role in the progress of society. An educated population leads to innovation, economic development, social equality, and better governance. It helps reduce poverty, improves health awareness, and promotes tolerance and understanding among diverse communities. In simple words, education empowers people to improve their own lives and contribute positively to the world around them.

"""

splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])

result = splitter.split_text(text)
print(result)