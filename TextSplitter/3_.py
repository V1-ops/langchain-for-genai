# Document structure Based Text Splitting
from langchain_text_splitters import  RecursiveCharacterTextSplitter,Language
text = """class Student:
    def __init__(self, name, student_class, marks):
        self.name = name
        self.student_class = student_class
        self.marks = marks
        self.gradepoint = self.calculate_gradepoint()

    def calculate_gradepoint(self):
        if self.marks >= 90:
            return 10
        elif self.marks >= 80:
            return 9
        elif self.marks >= 70:
            return 8
        elif self.marks >= 60:
            return 7
        elif self.marks >= 50:
            return 6
        else:
            return 5

    def get_details(self):
        print("Student Name:", self.name)
        print("Class:", self.student_class)
        print("Marks:", self.marks)
        print("Grade Point:", self.gradepoint)


# Example usage
s1 = Student("Rahul", "10th", 85)
s1.get_details()

"""

splitter = RecursiveCharacterTextSplitter.from_language(language =Language.PYTHON,chunk_size=300, chunk_overlap=0)

result = splitter.split_text(text)
print(result[0])