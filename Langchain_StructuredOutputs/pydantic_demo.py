# Import Pydantic components for data validation and type checking
from pydantic import BaseModel, EmailStr, Field, Optional

# Define a Student model using Pydantic BaseModel
# BaseModel provides automatic type validation, conversion, and serialization
class Student(BaseModel):
    # String field with default value - if not provided, uses "Vanshdeep Singh"
    name: str = "Vanshdeep Singh"
    
    # Optional integer field - can be None or an integer value
    age: Optional[int] = None
    
    # EmailStr field - automatically validates email format (requires email-validator package)
    # This field is required (no default value provided)
    email: EmailStr
    
    # Float field with constraints using Field()
    # gt=0.0 means greater than 0.0, lt=10.0 means less than 10.0
    # So CGPA must be between 0 and 10
    cgpa: float = Field(gt=0.0, lt=10.0, description="The float number will represent the CGPA of the student on a scale of 10")

# Create a dictionary with student data
# Note: name is not provided, so it will use the default value
new_student = {'age': 20, 'email': "abc@example.com", 'cgpa': 9.0}

# Create a Student instance by unpacking the dictionary using **new_student
# Pydantic will:
# 1. Validate email format using EmailStr
# 2. Validate CGPA is between 0 and 10 using Field constraints
# 3. Use default name since it wasn't provided
# 4. Convert age to int type
student = Student(**new_student)

# Print the student object - shows all validated and converted data
print(student)

