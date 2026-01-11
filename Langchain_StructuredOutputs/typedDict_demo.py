# Import TypedDict for creating type-hinted dictionary structures
from typing import TypedDict

# Define a Person TypedDict with name (string) and age (integer) fields
class Person (TypedDict):
    name: str
    age: int
    
# Create a person object with correct types (name as string, age as integer)
new_person = Person(name="Vanshdeep", age=20)
print(new_person)  # Output: {'name': 'Vanshdeep', 'age': 20}

# Create another person object with incorrect age type (age as string instead of int)
new_person = Person(name="Vanshdeep", age='20')
print(new_person)  # Output: {'name': 'Vanshdeep', 'age': '20'}

# Note: This will raise a mypy type error since age is expected to be an int, not str
# However, Python still runs the code since TypedDict is not enforced at runtime