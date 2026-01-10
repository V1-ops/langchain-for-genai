"""
Structured Output — Overview

What it is:
- Structured output is a model response that follows a predefined schema
	(e.g., a dict/JSON with specific keys and types). Instead of free-form
	text, the model returns machine-readable data that can be reliably
	parsed and consumed by downstream code.

Why it’s needed:
- Reliability: predictable fields allow safe parsing and fewer edge cases.
- Validation: schemas enable automatic checking for required fields/types.
- Integration: APIs, UIs, and pipelines expect well-formed data structures.
- Traceability: clearer contracts between prompts and application logic.
 - Interoperability: without structured outputs, interaction is largely AI ↔ human;
	 with structured outputs, AI can drive databases, ETL pipelines, microservices,
	 schedulers, and automated workflows programmatically.

Model support note:
- Some LLMs can natively produce structured outputs when guided by schemas
	or response-format tools; others may not and will require post-processing
	(parsing/extraction) from free text.
- In this section, we focus on models that can generate structured outputs
	directly. Models that cannot will be handled in the next section.

Three ways to define structured output in Python/LangChain:
1) TypedDict (typing.TypedDict)
	 - Define a dictionary-like schema with required/optional keys and types.
	 - Good for lightweight contracts without runtime validation overhead.

2) Pydantic (pydantic.BaseModel)
	 - Define a class with typed fields; get validation, defaults, and
		 serialization/deserialization out of the box.
	 - Ideal when you want strong runtime validation and clearer error messages.

3) JSON Schema (dict following the JSON Schema spec)
	 - Provide a JSON-compatible schema describing properties, types, and
		 constraints. Many LLM providers and tools can use it to steer outputs.
	 - Useful for interoperability and when the target system expects JSON
		 Schema (e.g., API contracts, validation libraries).

# Implementation examples and model-specific usage will follow below in code.
# For models without native structured output support, the next section will
# demonstrate robust parsing strategies and validation fallbacks.
"""

