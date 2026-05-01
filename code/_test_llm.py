import os
from llm_client import generate_text

print("Testing LLM...")
res = generate_text("Say hello")
print(f"Result: {res}")
