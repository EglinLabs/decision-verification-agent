import os
from dotenv import load_dotenv

load_dotenv()
print("OPENROUTER_API_KEY loaded:", bool(os.getenv("OPENROUTER_API_KEY")))
print("OPENROUTER_MODEL:", os.getenv("OPENROUTER_MODEL"))
print("OPENROUTER_BASE_URL:", os.getenv("OPENROUTER_BASE_URL"))