import os
from dotenv import load_dotenv
load_dotenv()
pk = os.getenv("LANGFUSE_PUBLIC_KEY")
sk = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_BASE_URL")
print(f"PK: |{pk}|")
print(f"SK: |{sk}|")
print(f"Host: |{host}|")
