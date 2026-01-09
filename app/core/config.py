import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
QWEN_HOST = os.getenv("QWEN_HOST", "127.0.0.1")
QWEN_PORT = int(os.getenv("QWEN_PORT", 8001)) 
