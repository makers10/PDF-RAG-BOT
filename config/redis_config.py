# config/redis_config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Redis Configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD", None),
    "decode_responses": False,  # We'll use pickle for complex objects
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
}

# Cache TTL Configuration (in seconds)
CACHE_TTL = {
    "embeddings": 86400,  # 24 hours
    "qa_pairs": 86400,    # 24 hours
    "vector_store": 604800,  # 7 days
    "default": 86400,     # 24 hours
}

# Cache Keys Prefix
CACHE_PREFIX = {
    "embeddings": "emb",
    "qa": "qa",
    "vector_store": "vs",
    "chunks": "chunks",
}
