import redis
from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)
