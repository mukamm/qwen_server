import redis
import orjson
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_DB = 0

# Redis prefixes
SESSION_PREFIX = "session:"
SUMMARY_PREFIX = "summary:"
USER_PROFILE_PREFIX = "user_profile:"
DIALOG_STATE_PREFIX = "dialog_state:"
RATE_LIMIT_PREFIX = "ratelimit:"

# TTL
SESSION_TTL = 86400 * 7  # 7 days

# Retry config
REDIS_MAX_RETRIES = 3
REDIS_RETRY_DELAY = 0.5

# Redis connection
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
    retry_on_timeout=True,
    health_check_interval=30
)


def safe_redis_get(key: str, default=None, retries: int = REDIS_MAX_RETRIES):
    """Safe Redis GET with retry logic"""
    for attempt in range(retries):
        try:
            data = r.get(key)
            return data if data else default
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis GET failed after {retries} attempts for key: {key}")
                return default
        except Exception as e:
            logger.error(f"Redis GET unexpected error: {e}")
            return default


def safe_redis_set(key: str, value: str, ex: int = None, retries: int = REDIS_MAX_RETRIES) -> bool:
    """Safe Redis SET with retry logic"""
    for attempt in range(retries):
        try:
            r.set(key, value, ex=ex)
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis SET failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis SET unexpected error: {e}")
            return False


def safe_redis_hgetall(key: str, default: Dict = None, retries: int = REDIS_MAX_RETRIES):
    """Safe Redis HGETALL with retry logic"""
    for attempt in range(retries):
        try:
            data = r.hgetall(key)
            return data if data else (default or {})
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis HGETALL failed after {retries} attempts for key: {key}")
                return default or {}
        except Exception as e:
            logger.error(f"Redis HGETALL unexpected error: {e}")
            return default or {}


def safe_redis_hset(key: str, mapping: Dict, ex: int = None, retries: int = REDIS_MAX_RETRIES) -> bool:
    """Safe Redis HSET with retry logic"""
    for attempt in range(retries):
        try:
            r.hset(key, mapping=mapping)
            if ex:
                r.expire(key, ex)
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis HSET failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis HSET unexpected error: {e}")
            return False


def safe_redis_delete(key: str, retries: int = REDIS_MAX_RETRIES) -> bool:
    """Safe Redis DELETE with retry logic"""
    for attempt in range(retries):
        try:
            result = r.delete(key)
            return result > 0
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis DELETE failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis DELETE unexpected error: {e}")
            return False


def safe_redis_pipeline_execute(operations: List[Tuple], retries: int = REDIS_MAX_RETRIES):
    """
    Safe Redis pipeline execution
    operations: List of tuples like [('get', 'key1'), ('hgetall', 'key2')]
    """
    for attempt in range(retries):
        try:
            pipe = r.pipeline()
            for op, *args in operations:
                getattr(pipe, op)(*args)
            return pipe.execute()
        except redis.ConnectionError as e:
            logger.warning(f"Redis pipeline error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis pipeline failed after {retries} attempts")
                return [None] * len(operations)
        except Exception as e:
            logger.error(f"Redis pipeline unexpected error: {e}")
            return [None] * len(operations)


def json_encode(data: Any) -> str:
    """Encode data to JSON string using orjson"""
    return orjson.dumps(data).decode('utf-8')


def json_decode(data: str) -> Any:
    """Decode JSON string using orjson"""
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None


def check_rate_limit(user_id: str, session_id: str, max_requests: int, window: int) -> bool:
    """
    Rate limiting in Redis (multi-instance safe)
    """
    key = f"{RATE_LIMIT_PREFIX}{user_id}:{session_id}"
    
    try:
        count = r.incr(key)
        
        # Set expiry only on first increment
        if count == 1:
            r.expire(key, window)
        
        if count > max_requests:
            logger.warning(f"Rate limit exceeded: {user_id}:{session_id} ({count}/{max_requests})")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        # On error, allow request (fail open)
        return True