import redis
import orjson
import logging
import time
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = "192.168.3.98"
REDIS_PORT = 6379
REDIS_PASSWORD = "Myl@yymM1112"
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
    socket_keepalive=True
)


# ---------------- Safe Redis Operations ---------------- #

def safe_redis_get(key: str, default=None, retries: int = REDIS_MAX_RETRIES):
    for attempt in range(retries):
        try:
            data = r.get(key)
            if data is not None and isinstance(data, bytes):
                return data.decode('utf-8')
            return data if data else default
        except redis.ConnectionError as e:
            logger.warning(f"Redis GET connection error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis GET failed after {retries} attempts for key: {key}")
                return default
        except Exception as e:
            logger.error(f"Redis GET unexpected error: {e}")
            return default


def safe_redis_set(key: str, value: str, ex: int = None, retries: int = REDIS_MAX_RETRIES) -> bool:
    for attempt in range(retries):
        try:
            r.set(key, value, ex=ex)
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis SET connection error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis SET failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis SET unexpected error: {e}")
            return False


def safe_redis_hgetall(key: str, default: Dict = None, retries: int = REDIS_MAX_RETRIES):
    for attempt in range(retries):
        try:
            data = r.hgetall(key)
            if data:
                # Декодируем ключи и значения
                return {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in data.items()
                }
            else:
                return default or {}
        except redis.ConnectionError as e:
            logger.warning(f"Redis HGETALL connection error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis HGETALL failed after {retries} attempts for key: {key}")
                return default or {}
        except Exception as e:
            logger.error(f"Redis HGETALL unexpected error: {e}")
            return default or {}


def safe_redis_hset(key: str, mapping: Dict, ex: int = None, retries: int = REDIS_MAX_RETRIES) -> bool:
    for attempt in range(retries):
        try:
            r.hset(key, mapping=mapping)
            if ex:
                r.expire(key, ex)
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis HSET connection error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis HSET failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis HSET unexpected error: {e}")
            return False


def safe_redis_delete(key: str, retries: int = REDIS_MAX_RETRIES) -> bool:
    for attempt in range(retries):
        try:
            result = r.delete(key)
            return result > 0
        except redis.ConnectionError as e:
            logger.warning(f"Redis DELETE connection error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis DELETE failed after {retries} attempts for key: {key}")
                return False
        except Exception as e:
            logger.error(f"Redis DELETE unexpected error: {e}")
            return False


def safe_redis_pipeline_execute(operations: List[Tuple], retries: int = REDIS_MAX_RETRIES):
    for attempt in range(retries):
        try:
            pipe = r.pipeline()
            for op, *args in operations:
                getattr(pipe, op)(*args)
            return pipe.execute()
        except redis.ConnectionError as e:
            logger.warning(f"Redis pipeline error ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(REDIS_RETRY_DELAY)
            else:
                logger.error(f"Redis pipeline failed after {retries} attempts")
                return [None] * len(operations)
        except Exception as e:
            logger.error(f"Redis pipeline unexpected error: {e}")
            return [None] * len(operations)


# ---------------- JSON helpers ---------------- #

def json_encode(data: Any) -> str:
    return orjson.dumps(data).decode('utf-8')


def json_decode(data: str) -> Any:
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None


# ---------------- Rate Limit ---------------- #

def check_rate_limit(user_id: str, session_id: str, max_requests: int, window: int) -> bool:
    key = f"{RATE_LIMIT_PREFIX}{user_id}:{session_id}"
    try:
        count = r.incr(key)
        if count == 1:
            r.expire(key, window)
        if count > max_requests:
            logger.warning(f"Rate limit exceeded: {user_id}:{session_id} ({count}/{max_requests})")
            return False
        return True
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        return True  # fail open


# ---------------- Test Connection ---------------- #

if __name__ == "__main__":
    try:
        if r.ping():
            print("Redis подключён ✅")
    except Exception as e:
        print("Ошибка подключения к Redis:", e)
