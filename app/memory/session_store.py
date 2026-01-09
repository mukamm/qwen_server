from app.memory.redis_client import r
import uuid

SESSION_PREFIX = "session:"

def create_session():
    session_id = str(uuid.uuid4())
    r.hset(f"{SESSION_PREFIX}{session_id}", mapping={"messages": "[]"})
    return session_id

def get_session(session_id):
    data = r.hgetall(f"{SESSION_PREFIX}{session_id}")
    return data
