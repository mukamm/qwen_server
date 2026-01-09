from app.memory.redis_client import r
import uuid
import json

SESSION_PREFIX = "session:"

def create_session():
    session_id = str(uuid.uuid4())
    r.hset(f"{SESSION_PREFIX}{session_id}", mapping={"messages": json.dumps([])})
    return session_id

def get_session(session_id):
    data = r.hgetall(f"{SESSION_PREFIX}{session_id}")
    if not data:
        return None
    # Преобразуем JSON строку обратно в список
    if "messages" in data:
        data["messages"] = json.loads(data["messages"])
    return data

def update_session(session_id, user_message, ai_message):
    """
    Добавляет новое сообщение и ответ AI в историю сессии
    """
    session = get_session(session_id)
    if not session:
        return None
    messages = session.get("messages", [])
    messages.append({"user": user_message, "ai": ai_message})
    r.hset(f"{SESSION_PREFIX}{session_id}", "messages", json.dumps(messages))
    return True
