from app.memory.redis_client import r

SUMMARY_PREFIX = "session_summary:"

def get_summary(session_id: str) -> str:
    """Получаем текущее summary сессии"""
    data = r.get(f"{SUMMARY_PREFIX}{session_id}")
    if not data:
        return ""
    return data

def update_summary(session_id: str, new_summary: str) -> None:
    """Обновляем summary сессии"""
    r.set(f"{SUMMARY_PREFIX}{session_id}", new_summary)
