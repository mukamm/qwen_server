from app.llm.summary_builder import generate_summary
from app.memory.summary_store import get_summary, update_summary
from app.memory.session_store import get_session, update_session
from app.llm.client import send_to_ai

def process_message(session_id: str, user_message: str) -> str:
    session = get_session(session_id)
    if not session:
        return None

    messages = session.get("messages", [])
    old_summary = get_summary(session_id) or ""  # <- на всякий случай

    # Формируем prompt для AI chat
    history_text = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in messages])
    prompt = f"Summary of previous chat:\n{old_summary}\n\nChat history:\n{history_text}\nUser: {user_message}\nAI:"
    ai_response = send_to_ai(prompt)

    # Сохраняем новые сообщения
    update_session(session_id, user_message, ai_response)

    # Генерируем умное summary через AI
    new_summary = generate_summary(old_summary, user_message, ai_response)
    if new_summary:  # проверка на пустой результат
        update_summary(session_id, new_summary)

    return ai_response
