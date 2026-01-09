from app.llm.client import send_to_ai
from app.memory.session_store import get_session, update_session
from app.memory.summary_store import get_summary, update_summary

def generate_prompt(summary: str, history_messages: list, user_message: str) -> str:
    """
    Формируем prompt для AI с учётом summary и новых сообщений
    """
    history_text = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in history_messages])
    prompt = f"Summary of previous chat:\n{summary}\n\nChat history:\n{history_text}\nUser: {user_message}\nAI:"
    return prompt

def process_message(session_id: str, user_message: str) -> str:
    session = get_session(session_id)
    if not session:
        return None
    messages = session.get("messages", [])
    summary = get_summary(session_id)

    # Формируем prompt для AI
    prompt = generate_prompt(summary, messages, user_message)
    ai_response = send_to_ai(prompt)

    # Сохраняем новые сообщения
    update_session(session_id, user_message, ai_response)

    # Обновляем summary (пока простая стратегия: можно слать в AI для генерации нового summary)
    # Простейший вариант — берём последние N сообщений
    new_summary = f"{summary}\nUser: {user_message}\nAI: {ai_response}"
    update_summary(session_id, new_summary)

    return ai_response
