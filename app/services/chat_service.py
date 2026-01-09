from app.llm.client import send_to_ai
from app.memory.session_store import get_session, update_session

def process_message(session_id: str, message: str):
    # Получаем предыдущие сообщения
    session = get_session(session_id)
    history = session.get("messages", "[]")  # строки JSON
    # Формируем prompt для AI
    prompt = f"{history}\nUser: {message}\nAI:"
    # Отправляем в llama-server (Qwen)
    ai_response = send_to_ai(prompt)
    # Обновляем историю
    update_session(session_id, message, ai_response)
    return ai_response
