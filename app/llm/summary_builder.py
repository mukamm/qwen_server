from app.llm.client import send_to_ai

def generate_summary(old_summary: str, new_user_message: str, new_ai_response: str) -> str:
    """
    Генерируем сжатое summary с помощью AI.
    """
    prompt = f"""
You are an AI that summarizes chat history for memory purposes.
Old summary:
{old_summary}

New message:
User: {new_user_message}
AI: {new_ai_response}

Update the summary with only key facts, names, important details.
Return concise summary suitable for future conversation.
"""
    new_summary = send_to_ai(prompt)
    return new_summary.strip()
