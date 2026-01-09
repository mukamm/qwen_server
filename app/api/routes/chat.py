from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.memory.session_store import get_session

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Пока просто эхо
    return {"response": f"Echo: {req.message}"}
