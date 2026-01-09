from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.chat_service import process_message

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        response = process_message(req.session_id, req.message)
        return {"response": response}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
