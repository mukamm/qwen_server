from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.memory.summary_store import get_summary, update_summary

class SummaryUpdate(BaseModel):
    new_summary: str


router = APIRouter(prefix="/summary", tags=["summary"])

@router.get("/{session_id}")
def read_summary(session_id: str):
    """Получить текущий summary сессии"""
    summary = get_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"session_id": session_id, "summary": summary}

@router.post("/{session_id}")
def edit_summary(session_id: str, summary: SummaryUpdate):
    update_summary(session_id, summary.new_summary)
    return {"session_id": session_id, "summary": summary.new_summary}
