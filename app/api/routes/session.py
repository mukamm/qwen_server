from fastapi import APIRouter
from app.memory.session_store import create_session

router = APIRouter()

@router.post("/session/create")
def new_session():
    session_id = create_session()
    return {"session_id": session_id}
