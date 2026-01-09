from fastapi import FastAPI
from app.api.routes import session, chat, summary

app = FastAPI(title="AI Server")

app.include_router(session.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(summary.router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok"}
